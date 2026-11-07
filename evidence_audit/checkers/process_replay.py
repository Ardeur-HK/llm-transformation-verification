"""Replay a selected interval of a controlled FP32 CPU LoRA run.

At the boundary after `start` completed updates, load the adapter, AdamW
state, and Python, NumPy and Torch CPU RNG states. Apply the recorded batches
for updates indexed `[start, end)`, then compare the adapter and optimizer
slots against the end record. The historical policy identifier is
`bitwise_fp32_v0`; the implementation uses exact `torch.equal` comparisons.
Losses, restoration checks and step-zero buffer comparisons are diagnostics.

Evidence interface (files read from `run_dir`):
    run_metadata.json
    train_log.json
    batches.safetensors
    boundaries/step{start:04d}/{adapter.safetensors, optimizer.json,
        optimizer.safetensors, rng.json, state.json}
    boundaries/step{end:04d}/{same}
    boundaries/step0000/{buffers.safetensors, state.json}

The verdict concerns the selected transition under the supplied evidence.
Manifest verification and evidence authenticity are outside its scope.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from evidence_audit.evidence import recorder

try:
    import resource
except ImportError:  # Unavailable on Windows.
    resource = None

SUPPORTED = "supported"
REJECTED = "rejected"

TOLERANCE_POLICY = "bitwise_fp32_v0"

REQUESTED_CLAIM = "selected_transition_consistency"
SUPPORTED_CLAIM = (
    "selected_transition_consistent_under_retained_evidence_and_replay_semantics"
)

DEFAULT_UNVERIFIED_PARTS = [
    "provider_personhood_or_historical_execution",
    "evidence_authenticity_chain",
    # The replay verifies only the selected interval; transitions outside
    # [start, end) may escape detection.
    "unselected_interval_transitions",
    # Any mutable state not captured at the boundaries (none expected for
    # this fixture) is not verified.
    "uncaptured_mutable_state",
    # Manifest SHA-256 hashes are not checked by this checker.
    "manifest_hash_verification",
]

BOUNDARY_FILES = (
    "adapter.safetensors",
    "optimizer.json",
    "optimizer.safetensors",
    "rng.json",
    "state.json",
)


class ReplayError(Exception):
    """Explicit execution error: evidence missing or inconsistent."""


def _tuplify(obj):
    if isinstance(obj, list):
        return tuple(_tuplify(v) for v in obj)
    return obj


def _split_slot_key(key: str) -> tuple[int, str]:
    param_key, separator, slot = key.partition(".")
    if not separator or not slot or not param_key.startswith("p"):
        raise ReplayError(f"invalid optimizer slot key {key!r}")
    index = param_key[1:]
    if not index.isascii() or not index.isdecimal():
        raise ReplayError(f"invalid optimizer slot key {key!r}")
    return int(index), slot


def _boundary_dir(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / "boundaries" / f"step{step:04d}"


def _load_interval_evidence(run_dir: Path, start_step: int, end_step: int) -> dict:
    """Load and structurally validate everything the replay needs."""
    run_dir = Path(run_dir)
    problems = []
    for name in ("run_metadata.json", "train_log.json", "batches.safetensors"):
        if not (run_dir / name).is_file():
            problems.append(name)
    for step in (start_step, end_step):
        bdir = _boundary_dir(run_dir, step)
        for name in BOUNDARY_FILES:
            if not (bdir / name).is_file():
                problems.append(f"boundaries/step{step:04d}/{name}")
    step0 = _boundary_dir(run_dir, 0)
    for name in ("buffers.safetensors", "state.json"):
        if not (step0 / name).is_file():
            problems.append(f"boundaries/step0000/{name}")
    if problems:
        raise ReplayError(f"missing evidence: {', '.join(sorted(problems))}")

    metadata = recorder.load_json(run_dir / "run_metadata.json")
    batches = safetensors.torch.load_file(
        str(run_dir / "batches.safetensors"), device="cpu"
    )
    train_log = recorder.load_json(run_dir / "train_log.json")["train_log"]
    start = recorder.load_boundary(_boundary_dir(run_dir, start_step))
    end = recorder.load_boundary(_boundary_dir(run_dir, end_step))
    buffers0 = safetensors.torch.load_file(
        str(_boundary_dir(run_dir, 0) / "buffers.safetensors"), device="cpu"
    )

    if start["state"].get("global_step") != start_step:
        raise ReplayError(
            f"start boundary global_step {start['state'].get('global_step')} "
            f"!= {start_step}"
        )
    if end["state"].get("global_step") != end_step:
        raise ReplayError(
            f"end boundary global_step {end['state'].get('global_step')} != {end_step}"
        )
    if start["optimizer_slots"] is None or end["optimizer_slots"] is None:
        raise ReplayError("optimizer slots missing at interval boundary")

    hp = start["state"]["hyperparameters"]
    if end["state"]["hyperparameters"] != hp:
        raise ReplayError("start/end boundary hyperparameters differ")
    if hp["device"] != "cpu":
        raise ReplayError(f"recorded device {hp['device']!r} is not cpu")
    if hp["dtype"] != "float32":
        raise ReplayError(f"recorded dtype {hp['dtype']!r} is not float32")
    if start_step <= 0 or end_step <= start_step or end_step > hp["steps"]:
        raise ReplayError(
            f"interval [{start_step},{end_step}) outside 1..{hp['steps']}"
        )

    ids = batches["input_ids"]
    if ids.shape != (hp["steps"] * hp["batch_size"], hp["seq_len"]):
        raise ReplayError(
            f"batches input_ids shape {tuple(ids.shape)} != "
            f"({hp['steps'] * hp['batch_size']}, {hp['seq_len']})"
        )
    for key in ("attention_mask", "labels"):
        if batches[key].shape != ids.shape:
            raise ReplayError(f"batches {key} shape != input_ids shape")
    if len(train_log) != hp["steps"]:
        raise ReplayError(f"train_log has {len(train_log)} entries, want {hp['steps']}")

    return {
        "metadata": metadata,
        "batches": batches,
        "train_log": train_log,
        "start": start,
        "end": end,
        "buffers0": buffers0,
        "hyperparameters": hp,
    }


def _validate_optimizer_meta(hp: dict, opt_meta: dict) -> None:
    """Recorded optimizer meta must be consistent with recorded hyperparameters."""
    if opt_meta["optimizer_class"] != "AdamW":
        raise ReplayError(
            f"optimizer class {opt_meta['optimizer_class']!r} is not AdamW"
        )
    recorded = hp["optimizer"]
    for gi, group in enumerate(opt_meta["param_groups"]):
        for field in ("lr", "betas", "eps", "weight_decay"):
            if group[field] != recorded[field]:
                raise ReplayError(
                    f"optimizer meta group {gi} {field}={group[field]!r} "
                    f"contradicts recorded hyperparameters {recorded[field]!r}"
                )


def _build_model(base_snapshot: Path, hp: dict):
    if not base_snapshot.is_dir():
        raise ReplayError(
            f"local base model snapshot is not a directory: {base_snapshot}"
        )

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    lora = hp["lora"]
    model = AutoModelForCausalLM.from_pretrained(
        str(base_snapshot),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        target_modules=list(lora["target_modules"]),
        bias=lora["bias"],
        init_lora_weights=lora["init_lora_weights"],
    )
    return get_peft_model(model, lora_config)


def _restore_rng(rng: dict) -> None:
    """Restore Python/NumPy/Torch CPU RNG from the boundary record."""
    if rng.get("torch_cuda"):
        raise ReplayError("recorded CUDA RNG state on a cpu-only run")
    random.setstate(_tuplify(rng["python"]))
    n = rng["numpy"]
    np.random.set_state(
        (n[0], np.array(n[1], dtype=np.uint32), int(n[2]), bool(n[3]), float(n[4]))
    )
    torch.set_rng_state(torch.tensor(rng["torch_cpu"], dtype=torch.uint8))


def _verify_rng_restored(rng: dict) -> bool:
    ok = random.getstate() == _tuplify(rng["python"])
    cur = np.random.get_state()
    n = rng["numpy"]
    ok &= bool(
        cur[0] == n[0]
        and np.array_equal(cur[1], np.array(n[1], dtype=np.uint32))
        and int(cur[2]) == int(n[2])
        and bool(cur[3]) == bool(n[3])
        and float(cur[4]) == float(n[4])
    )
    ok &= torch.get_rng_state().tolist() == rng["torch_cpu"]
    return bool(ok)


def _restore_adapter(model, adapter: dict[str, torch.Tensor]) -> None:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    _, unexpected = set_peft_model_state_dict(model, adapter)
    # Missing frozen weights are expected in an adapter-only state dict.
    if unexpected:
        raise ReplayError(f"adapter restore failed: unexpected={unexpected}")
    live = get_peft_model_state_dict(model)
    if set(live) != set(adapter):
        raise ReplayError("adapter key set differs after restore")
    for name, t in adapter.items():
        if not torch.equal(live[name], t):
            raise ReplayError(f"adapter tensor {name} differs after restore")


def _restore_optimizer(
    model, opt, opt_meta: dict, slots: dict[str, torch.Tensor]
) -> None:
    """Restore AdamW param groups and state by manual assignment.

    The recorded slot keys are `p<param_index>.<slot_name>` with the index
    referring to `group_param_names` order (params appear in group order).
    State is assigned directly (not via `load_state_dict`) so the recorded
    tensors are restored exactly, with the parameter order validated first.
    """
    trainables = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, _p in model.named_parameters() if _p.requires_grad]
    expected = [
        n
        for gi in sorted(opt_meta["group_param_names"], key=int)
        for n in opt_meta["group_param_names"][gi]
    ]
    if expected != names:
        raise ReplayError(
            "trainable parameter names/order do not match recorded grouping"
        )
    if opt_meta["param_count"] != len(trainables):
        raise ReplayError(
            f"param_count {opt_meta['param_count']} != trainable {len(trainables)}"
        )
    if len(opt.param_groups) != len(opt_meta["param_groups"]):
        raise ReplayError("parameter group count mismatch")

    for gi, group in enumerate(opt_meta["param_groups"]):
        for k, v in group.items():
            opt.param_groups[gi][k] = v
    state_by_idx: dict[int, dict[str, torch.Tensor]] = {}
    for key, t in slots.items():
        idx, slot = _split_slot_key(key)
        state_by_idx.setdefault(idx, {})[slot] = t
    for idx, slot_dict in state_by_idx.items():
        if idx >= len(trainables):
            raise ReplayError(f"slot param index {idx} out of range")
        opt.state[trainables[idx]] = slot_dict


def _capture_slots(opt) -> dict[str, torch.Tensor]:
    """Slot tensors keyed `p<param_index>.<slot_name>`."""
    sd = opt.state_dict()
    slots: dict[str, torch.Tensor] = {}
    for idx_str, st in sd["state"].items():
        for k, t in st.items():
            slots[f"p{int(idx_str)}.{k}"] = t.detach().cpu().clone()
    return slots


def _tensor_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    if set(a) != set(b):
        return False, sorted(set(a) ^ set(b))
    mismatched = [n for n in a if not torch.equal(a[n], b[n])]
    return not mismatched, mismatched


def _run_updates(model, opt, batches: dict, hp: dict, start_step: int, end_step: int):
    """Execute updates start..end-1 (training steps start+1..end)."""
    losses: dict[str, str] = {}
    model.train()
    for step in range(start_step + 1, end_step + 1):
        rows = slice(hp["batch_size"] * (step - 1), hp["batch_size"] * step)
        ids = batches["input_ids"][rows]
        mask = batches["attention_mask"][rows]
        labels = batches["labels"][rows]
        out = model(input_ids=ids, attention_mask=mask)
        loss = torch.nn.functional.cross_entropy(
            out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del out
        losses[str(step)] = float(loss.detach()).hex()
    return losses


def replay_interval(
    run_dir: str | Path,
    start_step: int = 8,
    end_step: int = 12,
    verbose: bool = False,
    *,
    base_snapshot: str | Path | None = None,
) -> dict:
    """Replay `[start_step, end_step)` using trusted local evidence.

    Pass `base_snapshot` when the metadata path needs to be relocated.
    Otherwise, the original metadata path is used relative to the working
    directory. The snapshot must contain local safetensors weights.
    This operation resets the process-global CPU RNG streams.
    """
    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    updates = list(range(start_step, end_step))
    steps = list(range(start_step + 1, end_step + 1))
    base = {
        "checker": "selective_process_replay",
        "tolerance_policy": TOLERANCE_POLICY,
        "requested_claim": REQUESTED_CLAIM,
        "supported_claim": SUPPORTED_CLAIM,
        "interval": {
            "start_step": start_step,
            "end_step": end_step,
            "updates": updates,
            "steps": steps,
            "batch_rows": None,  # filled from recorded hyperparameters below
        },
        "unverified_parts": DEFAULT_UNVERIFIED_PARTS,
    }

    def finish(status: str, extra: dict) -> dict:
        verdict = dict(base)
        verdict.update(extra)
        verdict["status"] = status
        verdict["execution_status"] = "error" if status == "error" else "ok"
        peak_rss_kb = None
        if resource is not None:
            peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                peak_rss_kb //= 1024
        verdict["metrics"] = {
            "wall_clock_s": round(time.perf_counter() - t0, 3),
            "peak_rss_kb": peak_rss_kb,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        }
        return verdict

    try:
        ev = _load_interval_evidence(run_dir, start_step, end_step)
        hp = ev["hyperparameters"]
        base_model = ev["metadata"]["base_model"]
        snapshot = Path(
            base_model["snapshot_path"] if base_snapshot is None else base_snapshot
        )
        base["interval"]["batch_rows"] = [
            hp["batch_size"] * start_step,
            hp["batch_size"] * end_step,
        ]
        base["interval"]["interval_label"] = f"[{start_step},{end_step})"
        base["evidence"] = {
            "run_dir": str(run_dir),
            "start_boundary": str(_boundary_dir(run_dir, start_step)),
            "end_boundary": str(_boundary_dir(run_dir, end_step)),
            "base_model": base_model["name"],
            "base_revision": base_model["revision"],
        }

        _validate_optimizer_meta(hp, ev["start"]["optimizer_meta"])
        if (
            ev["end"]["optimizer_meta"]["param_groups"]
            != ev["start"]["optimizer_meta"]["param_groups"]
        ):
            raise ReplayError("start/end optimizer param_groups differ")

        if verbose:
            print("[replay] building fresh base model + LoRA", flush=True)
        model = _build_model(snapshot, hp)

        buffers = {n: b for n, b in model.named_buffers()}
        buffers_ok, buffers_diff = _tensor_dicts_equal(buffers, ev["buffers0"])

        _restore_adapter(model, ev["start"]["adapter"])
        _restore_rng(ev["start"]["rng"])
        rng_ok = _verify_rng_restored(ev["start"]["rng"])

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1.0
        )
        _restore_optimizer(
            model, opt, ev["start"]["optimizer_meta"], ev["start"]["optimizer_slots"]
        )
        slots_ok, slots_diff = _tensor_dicts_equal(
            _capture_slots(opt), ev["start"]["optimizer_slots"]
        )

        if verbose:
            print("[replay] executing updates", updates, flush=True)
        losses = _run_updates(model, opt, ev["batches"], hp, start_step, end_step)

        from peft import get_peft_model_state_dict

        end_adapter = get_peft_model_state_dict(model)
        end_slots = _capture_slots(opt)

        adapter_eq, adapter_bad = _tensor_dicts_equal(end_adapter, ev["end"]["adapter"])
        slots_eq, slots_bad = _tensor_dicts_equal(
            end_slots, ev["end"]["optimizer_slots"]
        )
        log_by_step = {str(r["step"]): r["loss_hex"] for r in ev["train_log"]}
        loss_matches = {s: losses[s] == log_by_step.get(s) for s in losses}
        end_loss_equal = losses[str(end_step)] == ev["end"]["state"]["loss_hex"]

        comparison = {
            "adapter_tensors": len(end_adapter),
            "adapter_equal": bool(adapter_eq),
            "adapter_mismatched": adapter_bad,
            "optimizer_slot_tensors": len(end_slots),
            "optimizer_slots_equal": bool(slots_eq),
            "optimizer_slot_mismatched": slots_bad,
            "end_loss_equal": bool(end_loss_equal),
            "replay_losses": losses,
            "replay_losses_match_train_log": bool(all(loss_matches.values())),
        }
        restore = {
            "buffers_match_step0": bool(buffers_ok),
            "buffer_mismatched": buffers_diff,
            "rng_restored": bool(rng_ok),
            "optimizer_slots_match_start": bool(slots_ok),
            "optimizer_slot_restore_mismatched": slots_diff,
        }
        verdict = finish(
            SUPPORTED if (adapter_eq and slots_eq) else REJECTED,
            {
                "restore": restore,
                "comparison": comparison,
                "coverage": {
                    "verified_interval": f"[{start_step},{end_step})",
                    "unverified_intervals": [
                        f"[0,{start_step})",
                        f"[{end_step},{hp['steps']})",
                    ],
                },
            },
        )
        del model, opt
        return verdict
    except ReplayError as exc:
        return finish("error", {"error": str(exc), "comparison": None})
    except Exception as exc:  # noqa: BLE001 — never map execution failure to supported
        return finish(
            "error",
            {"error": f"{type(exc).__name__}: {exc}", "comparison": None},
        )
