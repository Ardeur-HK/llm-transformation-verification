"""Record and replay a tiny local LoRA run using generated tokens and weights."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from peft import get_peft_model_state_dict
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.utils.logging import disable_progress_bar

from evidence_audit.checkers import process_replay
from evidence_audit.evidence import recorder

START_STEP = 1
END_STEP = 4
STEPS = 5
BATCH_SIZE = 2
SEQ_LEN = 8
VOCAB_SIZE = 32
SEED = 31


def _record_boundary(run_dir, model, optimizer, hyperparameters, step, loss_hex):
    optimizer_state = optimizer.state_dict()
    trainable_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    groups = optimizer_state["param_groups"]
    slots = process_replay._capture_slots(optimizer)
    recorder.save_boundary(
        run_dir / "boundaries" / f"step{step:04d}",
        adapter=get_peft_model_state_dict(model),
        optimizer_meta={
            "optimizer_class": "AdamW",
            "param_groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in groups
            ],
            "group_param_names": {
                str(index): [trainable_names[i] for i in group["params"]]
                for index, group in enumerate(groups)
            },
            "param_count": len(trainable_names),
            "slot_shapes": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in slots.items()
            },
            "has_slots": bool(slots),
        },
        optimizer_slots=slots or None,
        rng=recorder.serialize_rng_state(include_cuda=False),
        state={
            "global_step": step,
            "boundary_semantics": "state after the stated number of completed updates",
            "loss": None if loss_hex is None else float.fromhex(loss_hex),
            "loss_hex": loss_hex,
            "hyperparameters": hyperparameters,
        },
        buffers=dict(model.named_buffers()) if step == 0 else None,
    )


def record_fixture(run_dir: Path) -> None:
    """Write a fresh local model and a complete, five-update replay record."""
    run_dir.mkdir()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.random.default_generator.manual_seed(SEED)

    snapshot = run_dir / "base"
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=SEQ_LEN,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen2ForCausalLM(config).save_pretrained(snapshot)

    optimizer_settings = {
        "lr": 0.003,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    hyperparameters = {
        "device": "cpu",
        "dtype": "float32",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "lora": {
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.2,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "init_lora_weights": True,
        },
        "optimizer": optimizer_settings,
    }
    model = process_replay._build_model(snapshot, hyperparameters)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        **optimizer_settings,
    )
    model.train()

    generator = torch.Generator().manual_seed(SEED + 1)
    ids = torch.randint(
        3, VOCAB_SIZE, (STEPS * BATCH_SIZE, SEQ_LEN), generator=generator
    )
    ids[1::2, -1] = 0
    mask = ids.ne(0).long()
    labels = ids.masked_fill(mask == 0, -100)
    safetensors.torch.save_file(
        {"input_ids": ids, "attention_mask": mask, "labels": labels},
        str(run_dir / "batches.safetensors"),
    )

    _record_boundary(run_dir, model, optimizer, hyperparameters, 0, None)
    train_log = []
    for update in range(STEPS):
        rows = slice(update * BATCH_SIZE, (update + 1) * BATCH_SIZE)
        logits = model(input_ids=ids[rows], attention_mask=mask[rows]).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1), labels[rows, 1:].flatten(), ignore_index=-100
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step = update + 1
        value = float(loss.detach())
        train_log.append({"step": step, "loss": value, "loss_hex": value.hex()})
        if step in (START_STEP, END_STEP):
            _record_boundary(
                run_dir, model, optimizer, hyperparameters, step, value.hex()
            )

    recorder.save_json(run_dir / "train_log.json", {"train_log": train_log})
    recorder.save_json(
        run_dir / "run_metadata.json",
        {
            "base_model": {
                "name": "synthetic-qwen2",
                "revision": "seed-31",
                "snapshot_path": str(snapshot.resolve()),
            },
            "hyperparameters": hyperparameters,
        },
    )
    recorder.write_manifest(run_dir, [0, START_STEP, END_STEP])


def main() -> None:
    disable_progress_bar()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        record_fixture(run_dir)
        honest = process_replay.replay_interval(run_dir, START_STEP, END_STEP)
        if honest["status"] != process_replay.SUPPORTED:
            raise SystemExit(f"Honest replay failed: {honest}")
        comparison = honest["comparison"]
        print(
            f"Honest replay [{START_STEP},{END_STEP}) -> supported "
            f"({comparison['adapter_tensors']} adapter tensors, "
            f"{comparison['optimizer_slot_tensors']} optimizer slots match)"
        )

        end_adapter = (
            run_dir / "boundaries" / f"step{END_STEP:04d}" / "adapter.safetensors"
        )
        tensors = safetensors.torch.load_file(str(end_adapter))
        tensors[min(tensors)][0, 0] += 0.01
        safetensors.torch.save_file(tensors, str(end_adapter))
        tampered = process_replay.replay_interval(run_dir, START_STEP, END_STEP)
        if tampered["status"] != process_replay.REJECTED:
            raise SystemExit(f"Tampered end adapter was not rejected: {tampered}")
        comparison = tampered["comparison"]
        print(
            "Tampered end adapter -> rejected "
            f"({len(comparison['adapter_mismatched'])} adapter tensor differs, "
            f"{len(comparison['optimizer_slot_mismatched'])} optimizer slots differ)"
        )


if __name__ == "__main__":
    main()
