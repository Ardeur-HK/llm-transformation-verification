"""Verify a declared float32 linear-merge endpoint.

The checker recomputes the delivered source tensors in manifest order
using sequential float32 accumulation. If normalization is enabled, it
divides the weighted tensor sum by the float32 sum of the weights.

Normalization after accumulation follows the order of operations used by
MergeKit 0.1.4. MergeKit uses a stacked reduction, which can round
differently from sequential accumulation. The checker uses exact tensor
equality under torch.equal for the legacy bitwise_fp32_v0 policy.

Pass a manifest dict or load one from JSON using load_manifest.

    {
      "sources": [
        {"path": "source_a", "weight": 0.75},
        {"path": "source_b", "weight": 0.25}
      ],
      "target_path": "target",
      "merge_method": "linear",
      "normalize": true,
      "dtype": "float32"
    }

Source and target directories must contain config.json and Safetensors
shards. The result describes consistency of the delivered evidence.
"""

from __future__ import annotations

import json
import time
from os import PathLike
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

SUPPORTED = "supported"
REJECTED = "rejected"
INCONCLUSIVE = "inconclusive"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

SUPPORTED_MERGE_METHOD = "linear"
SUPPORTED_DTYPE = "float32"
TOLERANCE_POLICY = "bitwise_fp32_v0"

REQUESTED_CLAIM = "declared_linear_merge_endpoint"
SUPPORTED_CLAIM = (
    "delivered_target_satisfies_declared_source_recipe_target_endpoint_relation"
)

DEFAULT_UNVERIFIED_PARTS = [
    "provider_personhood_or_historical_execution",
    "evidence_authenticity_chain",
    # v0 compares config vocab_size / tie flags; tokenizer files themselves
    # are not diffed.
    "tokenizer_file_identity",
    # v0 is validated on untied fixtures; tied-weight merge semantics are
    # checked only as flag/value consistency, not full transform semantics.
    "tied_weight_transform_semantics",
]

_IDENTITY_FIELDS = ("model_type", "architectures", "vocab_size", "tie_word_embeddings")


class MissingEvidence(Exception):
    """A required manifest field or delivered artifact is absent."""


class CheckerExecutionError(Exception):
    """A delivered artifact exists but could not be processed."""


def load_manifest(path: str | Path) -> dict:
    """Load a manifest dict from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise MissingEvidence("manifest must be an object")
    problems: list[str] = []
    for field in ("sources", "target_path", "merge_method", "normalize", "dtype"):
        if field not in manifest or manifest[field] in (None, "", []):
            problems.append(field)
    if "normalize" in manifest and not isinstance(manifest["normalize"], bool):
        problems.append("normalize (not a boolean)")
    if "target_path" in manifest and not isinstance(
        manifest["target_path"], (str, PathLike)
    ):
        problems.append("target_path (not a path)")
    sources = manifest.get("sources")
    if isinstance(sources, tuple) and not sources:
        problems.append("sources")
    elif sources is not None and not isinstance(sources, (list, tuple)):
        problems.append("sources (not an array)")
    for i, source in enumerate(_manifest_sources(manifest)):
        if not isinstance(source, dict):
            problems.append(f"sources[{i}] (not an object)")
            continue
        for field in ("path", "weight"):
            if field not in source or source[field] in (None, ""):
                problems.append(f"sources[{i}].{field}")
        if "path" in source and not isinstance(source["path"], (str, PathLike)):
            problems.append(f"sources[{i}].path (not a path)")
        if "weight" in source and not isinstance(source["weight"], (int, float)):
            problems.append(f"sources[{i}].weight (not a number)")
    if problems:
        raise MissingEvidence(
            "missing or invalid required manifest field(s): " + ", ".join(problems)
        )


def _manifest_sources(manifest: dict) -> list | tuple:
    sources = manifest.get("sources")
    return sources if isinstance(sources, (list, tuple)) else []


def _load_model_dir(path: Path, kind: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return (config_json, tensors) for a delivered model directory."""
    if not path.is_dir():
        raise MissingEvidence(f"{kind} directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise MissingEvidence(f"{kind} is missing config.json: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckerExecutionError(
            f"{kind} config.json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise CheckerExecutionError(f"{kind} config.json must be an object")
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise MissingEvidence(f"{kind} has no *.safetensors shards: {path}")
    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        try:
            shard_tensors = safetensors.torch.load_file(str(shard), device="cpu")
        except Exception as exc:  # corrupt/foreign file: execution failure
            raise CheckerExecutionError(
                f"could not load {kind} shard {shard.name}: {exc}"
            ) from exc
        overlap = set(tensors) & set(shard_tensors)
        if overlap:
            raise CheckerExecutionError(
                f"{kind} has duplicate tensor keys across shards: {sorted(overlap)}"
            )
        tensors.update(shard_tensors)
    return config, tensors


def _config_identity(config: dict, kind: str) -> tuple:
    """Identity tuple used to enforce the v0 same-architecture restriction."""
    for field in _IDENTITY_FIELDS:
        if field not in config:
            raise MissingEvidence(f"{kind} config.json is missing field: {field}")
    return tuple(
        json.dumps(config[field], sort_keys=True) for field in _IDENTITY_FIELDS
    )


def _relative_l2(
    expected: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> float:
    """||target - expected||_2 / ||expected||_2 over common, comparable keys."""
    e_parts: list[torch.Tensor] = []
    t_parts: list[torch.Tensor] = []
    for key, exp in expected.items():
        got = target.get(key)
        if got is None or got.shape != exp.shape or got.dtype != torch.float32:
            continue
        e_parts.append(exp.detach().flatten())
        t_parts.append(got.detach().flatten())
    if not e_parts:
        return 0.0
    e_cat = torch.cat(e_parts)
    t_cat = torch.cat(t_parts)
    denom = float(e_cat.norm(2))
    if denom == 0.0:
        return 0.0
    return float((t_cat - e_cat).norm(2)) / denom


def _evidence_bytes_read(manifest: dict) -> int:
    """Total bytes of delivered artifacts present on disk (post-hoc audit metric)."""
    total = 0
    paths = [
        source.get("path")
        for source in _manifest_sources(manifest)
        if isinstance(source, dict)
    ]
    paths.append(manifest.get("target_path"))
    for entry in paths:
        if not entry or not isinstance(entry, (str, PathLike)):
            continue
        try:
            path = Path(entry)
            if not path.is_dir():
                continue
            config = path / "config.json"
            if config.is_file():
                total += config.stat().st_size
            for shard in path.glob("*.safetensors"):
                total += shard.stat().st_size
        except (OSError, TypeError, ValueError):
            continue
    return total


def _evaluate(manifest: dict) -> tuple[str, list[str], dict | None]:
    """Run the checker; returns (verdict, reasons, mismatch_report)."""
    _validate_manifest(manifest)
    merge_method = manifest["merge_method"]
    dtype = manifest["dtype"]
    normalize = manifest["normalize"]
    sources = manifest["sources"]
    target_path = Path(manifest["target_path"])
    weights = [float(s["weight"]) for s in sources]

    scope_reasons: list[str] = []
    if merge_method != SUPPORTED_MERGE_METHOD:
        scope_reasons.append(
            f"merge_method {merge_method!r} outside v0 scope (only 'linear')"
        )
    if dtype != SUPPORTED_DTYPE:
        scope_reasons.append(f"dtype {dtype!r} outside v0 scope (only 'float32')")

    # Load delivered artifacts. Missing anything -> insufficient evidence.
    source_artifacts = [
        _load_model_dir(Path(s["path"]), f"source[{i}]") for i, s in enumerate(sources)
    ]
    source_configs = [cfg for cfg, _ in source_artifacts]
    source_tensors = [t for _, t in source_artifacts]
    target_config, target_tensors = _load_model_dir(target_path, "target")

    # Source-side v0 scope checks: same architecture / vocab / tie settings,
    # same tensor keys, same shapes, float32 tensors.
    identities = [
        _config_identity(cfg, f"source[{i}]") for i, cfg in enumerate(source_configs)
    ]
    if len(set(identities)) != 1:
        scope_reasons.append(
            "sources disagree on model_type/architectures/vocab_size/"
            "tie_word_embeddings (cross-architecture or vocab-union merge "
            "outside v0 scope)"
        )
    key_sets = [set(t) for t in source_tensors]
    if len({frozenset(keys) for keys in key_sets}) != 1:
        scope_reasons.append(
            "source tensor key sets differ (key surgery / vocabulary union "
            "outside v0 scope)"
        )
    for i, (cfg, tensors) in enumerate(zip(source_configs, source_tensors)):
        cfg_dtype = cfg.get("dtype")
        if cfg_dtype is not None and cfg_dtype != dtype:
            scope_reasons.append(
                f"source[{i}] config dtype {cfg_dtype!r} contradicts recipe "
                f"dtype {dtype!r}"
            )
        non_f32 = sorted(k for k, t in tensors.items() if t.dtype != torch.float32)
        if non_f32:
            scope_reasons.append(
                f"source[{i}] has non-float32 tensors {non_f32} (outside v0 scope)"
            )
    common_keys = sorted(key_sets[0])
    for key in common_keys:
        shapes = {t[key].shape for t in source_tensors if key in t}
        if len(shapes) != 1:
            scope_reasons.append(
                f"source tensor shapes differ for {key!r} (outside v0 scope)"
            )

    if scope_reasons:
        return INCONCLUSIVE, scope_reasons, None

    # Target metadata must match the declared source identity.
    target_identity = _config_identity(target_config, "target")
    if target_identity != identities[0]:
        return (
            REJECTED,
            [
                "target config.json contradicts the declared sources on "
                "model_type/architectures/vocab_size/tie_word_embeddings: "
                f"target={target_identity}, sources={identities[0]}"
            ],
            None,
        )

    # Recompute the declared relation (independent of MergeKit).
    w = torch.tensor(weights, dtype=torch.float32)
    sum_w = w.sum()
    if normalize and float(sum_w) == 0.0:
        return (
            INCONCLUSIVE,
            ["normalize=true with total declared weight 0 is degenerate"],
            None,
        )
    expected: dict[str, torch.Tensor] = {}
    for key in common_keys:
        acc = w[0] * source_tensors[0][key]
        for i in range(1, len(sources)):
            acc = acc + w[i] * source_tensors[i][key]
        if normalize:
            acc = acc / sum_w
        expected[key] = acc

    # Compare target vs recomputation; report per-tensor and global stats.
    missing_keys = sorted(set(expected) - set(target_tensors))
    extra_keys = sorted(set(target_tensors) - set(expected))
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    tied_weight_issues: list[str] = []
    failing_tensors: list[str] = []
    per_tensor_max_abs: dict[str, float] = {}

    for key, exp in expected.items():
        got = target_tensors.get(key)
        if got is None:
            continue  # counted in missing_keys
        if got.shape != exp.shape:
            shape_mismatches.append(key)
            continue
        if got.dtype != torch.float32:
            dtype_mismatches.append(key)
            continue
        dmax = float((got - exp).abs().max())
        per_tensor_max_abs[key] = dmax
        if not torch.equal(got, exp):
            failing_tensors.append(key)

    if target_config.get("tie_word_embeddings"):
        wte = target_tensors.get("transformer.wte.weight")
        lm_head = target_tensors.get("lm_head.weight")
        if wte is not None and lm_head is not None and not torch.equal(wte, lm_head):
            tied_weight_issues.append(
                "target declares tie_word_embeddings=true but its wte and "
                "lm_head tensors differ"
            )

    elements_checked = sum(expected[key].numel() for key in per_tensor_max_abs)
    report = {
        "tensors_checked": len(per_tensor_max_abs),
        "elements_checked": elements_checked,
        "bitwise_identical": len(per_tensor_max_abs) - len(failing_tensors),
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "tied_weight_issues": tied_weight_issues,
        "failing_tensors": failing_tensors,
        "max_abs_error": max(per_tensor_max_abs.values(), default=0.0),
        "relative_l2": _relative_l2(expected, target_tensors),
        "per_tensor_max_abs_error": per_tensor_max_abs,
    }

    problems: list[str] = []
    if missing_keys:
        problems.append(f"missing tensor keys in target: {missing_keys}")
    if extra_keys:
        problems.append(f"extra tensor keys in target: {extra_keys}")
    if shape_mismatches:
        problems.append(f"shape mismatch for keys: {shape_mismatches}")
    if dtype_mismatches:
        problems.append(f"dtype mismatch (not float32) for keys: {dtype_mismatches}")
    if tied_weight_issues:
        problems.extend(tied_weight_issues)
    if failing_tensors:
        problems.append(
            f"{len(failing_tensors)}/{len(per_tensor_max_abs)} tensor(s) differ "
            f"from recomputation: {failing_tensors}"
        )

    if problems:
        return REJECTED, problems, report
    return SUPPORTED, ["target matches recomputation bitwise"], report


def check_endpoint(manifest: dict) -> dict:
    """Audit a declared linear-merge endpoint against delivered artifacts.

    Returns a result dict with the verdict, execution status, scope,
    limitations, and any computed comparison metrics.
    """
    t0 = time.perf_counter()
    fields = manifest if isinstance(manifest, dict) else {}
    sources = _manifest_sources(fields)
    result: dict[str, Any] = {
        "requested_claim": REQUESTED_CLAIM,
        "audit_method": "direct_endpoint_recompute",
        "evidence_used": [
            str(s["path"]) if isinstance(s, dict) and s.get("path") else "<unset>"
            for s in sources
        ]
        + [str(fields.get("target_path", "<unset>"))],
        "supported_claim": SUPPORTED_CLAIM,
        "unverified_parts": list(DEFAULT_UNVERIFIED_PARTS),
        "scope": {
            "merge_method": fields.get("merge_method"),
            "dtype": fields.get("dtype"),
            "normalize": fields.get("normalize"),
            "n_sources": len(sources),
        },
        "tolerance_policy": TOLERANCE_POLICY,
    }

    verdict: str
    reasons: list[str]
    report: dict | None
    execution_status = "ok"
    try:
        verdict, reasons, report = _evaluate(manifest)
    except MissingEvidence as exc:
        verdict, reasons, report = INSUFFICIENT_EVIDENCE, [str(exc)], None
    except CheckerExecutionError as exc:
        verdict, reasons, report, execution_status = (
            INCONCLUSIVE,
            [str(exc)],
            None,
            "error",
        )
    except Exception as exc:  # unexpected: never map to supported
        verdict, reasons, report, execution_status = (
            INCONCLUSIVE,
            [f"unexpected execution failure: {type(exc).__name__}: {exc}"],
            None,
            "error",
        )

    result.update(
        {
            "verdict": verdict,
            "reasons": reasons,
            "mismatch_report": report,
            "metrics": {
                "wall_clock_seconds": round(time.perf_counter() - t0, 6),
                "device": "cpu",
                "tensors_checked": (report or {}).get("tensors_checked", 0),
                "elements_checked": (report or {}).get("elements_checked", 0),
                "evidence_bytes_read": _evidence_bytes_read(fields),
            },
            "execution_status": execution_status,
        }
    )
    return result
