"""Measure representation similarity under a specified set of text probes.

The REEF-style procedure extracts the last real token at each selected
hidden-state index, standardizes each feature across probes, and reports
raw linear CKA per layer. The result contains scores and measurement
conditions. Model lineage and training history are outside this procedure's
scope.

Valid measurements return `measured`. Zero-variance features, a single
probe, or a zero CKA denominator give `inconclusive`. Invalid inputs
and execution failures give `error`. The score has no decision threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import resource
except ImportError:
    resource = None

MEASURED = "measured"
INCONCLUSIVE = "inconclusive"

# Three fixed decoder layers (0-indexed). Hidden-state tuple index = layer + 1.
LAYERS = (8, 14, 20)

EXTRACTION_MODE = "last_token"
STANDARDIZATION = "per_feature_center_std"
KERNEL = "linear"
TOLERANCE_POLICY = "linear_cka_raw_v0"

REQUESTED_CLAIM = "representation_level_relatedness"
MEASURED_CLAIM = (
    "raw_linear_cka_scores_under_specified_probes_and_measurement_procedure"
)

DEFAULT_UNVERIFIED_PARTS = [
    "exact_finetuning_recipe",
    "training_dataset_identity",
    "optimizer_identity",
    "full_process_execution",
    "evidence_authenticity_chain",
    "provider_personhood_or_historical_execution",
]


class RelationError(Exception):
    """Explicit execution error: model/probe evidence missing or unusable."""


class DegenerateInput(Exception):
    """Degenerate representation input (zero variance / zero denominator)."""


def _canonical_rows_bytes(rows: list[dict]) -> bytes:
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_probes(probes_path: str | Path) -> dict:
    """Load a JSON probe set and check its optional rows digest.

    The file contains `n_rows` and a `rows` list whose entries have a
    non-empty `statement`. A sibling `<stem>.digest.json` can specify
    `sha256_rows` for the compact, key-sorted JSON serialization of the
    complete rows. The digest checks consistency with the supplied sidecar.
    """
    probes_path = Path(probes_path)
    try:
        probe_bytes = probes_path.read_bytes()
        raw = json.loads(probe_bytes)
    except (OSError, ValueError) as exc:
        raise RelationError(f"could not read probe file: {exc}") from exc
    if not isinstance(raw, dict):
        raise RelationError("probe file must contain a JSON object")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RelationError(f"probe file has no rows: {probes_path}")
    try:
        n_rows = int(raw["n_rows"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RelationError("probe file must specify n_rows") from exc
    if n_rows != len(rows):
        raise RelationError(f"probe n_rows {n_rows} != len(rows) {len(rows)}")
    statements = []
    for index, row in enumerate(rows):
        statement = row.get("statement") if isinstance(row, dict) else None
        if not isinstance(statement, str) or not statement:
            raise RelationError(f"probe row {index} has no non-empty statement")
        statements.append(statement)

    digest_rows = hashlib.sha256(_canonical_rows_bytes(rows)).hexdigest()
    digest_path = probes_path.with_name(probes_path.stem + ".digest.json")
    if digest_path.is_file():
        try:
            recorded = json.loads(digest_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise RelationError(f"could not read probe digest: {exc}") from exc
        recorded_digest = (
            recorded.get("sha256_rows") if isinstance(recorded, dict) else None
        )
        if recorded_digest != digest_rows:
            raise RelationError(
                f"probe rows digest mismatch: computed {digest_rows} != "
                f"recorded {recorded_digest}"
            )

    meta = {
        "source": raw.get("source"),
        "source_repo": raw.get("source_repo"),
        "source_file": raw.get("source_file"),
        "source_file_sha256": raw.get("source_file_sha256"),
        "total_rows": raw.get("total_rows"),
        "row_selection": raw.get("row_selection"),
        "n_rows": n_rows,
        "probe_file": str(probes_path),
        "probe_file_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "rows_sha256": digest_rows,
    }
    return {"statements": statements, "meta": meta}


def center_standardize(X: np.ndarray, name: str) -> np.ndarray:
    """Per-feature centering + standardization (population std, ddof=0).

    Raises `DegenerateInput` on fewer than two samples or any feature with
    zero standard deviation; no epsilon is added.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise DegenerateInput(f"{name}: expected a 2D matrix, got {X.ndim}D")
    n, p = X.shape
    if n < 2:
        raise DegenerateInput(f"{name}: {n} sample(s), variance undefined (<2)")
    if p < 1:
        raise DegenerateInput(f"{name}: zero features")
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0)  # population std
    zero = np.nonzero(std == 0.0)[0]
    if zero.size:
        raise DegenerateInput(
            f"{name}: {zero.size} zero-variance feature(s), first at "
            f"column {int(zero[0])}"
        )
    return Xc / std[None, :]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA of two per-feature centered+standardized matrices.

    ``X`` and ``Y`` are float64 with the same number of samples.
    Their feature counts may differ. The score is
    ``||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)``. Raises
    `DegenerateInput` on a zero denominator. For ``X is Y`` the numerator and
    denominator are the same value, so the score is exactly 1.0.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise DegenerateInput("linear_cka: inputs must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise DegenerateInput(
            f"linear_cka: sample counts differ ({X.shape[0]} vs {Y.shape[0]})"
        )
    gx = np.linalg.norm(X.T @ X, "fro")
    gy = np.linalg.norm(Y.T @ Y, "fro")
    cross = np.linalg.norm(Y.T @ X, "fro")
    denom = gx * gy
    if denom == 0.0:
        raise DegenerateInput("linear_cka: zero CKA denominator")
    return float((cross * cross) / denom)


def _load_model(model_path: Path, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_path.is_dir():
        raise RelationError(f"model directory does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    )
    return model, tokenizer


def _extract_last_token(model, tokenizer, statements, layers, device, batch_size):
    """Extract last-token hidden states at fixed decoder layers.

    Returns ``{layer: np.ndarray(float64, (n, hidden))}``. The hidden-state
    tuple entry selected for decoder layer `i` is `i + 1` (index 0 is
    embeddings). The model defines the contents of each entry.
    """
    n_layers = int(model.config.num_hidden_layers)
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise RelationError(f"layer {layer} outside 0..{n_layers - 1}")
    hidden_size = int(model.config.hidden_size)
    n = len(statements)
    acc = {layer: np.empty((n, hidden_size), dtype=np.float64) for layer in layers}

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = statements[start : start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"]  # (b, s), right-padded
            last = mask.sum(dim=1) - 1  # last real token per sequence
            if (last < 0).any():
                raise RelationError("empty tokenization produced no last token")
            rows_idx = torch.arange(len(chunk), device=device)
            for layer in layers:
                hs = out.hidden_states[layer + 1]  # (b, s, hidden)
                sel = hs[rows_idx, last, :]  # (b, hidden)
                acc[layer][start : start + len(chunk)] = sel.float().cpu().numpy()
            del out
    return acc


def _evidence_bytes_read(model_paths: list[Path], probes_path: Path) -> int:
    """Estimate evidence size from file listings for the supplied paths."""
    try:
        total = probes_path.stat().st_size if probes_path.is_file() else 0
    except OSError:
        total = 0
    for mp in model_paths:
        if not mp.is_dir():
            continue
        try:
            for child in mp.iterdir():
                if child.is_file():
                    total += child.stat().st_size
        except OSError:
            continue
    return total


def _peak_rss_kb() -> int | None:
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == "darwin" else peak


def check_relation(
    model_a: str | Path,
    model_b: str | Path,
    probes_path: str | Path,
    layers: tuple[int, ...] = LAYERS,
    device: str = "cuda",
    batch_size: int = 64,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> dict:
    """Measure linear-CKA representation relatedness between two checkpoints.

    Supply local Transformers checkpoint and tokenizer directories. For a
    self comparison, the same path reuses one extraction. Otherwise each
    checkpoint is extracted independently with its tokenizer.
    """
    t0 = time.perf_counter()
    model_a = Path(model_a)
    model_b = Path(model_b)
    probes_path = Path(probes_path)
    n_probes = 0

    base = {
        "requested_claim": REQUESTED_CLAIM,
        "audit_method": "reef_style_last_token_linear_cka",
        "evidence_used": [str(model_a), str(model_b), str(probes_path)],
        "measured_claim": MEASURED_CLAIM,
        "unverified_parts": list(DEFAULT_UNVERIFIED_PARTS),
        "scope": {
            "layers": list(layers),
            "extraction_mode": EXTRACTION_MODE,
            "standardization": STANDARDIZATION,
            "kernel": KERNEL,
            "model_dtype": str(dtype).replace("torch.", ""),
        },
        "tolerance_policy": TOLERANCE_POLICY,
    }

    def finish(status: str, extra: dict) -> dict:
        result = dict(base)
        result.update(extra)
        result["status"] = status
        result["execution_status"] = "error" if status == "error" else "ok"
        result["metrics"] = {
            "wall_clock_seconds": round(time.perf_counter() - t0, 3),
            "device": device,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "evidence_bytes_read": _evidence_bytes_read(
                [model_a, model_b], probes_path
            ),
            "n_probes": n_probes,
            "n_layers": len(layers),
            "peak_rss_kb": _peak_rss_kb(),
        }
        return result

    try:
        if not layers:
            raise RelationError("select at least one layer")
        try:
            batch_size = operator.index(batch_size)
        except TypeError as exc:
            raise RelationError("batch_size must be a positive integer") from exc
        if batch_size < 1:
            raise RelationError("batch_size must be a positive integer")
        base["scope"]["batch_size"] = batch_size
        probes = load_probes(probes_path)
        statements = probes["statements"]
        n_probes = len(statements)
        base["scope"]["n_probes"] = n_probes
        base["evidence"] = {"probe_meta": probes["meta"]}

        model, tokenizer = _load_model(model_a, dtype)
        model.to(device)
        if verbose:
            print(
                f"[relation] extracting model_a={model_a.name} on "
                f"{len(statements)} probes, layers={list(layers)}",
                flush=True,
            )
        X = _extract_last_token(
            model, tokenizer, statements, layers, device, batch_size
        )
        if str(model_a) == str(model_b):
            Y = X  # self-vs-self: reuse the single extraction
            if verbose:
                print("[relation] self-vs-self (single extraction reused)", flush=True)
        else:
            del model
            model_b_instance, tokenizer_b = _load_model(model_b, dtype)
            model_b_instance.to(device)
            if verbose:
                print(f"[relation] extracting model_b={model_b.name}", flush=True)
            Y = _extract_last_token(
                model_b_instance, tokenizer_b, statements, layers, device, batch_size
            )
            del model_b_instance
    except RelationError as exc:
        return finish("error", {"error": str(exc), "measurement": None})
    except Exception as exc:  # never map execution failure to measured
        return finish(
            "error",
            {"error": f"{type(exc).__name__}: {exc}", "measurement": None},
        )

    layer_scores: dict[str, float] = {}
    feature_stats: dict[str, dict] = {}
    degenerate: list[str] = []
    for layer in layers:
        key = str(layer)
        if not (np.isfinite(X[layer]).all() and np.isfinite(Y[layer]).all()):
            return finish(
                "error",
                {
                    "error": f"layer {layer} produced non-finite hidden states",
                    "measurement": None,
                },
            )
        try:
            Xs = center_standardize(X[layer], f"layer_{layer}_X")
            Ys = center_standardize(Y[layer], f"layer_{layer}_Y")
            score = linear_cka(Xs, Ys)
            if not math.isfinite(score):
                return finish(
                    "error",
                    {
                        "error": f"layer {layer} produced non-finite CKA",
                        "measurement": None,
                    },
                )
            layer_scores[key] = score
            feature_stats[key] = {
                "n_samples": int(Xs.shape[0]),
                "n_features": int(Xs.shape[1]),
                "n_zero_variance": 0,
            }
        except DegenerateInput as exc:
            degenerate.append(f"layer_{layer}: {exc}")
            feature_stats[key] = {
                "n_samples": int(X[layer].shape[0]),
                "n_features": int(X[layer].shape[1]),
                "degenerate": str(exc),
            }

    measurement = {
        "layer_scores": layer_scores,
        "degenerate_layers": degenerate,
        "feature_stats": feature_stats,
    }
    if degenerate:
        return finish(
            INCONCLUSIVE,
            {
                "reasons": degenerate,
                "measurement": measurement,
            },
        )
    return finish(
        MEASURED,
        {
            "reasons": [
                "measurement completed with a finite linear-CKA score for "
                "every requested layer; no relatedness threshold applied"
            ],
            "measurement": measurement,
        },
    )
