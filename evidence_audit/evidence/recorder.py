"""Persist and load LoRA boundary evidence as safetensors and JSON.

The SHA-256 manifest records file identity. Evidence authenticity and
transition consistency require separate checks.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

MANIFEST_NAME = "manifest.json"


def json_safe(obj):
    """Convert nested data containing numpy types to plain JSON-able data."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def save_json(path: Path, obj) -> None:
    Path(path).write_text(
        json.dumps(json_safe(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def tensor_hash(t: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(t))


def tensor_hashes(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_hash(t) for name, t in tensors.items()}


def serialize_rng_state(*, include_cuda: bool = True) -> dict:
    """Capture Python/NumPy/Torch CPU/CUDA RNG states as JSON-able data.

    Set `include_cuda=False` to capture a CPU-only run on a machine with GPUs.
    """
    import random

    return {
        "python": json_safe(random.getstate()),
        "numpy": json_safe(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [t.tolist() for t in torch.cuda.get_rng_state_all()]
        if include_cuda
        else [],
    }


def save_boundary(
    boundary_dir: Path,
    *,
    adapter: dict[str, torch.Tensor],
    optimizer_meta: dict,
    optimizer_slots: dict[str, torch.Tensor] | None,
    rng: dict,
    state: dict,
    buffers: dict[str, torch.Tensor] | None,
) -> None:
    """Persist one boundary record as safetensors + JSON.

    Files written: `adapter.safetensors`, `optimizer.json`,
    `optimizer.safetensors` (when slots exist), `rng.json`, `state.json`,
    `buffers.safetensors` (when buffers are passed, i.e. the step-0 record).

    `state` is extended with identity fields: `adapter_sha256` (hash of the
    saved adapter file) and `buffer_sha256` (per-tensor byte hashes).
    """
    boundary_dir = Path(boundary_dir)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = boundary_dir / "adapter.safetensors"
    safetensors.torch.save_file(adapter, str(adapter_path))
    state = dict(state)
    state["adapter_sha256"] = sha256_file(adapter_path)
    state["buffer_sha256"] = tensor_hashes(buffers or {})
    if buffers is not None:
        safetensors.torch.save_file(buffers, str(boundary_dir / "buffers.safetensors"))
    if optimizer_slots is not None:
        safetensors.torch.save_file(
            optimizer_slots, str(boundary_dir / "optimizer.safetensors")
        )
    save_json(boundary_dir / "optimizer.json", optimizer_meta)
    save_json(boundary_dir / "rng.json", rng)
    save_json(boundary_dir / "state.json", state)


def load_boundary(boundary_dir: Path) -> dict:
    """Load a boundary record back into memory (inspection only)."""
    boundary_dir = Path(boundary_dir)
    data = {
        "adapter": safetensors.torch.load_file(
            str(boundary_dir / "adapter.safetensors"), device="cpu"
        ),
        "optimizer_meta": load_json(boundary_dir / "optimizer.json"),
        "rng": load_json(boundary_dir / "rng.json"),
        "state": load_json(boundary_dir / "state.json"),
    }
    slots_path = boundary_dir / "optimizer.safetensors"
    buffers_path = boundary_dir / "buffers.safetensors"
    data["optimizer_slots"] = (
        safetensors.torch.load_file(str(slots_path), device="cpu")
        if slots_path.is_file()
        else None
    )
    data["buffers"] = (
        safetensors.torch.load_file(str(buffers_path), device="cpu")
        if buffers_path.is_file()
        else None
    )
    return data


def write_manifest(run_dir: Path, boundaries) -> dict:
    """Hash every file under `run_dir` (except the manifest) and write it.

    Returns the manifest dict. The manifest is written last so it can cover
    all other evidence files.
    """
    files = {}
    for path in sorted(Path(run_dir).rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(run_dir).as_posix()
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "format_version": 1,
        "run": Path(run_dir).name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "boundaries": [int(b) for b in boundaries],
        "files": files,
    }
    save_json(Path(run_dir) / MANIFEST_NAME, manifest)
    return manifest
