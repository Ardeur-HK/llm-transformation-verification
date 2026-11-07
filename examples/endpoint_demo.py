"""Check a hand-computed merge and a target with one changed tensor value."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from evidence_audit.checkers.endpoint import REJECTED, SUPPORTED, check_endpoint

MODEL_CONFIG = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "vocab_size": 16,
    "tie_word_embeddings": False,
    "dtype": "float32",
}
TAMPERED_TENSOR = "lm_head.weight"


def fixture_tensors() -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    """Return sources A and B and the hand-computed target for weights 0.75/0.25."""
    a = {
        "wte.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "h.0.ln.weight": torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32),
        "lm_head.weight": torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]], dtype=torch.float32
        ),
    }
    b = {key: tensor + 4.0 for key, tensor in a.items()}
    # 0.75 * A + 0.25 * (A + 4) = A + 1, exactly for these values.
    expected = {key: tensor + 1.0 for key, tensor in a.items()}
    return a, b, expected


def write_model_dir(
    path: Path, tensors: dict[str, torch.Tensor], config: dict | None = None
) -> Path:
    path.mkdir()
    metadata = MODEL_CONFIG if config is None else config
    (path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    save_file(tensors, str(path / "model.safetensors"))
    return path


def build_manifest(
    source_a: Path,
    source_b: Path,
    target: Path,
    *,
    weights=(0.75, 0.25),
    normalize=True,
) -> dict:
    return {
        "sources": [
            {"path": str(source_a), "weight": weights[0]},
            {"path": str(source_b), "weight": weights[1]},
        ],
        "target_path": str(target),
        "merge_method": "linear",
        "normalize": normalize,
        "dtype": "float32",
    }


def run_demo() -> dict:
    with TemporaryDirectory(prefix="endpoint-demo-") as directory:
        root = Path(directory)
        a, b, expected = fixture_tensors()
        source_a = write_model_dir(root / "source_a", a)
        source_b = write_model_dir(root / "source_b", b)
        target = write_model_dir(root / "target", expected)

        altered_tensor = expected[TAMPERED_TENSOR].clone()
        altered_tensor[0, 0] += 0.5
        altered_target = write_model_dir(
            root / "altered_target", {**expected, TAMPERED_TENSOR: altered_tensor}
        )
        cases = [
            ("declared_merge", target, SUPPORTED, []),
            ("altered_target", altered_target, REJECTED, [TAMPERED_TENSOR]),
        ]
        summary = {}
        for name, delivered, verdict, failing_tensors in cases:
            result = check_endpoint(build_manifest(source_a, source_b, delivered))
            if result["execution_status"] != "ok" or result["verdict"] != verdict:
                raise RuntimeError(f"{name}: unexpected endpoint result {result}")
            report = result["mismatch_report"]
            if report["failing_tensors"] != failing_tensors:
                raise RuntimeError(f"{name}: unexpected comparison report {report}")
            summary[name] = {
                "verdict": result["verdict"],
                "tensors_checked": report["tensors_checked"],
                "elements_checked": report["elements_checked"],
                "failing_tensors": report["failing_tensors"],
                "max_abs_error": report["max_abs_error"],
            }
    return summary


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
