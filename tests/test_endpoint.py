import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from evidence_audit.checkers.endpoint import (
    INCONCLUSIVE,
    INSUFFICIENT_EVIDENCE,
    REJECTED,
    SUPPORTED,
    check_endpoint,
    load_manifest,
)
from examples.endpoint_demo import (
    MODEL_CONFIG,
    TAMPERED_TENSOR,
    build_manifest,
    fixture_tensors,
    run_demo,
    write_model_dir,
)


@pytest.fixture
def manifest(tmp_path):
    a, b, expected = fixture_tensors()
    source_a = write_model_dir(tmp_path / "a", a)
    source_b = write_model_dir(tmp_path / "b", b)
    target = write_model_dir(tmp_path / "target", expected)
    return build_manifest(source_a, source_b, target)


def check_verdict(manifest, verdict):
    result = check_endpoint(manifest)
    assert result["execution_status"] == "ok", result
    assert result["verdict"] == verdict, result
    return result


def test_demo():
    assert run_demo() == {
        "declared_merge": {
            "verdict": SUPPORTED,
            "tensors_checked": 3,
            "elements_checked": 16,
            "failing_tensors": [],
            "max_abs_error": 0.0,
        },
        "altered_target": {
            "verdict": REJECTED,
            "tensors_checked": 3,
            "elements_checked": 16,
            "failing_tensors": [TAMPERED_TENSOR],
            "max_abs_error": 0.5,
        },
    }


def test_manifest_roundtrip(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_verdict(load_manifest(path), SUPPORTED)
    assert result["mismatch_report"]["bitwise_identical"] == 3
    assert result["mismatch_report"]["relative_l2"] == 0.0
    assert result["metrics"]["evidence_bytes_read"] > 0


def test_post_sum_normalization(tmp_path):
    a = write_model_dir(tmp_path / "a", {"w": torch.tensor([1.0], dtype=torch.float32)})
    b = write_model_dir(tmp_path / "b", {"w": torch.tensor([5.0], dtype=torch.float32)})
    # The FP32 intermediate 0.6*1 + 0.6*5 rounds below 3.6. Dividing by
    # the FP32 weight sum gives the float32 neighbor immediately below 3.0.
    normalized = write_model_dir(
        tmp_path / "normalized",
        {"w": torch.tensor([2.999999761581421], dtype=torch.float32)},
    )
    raw = write_model_dir(
        tmp_path / "raw", {"w": torch.tensor([3.5999999046325684], dtype=torch.float32)}
    )
    pre_normalized = write_model_dir(
        tmp_path / "pre", {"w": torch.tensor([3.0], dtype=torch.float32)}
    )

    for target, normalize, verdict in [
        (normalized, True, SUPPORTED),
        (normalized, False, REJECTED),
        (raw, False, SUPPORTED),
        (raw, True, REJECTED),
        (pre_normalized, True, REJECTED),
    ]:
        result = check_verdict(
            build_manifest(a, b, target, weights=(0.6, 0.6), normalize=normalize),
            verdict,
        )
        assert result["mismatch_report"]["failing_tensors"] == (
            [] if verdict == SUPPORTED else ["w"]
        )


def test_manifest_order_controls_accumulation(tmp_path):
    sources = []
    for i, value in enumerate([1e10, -1e10, 3.0]):
        path = write_model_dir(
            tmp_path / str(i), {"w": torch.tensor([value], dtype=torch.float32)}
        )
        sources.append({"path": str(path), "weight": 1.0})
    target = write_model_dir(
        tmp_path / "target", {"w": torch.tensor([3.0], dtype=torch.float32)}
    )
    manifest = {
        "sources": sources,
        "target_path": str(target),
        "merge_method": "linear",
        "dtype": "float32",
        "normalize": False,
    }
    check_verdict(manifest, SUPPORTED)
    reordered = {**manifest, "sources": [sources[0], sources[2], sources[1]]}
    assert check_verdict(reordered, REJECTED)["mismatch_report"]["failing_tensors"] == [
        "w"
    ]


def test_changed_coefficients(manifest):
    manifest["normalize"] = False
    check_verdict(manifest, SUPPORTED)
    for source in manifest["sources"]:
        source["weight"] = 0.5
    result = check_verdict(manifest, REJECTED)
    assert len(result["mismatch_report"]["failing_tensors"]) == 3


def test_target_built_with_different_coefficients(tmp_path, manifest):
    a, _, _ = fixture_tensors()
    # A + 4 is the second source, so a 50/50 merge is exactly A + 2.
    manifest["target_path"] = write_model_dir(
        tmp_path / "different_merge", {key: tensor + 2.0 for key, tensor in a.items()}
    )
    result = check_verdict(manifest, REJECTED)
    assert len(result["mismatch_report"]["failing_tensors"]) == 3
    for source in manifest["sources"]:
        source["weight"] = 0.5
    check_verdict(manifest, SUPPORTED)


def test_replaced_source(tmp_path, manifest):
    _, b, _ = fixture_tensors()
    b[TAMPERED_TENSOR] += 1.0
    manifest["sources"][1]["path"] = write_model_dir(tmp_path / "replacement", b)
    result = check_verdict(manifest, REJECTED)
    assert result["mismatch_report"]["failing_tensors"] == [TAMPERED_TENSOR]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sources", []),
        ("sources", ()),
        ("sources", 7),
        ("sources", [42]),
        ("sources", [{"path": "a", "weight": "heavy"}]),
        ("sources", [{"path": "a"}]),
        ("sources", [{"path": 2, "weight": 1.0}]),
        ("sources", [{"weight": 1.0}]),
        ("target_path", 7),
        ("normalize", "yes"),
    ],
)
def test_invalid_manifest_field(manifest, field, value):
    manifest[field] = value
    result = check_verdict(manifest, INSUFFICIENT_EVIDENCE)
    assert field in result["reasons"][0]


@pytest.mark.parametrize(
    "field", ["sources", "target_path", "merge_method", "normalize", "dtype"]
)
def test_missing_manifest_field(manifest, field):
    del manifest[field]
    assert field in check_verdict(manifest, INSUFFICIENT_EVIDENCE)["reasons"][0]


def test_manifest_requires_object():
    assert "object" in check_verdict([], INSUFFICIENT_EVIDENCE)["reasons"][0]


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ("source", "source[0] directory does not exist"),
        ("target", "target directory does not exist"),
        ("config", "target is missing config.json"),
        ("shard", "target has no *.safetensors shards"),
    ],
)
def test_missing_artifact(tmp_path, manifest, missing, reason):
    target = Path(manifest["target_path"])
    if missing == "source":
        manifest["sources"][0]["path"] = tmp_path / "absent"
    elif missing == "target":
        manifest["target_path"] = tmp_path / "absent"
    elif missing == "config":
        (target / "config.json").unlink()
    else:
        (target / "model.safetensors").unlink()
    assert reason in check_verdict(manifest, INSUFFICIENT_EVIDENCE)["reasons"][0]


@pytest.mark.parametrize(
    ("change", "report_field"),
    [
        ("missing", "missing_keys"),
        ("extra", "extra_keys"),
        ("reshape", "shape_mismatches"),
        ("dtype", "dtype_mismatches"),
    ],
)
def test_target_tensor_metadata(tmp_path, manifest, change, report_field):
    _, _, target = fixture_tensors()
    key = TAMPERED_TENSOR
    if change == "missing":
        del target[key]
    elif change == "extra":
        key = "extra.weight"
        target[key] = torch.tensor([1.0])
    elif change == "reshape":
        target[key] = target[key].reshape(-1)
    else:
        target[key] = target[key].double()
    manifest["target_path"] = write_model_dir(tmp_path / "modified", target)
    assert check_verdict(manifest, REJECTED)["mismatch_report"][report_field] == [key]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architectures", ["DifferentArchitecture"]),
        ("model_type", "different_type"),
        ("vocab_size", 32),
        ("tie_word_embeddings", True),
    ],
)
def test_target_config_identity(tmp_path, manifest, field, value):
    _, _, target = fixture_tensors()
    manifest["target_path"] = write_model_dir(
        tmp_path / "modified", target, {**MODEL_CONFIG, field: value}
    )
    assert field in check_verdict(manifest, REJECTED)["reasons"][0]


@pytest.mark.parametrize(
    ("field", "value"),
    [("merge_method", "ties"), ("dtype", "float16")],
)
def test_out_of_scope_recipe(manifest, field, value):
    manifest[field] = value
    assert "outside v0 scope" in check_verdict(manifest, INCONCLUSIVE)["reasons"][0]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("config", "sources disagree"),
        ("keys", "key sets differ"),
        ("shape", "shapes differ"),
        ("dtype", "non-float32 tensors"),
    ],
)
def test_out_of_scope_sources(tmp_path, manifest, change, reason):
    _, b, _ = fixture_tensors()
    config = MODEL_CONFIG.copy()
    if change == "config":
        config["tie_word_embeddings"] = True
    elif change == "keys":
        del b[TAMPERED_TENSOR]
    elif change == "shape":
        b[TAMPERED_TENSOR] = b[TAMPERED_TENSOR].reshape(-1)
    else:
        b[TAMPERED_TENSOR] = b[TAMPERED_TENSOR].double()
    manifest["sources"][1]["path"] = write_model_dir(tmp_path / "different", b, config)
    assert reason in check_verdict(manifest, INCONCLUSIVE)["reasons"][0]


def test_zero_total_weight_with_normalization(manifest):
    manifest["sources"][0]["weight"] = 0.5
    manifest["sources"][1]["weight"] = -0.5
    result = check_verdict(manifest, INCONCLUSIVE)
    assert "total declared weight 0" in result["reasons"][0]


def test_sharded_target(manifest):
    target = Path(manifest["target_path"])
    _, _, expected = fixture_tensors()
    first = target / "model.safetensors"
    second = target / "model-2.safetensors"
    save_file({TAMPERED_TENSOR: expected.pop(TAMPERED_TENSOR)}, str(first))
    save_file(expected, str(second))
    assert check_verdict(manifest, SUPPORTED)["mismatch_report"]["tensors_checked"] == 3


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("bad_archive", "could not load target shard"),
        ("duplicate_key", "duplicate tensor keys across shards"),
        ("bad_json", "config.json is not valid JSON"),
        ("wrong_config_shape", "config.json must be an object"),
    ],
)
def test_malformed_artifact_reports_execution_error(manifest, damage, reason):
    target = Path(manifest["target_path"])
    if damage == "bad_archive":
        (target / "model.safetensors").write_bytes(b"not a Safetensors archive")
    elif damage == "duplicate_key":
        _, _, expected = fixture_tensors()
        save_file(expected, str(target / "duplicate.safetensors"))
    elif damage == "bad_json":
        (target / "config.json").write_text("{", encoding="utf-8")
    else:
        (target / "config.json").write_text("[]", encoding="utf-8")
    result = check_endpoint(manifest)
    assert result["verdict"] == INCONCLUSIVE, result
    assert result["execution_status"] == "error", result
    assert reason in result["reasons"][0]
