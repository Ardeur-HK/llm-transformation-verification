"""Exercise relation measurement using generated checkpoints and probes."""

import json
import math

import numpy as np
import pytest
import torch

from evidence_audit.checkers import relation
from examples.relation_demo import BATCH_SIZE, LAYERS, build_checkpoints


@pytest.fixture(scope="module")
def relation_fixture(tmp_path_factory):
    return build_checkpoints(tmp_path_factory.mktemp("relation"))


@pytest.mark.parametrize("other", ["base", "derivative", "independent"])
def test_checkpoint_pairs_are_measured(relation_fixture, other):
    paths, probes = relation_fixture
    result = relation.check_relation(
        paths["base"], paths[other], probes, LAYERS, "cpu", BATCH_SIZE
    )

    assert result["status"] == relation.MEASURED, result
    assert result["scope"]["n_probes"] == 48
    assert result["measurement"]["degenerate_layers"] == []
    scores = result["measurement"]["layer_scores"]
    assert set(scores) == {"0", "1"}
    assert all(
        math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores.values()
    )
    if other == "base":
        assert list(scores.values()) == [1.0, 1.0]


def test_selected_hidden_state_and_last_token_match_decoder_hook(relation_fixture):
    paths, probes = relation_fixture
    model, tokenizer = relation._load_model(paths["base"], torch.float32)
    statements = relation.load_probes(probes)["statements"][:4]
    captured = []
    layer = 1

    def capture(_module, _inputs, output):
        captured.append(output)

    hook = model.model.layers[layer].register_forward_hook(capture)
    try:
        batched = relation._extract_last_token(
            model, tokenizer, statements, (layer,), "cpu", 4
        )
    finally:
        hook.remove()

    last_indices = [len(statement.split()) - 1 for statement in statements]
    expected = captured[0][torch.arange(4), last_indices, :].numpy()
    np.testing.assert_array_equal(batched[layer], expected)
    individual = relation._extract_last_token(
        model, tokenizer, statements, (layer,), "cpu", 1
    )
    np.testing.assert_allclose(batched[layer], individual[layer], rtol=0, atol=1e-6)


def test_linear_cka_with_different_feature_counts():
    x = relation.center_standardize(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), "x")
    y = relation.center_standardize(np.array([[3], [3], [7], [7]]), "y")

    assert relation.linear_cka(x, y) == pytest.approx(1 / math.sqrt(2))


def test_linear_cka_zero_denominator():
    with pytest.raises(relation.DegenerateInput, match="zero CKA denominator"):
        relation.linear_cka(np.zeros((3, 1)), np.ones((3, 2)))


@pytest.mark.parametrize(
    ("statements", "reason"),
    [
        (["word1"], "variance undefined"),
        (["word1", "word1"], "zero-variance feature"),
    ],
)
def test_degenerate_layers_are_inconclusive(
    relation_fixture, tmp_path, statements, reason
):
    paths, _ = relation_fixture
    probes = tmp_path / "probes.json"
    probes.write_text(
        json.dumps(
            {
                "n_rows": len(statements),
                "rows": [{"statement": text} for text in statements],
            }
        ),
        encoding="utf-8",
    )

    result = relation.check_relation(
        paths["base"], paths["base"], probes, LAYERS, "cpu"
    )

    assert result["status"] == relation.INCONCLUSIVE
    assert result["measurement"]["layer_scores"] == {}
    assert all(reason in message for message in result["reasons"])


def test_probe_rows_tampered_after_recording_digest(relation_fixture, tmp_path):
    paths, original_probes = relation_fixture
    probes = tmp_path / original_probes.name
    raw = json.loads(original_probes.read_text(encoding="utf-8"))
    raw["rows"][0]["statement"] = "word3"
    probes.write_text(json.dumps(raw), encoding="utf-8")
    digest = original_probes.with_name("probes.digest.json").read_bytes()
    probes.with_name("probes.digest.json").write_bytes(digest)

    result = relation.check_relation(
        paths["base"], paths["base"], probes, LAYERS, "cpu"
    )

    assert result["status"] == "error"
    assert "probe rows digest mismatch" in result["error"]


def test_malformed_probe_json_returns_structured_error(relation_fixture, tmp_path):
    paths, _ = relation_fixture
    probes = tmp_path / "probes.json"
    probes.write_text("{", encoding="utf-8")

    result = relation.check_relation(
        paths["base"], paths["base"], probes, LAYERS, "cpu"
    )

    assert result["status"] == "error"
    assert "could not read probe file" in result["error"]


@pytest.mark.parametrize(
    ("layers", "batch_size", "reason"),
    [
        ((), BATCH_SIZE, "select at least one layer"),
        (LAYERS, 0, "batch_size must be a positive integer"),
        (LAYERS, -1, "batch_size must be a positive integer"),
        ((3,), BATCH_SIZE, "layer 3 outside 0..2"),
    ],
)
def test_invalid_layer_or_batch_selection(relation_fixture, layers, batch_size, reason):
    paths, probes = relation_fixture
    result = relation.check_relation(
        paths["base"], paths["base"], probes, layers, "cpu", batch_size
    )

    assert result["status"] == "error"
    assert reason in result["error"]


def test_nonfinite_hidden_states_return_error(relation_fixture, tmp_path):
    paths, probes = relation_fixture
    model, tokenizer = relation._load_model(paths["base"], torch.float32)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(float("nan"))
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    result = relation.check_relation(tmp_path, tmp_path, probes, LAYERS, "cpu")

    assert result["status"] == "error"
    assert "non-finite hidden states" in result["error"]
    assert result["measurement"] is None
