"""Measure three pairs of locally generated tiny checkpoints on CPU."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from transformers.utils.logging import disable_progress_bar

from evidence_audit.checkers import relation

LAYERS = (0, 1)
BATCH_SIZE = 8


def build_checkpoints(root: Path) -> tuple[dict[str, Path], Path]:
    """Generate a base, a rank-two weight derivative, and an independent model."""
    words = [f"word{index}" for index in range(48)]
    vocab = {word: index for index, word in enumerate(["[PAD]", "[UNK]", *words])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(13)
        base = Qwen2ForCausalLM(copy.deepcopy(config))
        torch.random.default_generator.manual_seed(29)
        independent = Qwen2ForCausalLM(config)

    derivative = copy.deepcopy(base)
    generator = torch.Generator().manual_seed(41)
    weight = derivative.model.layers[0].self_attn.v_proj.weight
    left = torch.randn((weight.shape[0], 2), generator=generator)
    right = torch.randn((2, weight.shape[1]), generator=generator)
    with torch.no_grad():
        weight.add_(0.02 * (left @ right))

    paths = {}
    for name, model in (
        ("base", base),
        ("derivative", derivative),
        ("independent", independent),
    ):
        path = root / name
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        paths[name] = path

    rng = random.Random(17)
    rows = []
    for index, word in enumerate(words):
        prefix = rng.sample(words, index % 4)
        rows.append({"statement": " ".join([*prefix, word])})
    probes = root / "probes.json"
    probes.write_text(
        json.dumps({"source": "synthetic", "n_rows": len(rows), "rows": rows}),
        encoding="utf-8",
    )
    canonical_rows = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    rows_digest = hashlib.sha256(canonical_rows.encode("utf-8")).hexdigest()
    probes.with_name("probes.digest.json").write_text(
        json.dumps({"sha256_rows": rows_digest}), encoding="utf-8"
    )
    return paths, probes


def main() -> None:
    disable_progress_bar()
    with TemporaryDirectory() as workdir:
        paths, probes = build_checkpoints(Path(workdir))
        for label, other in (
            ("self", "base"),
            ("derivative", "derivative"),
            ("independent", "independent"),
        ):
            result = relation.check_relation(
                paths["base"],
                paths[other],
                probes,
                layers=LAYERS,
                device="cpu",
                batch_size=BATCH_SIZE,
            )
            if result["status"] != relation.MEASURED:
                detail = result.get("error", result.get("reasons"))
                raise SystemExit(f"{label}: {result['status']}: {detail}")
            scores = result["measurement"]["layer_scores"]
            formatted = ", ".join(
                f"layer {layer}={scores[str(layer)]:.6f}" for layer in LAYERS
            )
            print(f"{label}: {result['status']}, {formatted}")


if __name__ == "__main__":
    main()
