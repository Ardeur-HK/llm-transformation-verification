# Technical results

## Reproduction scope

Run the three modules under `examples/` and the pytest suite using the [README](../README.md) commands. The reported environment uses Python 3.12 on Apple Silicon macOS, PyTorch 2.8.0 on CPU, NumPy 1.26.4, Transformers 5.5.4, and PEFT 0.18.0. The fixtures are generated from code in temporary directories. The suite requires no model downloads, credentials, network services, or external research data at runtime.

## Observed CPU results

### Endpoint

The fixture has two same-layout checkpoints with three hand-written FP32 tensors, sixteen elements in total. Each value in the second checkpoint equals the corresponding value in the first plus four. With weights 0.75 and 0.25, the hand-computed target is the first checkpoint plus one. The demo then adds 0.5 to one target element.

| Case | Verdict | Tensors checked | Failing tensors | Maximum absolute error |
| --- | --- | ---: | --- | ---: |
| Declared merge | `supported` | 3 | None | 0.0 |
| Altered target | `rejected` | 3 | `lm_head.weight` | 0.5 |

The suite also tests changed coefficients, normalization order, source substitution, shard loading, and missing or malformed evidence.

### Selective replay

The example generates a one-layer Qwen2 model from scratch and records five FP32 CPU AdamW updates with two eight-token examples per update. It uses rank-2 LoRA on `q_proj` and `v_proj`, with alpha 4 and dropout 0.2. The run uses seed 31 and one Torch CPU thread. Replay restores boundary 1 and performs updates 1 through 3 before comparing boundary 4.

| Case | Observed outcome |
| --- | --- |
| Honest replay of `[1,4)` | `supported`, all 4 adapter tensors and 12 optimizer slot tensors matched |
| Changed end adapter | `rejected`, one adapter tensor differed and all optimizer slots matched |

The tests also swap recorded batches, change a start optimizer slot, and change a start adapter. Each completes replay and produces a tensor mismatch. Changing the start optimizer learning rate produces the expected metadata conflict before model initialization. The positive control checks RNG restoration, buffers, and loss diagnostics in addition to the decision.

### Relation measurement

The demo uses 48 generated text probes, a shared synthetic tokenizer, and a randomly initialized three-layer Qwen2 model with hidden size 16. The derivative is a copy with a rank-two perturbation applied to one projection. The independent control uses the same architecture with a separate initialization. Inference uses CPU FP32, eager attention, and batches of eight. Scores are rounded to six decimal places.

| Case | Status | Layer 0 linear CKA | Layer 1 linear CKA |
| --- | --- | ---: | ---: |
| Self-comparison | `measured` | 1.000000 | 1.000000 |
| Perturbed copy | `measured` | 0.906822 | 0.900314 |
| Separately initialized control | `measured` | 0.260445 | 0.245364 |

## Interpretation and limits

**Endpoint.** The legacy `bitwise_fp32_v0` label denotes exact elementwise tensor equality after shape and dtype checks, using `torch.equal`. Positive and negative zero compare equal. Interpret the result on nonempty, finite tensors. A set of empty tensor archives can return `supported` with zero tensors compared, and matching infinities can pass while making error diagnostics non-finite.

The sequential FP32 accumulation may differ from a stacked reduction at the same nominal precision. For example, give five four-element FP32 tensors constant values of `1e10`, `2`, `1`, `-1e10`, and `3`, with unit weights and normalization off. On the tested CPU, ordered addition returns four copies of `3`, while the stacked weighted sum returns four copies of `0`. Producer implementations must agree on the reduction semantics for an exact match. [MergeKit's stacked reduction](https://github.com/arcee-ai/mergekit/blob/a6e402884ba9/mergekit/merge_methods/linear.py#L44-L51).

**Replay.** The replay status is based on end adapter and optimizer tensors. Reported restore equality flags and recorded loss comparisons are diagnostics. `torch.equal` can treat equal values with different dtypes as equal, so the result does not establish byte identity. The prototype assumes the supplied base model, compatible library versions and CPU settings, and complete retained state for the chosen update sequence. Evidence authentication and updates outside the selected interval are outside its scope.

**Relation.** Features are standardized across probes with population standard deviation, and linear CKA is computed in float64 from standardized activations. Degenerate features cause an inconclusive result without epsilon regularization. Each model uses its own tokenizer. The same probes, specified layers, and inference settings define each comparison. Self-comparison reuses activations. The synthetic controls demonstrate the measurement pipeline. Threshold calibration, false-positive rates, and lineage are outside their scope. [Linear CKA](https://proceedings.mlr.press/v97/kornblith19a.html).

## Historical evidence

Earlier model-scale merge checks, LoRA interval replay, and REEF/TruthfulQA-based relation comparisons are separate from this reproduction suite. Their checkpoint files, probe data, and numerical results are excluded pending release permission. The CPU runs here do not reproduce those historical experiments or establish compatibility with their original GPU environment.
