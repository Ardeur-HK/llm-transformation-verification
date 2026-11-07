# LLM Transformation Verification

This research prototype checks evidence about a model transformation. Given local checkpoints, a merge recipe, or a recorded training interval, it can test numerical consistency. Given two checkpoints and a set of text probes, it can measure how similarly the models represent those probes.

| Approach | Inputs | Result |
| --- | --- | --- |
| Direct endpoint verification | Ordered source checkpoints, a target checkpoint, and a linear merge manifest | Whether the delivered target matches the recomputed merge under the checker's FP32 policy |
| Selective process replay | Local base checkpoint, recorded LoRA and AdamW boundaries, RNG state, and tokenized batches | Whether replay of a selected interval reproduces the recorded end adapter and optimizer tensors |
| Post-hoc relation measurement | Two local causal-LM checkpoints with tokenizers, ordered text probes, and decoder-layer indices | One raw linear centered kernel alignment (CKA) score per layer |

## Installation

Run from the project root in a shell on Apple Silicon macOS or Linux. The commands were validated with Python 3.12 on Apple Silicon. Package installation needs access to a Python package index. The examples and tests use the CPU and generate their own fixtures locally.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
```

On Linux, install the CPU-only PyTorch wheel inside the environment before running `pip install '.[test]'`. [PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/#v280).

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

```sh
python -m examples.endpoint_demo
python -m examples.replay_demo
python -m examples.relation_demo
python -m pytest -q
```

The endpoint demo compares a hand-computed merge with a modified target. The replay demo records and replays part of a short CPU LoRA run on a small model initialized locally. The relation demo compares a model with itself, a modified copy, and a separately initialized model. The examples print summaries and use temporary directories for their artifacts. [Settings and observed outputs](docs/RESULTS.md).

## Methods and scope

### Direct endpoint verification

Call `check_endpoint(manifest)` with a dictionary, or use `load_manifest(path)` to read a JSON manifest. Supply an ordered `sources` list of `{"path": "...", "weight": 0.75}` entries, `target_path`, `merge_method: "linear"`, an explicit Boolean `normalize`, and `dtype: "float32"`. Each checkpoint directory contains `config.json` and its safetensors files. Relative paths are resolved from the working directory.

The checker compares selected architecture metadata and stored tensor keys, shapes, and dtypes. It computes an ordered FP32 weighted sum and, when requested, divides the result by the FP32 sum of weights. It compares tensors using `torch.equal` and reports missing evidence, out-of-scope inputs, and mismatches separately. The scope is same-architecture linear merging over matching tensor layouts. Tied-weight checks are limited, and tokenizer identity and provenance remain outside the check. Floating-point reduction order can matter when comparing outputs from another implementation. [Numerical limitations and a reduction-order example](docs/RESULTS.md).

### Selective process replay

Call `replay_interval(run_dir, start_step=..., end_step=...)` on evidence saved with the recorder. The checker loads a local safetensors base checkpoint in FP32, reconstructs LoRA and AdamW, restores the starting adapter, optimizer slots and Python, NumPy and Torch CPU RNG, then replays the recorded batches in order for `[start_step, end_step)`. Pass `base_snapshot=path` when relocating a recorded checkpoint. The [replay example](examples/replay_demo.py) shows the evidence files and metadata used by the checker.

Evidence-loading or state-restoration errors stop replay. After a completed run, the decision compares end adapter tensors and optimizer slots with `torch.equal`. The step-zero buffer check, post-restore equality flags, and loss comparisons are reported as diagnostics.

The decision is conditional on the supplied evidence and a matching deterministic CPU environment. The replay loop uses one AdamW parameter group with one optimizer step per recorded batch. Scheduler state, gradient accumulation, unselected updates, uncaptured state, authenticity, and the recorder's file hashes remain outside the check. Inspect `status` and `execution_status` before interpreting diagnostics. Run replay in a separate process if the caller needs its own random streams.

### Post-hoc relation measurement

Call `check_relation(model_a, model_b, probes_path, layers=(...), device="cpu")` on local model directories. A probe file is a JSON object with an ordered `rows` array and an `n_rows` count. Each row contains a `statement` string. A sibling `<stem>.digest.json` can provide `sha256_rows` to check the rows against a recorded digest. The checker tokenizes the same statements for each model, extracts each row's last real token from hidden-state slot `layer + 1`, standardizes features across probes, and measures linear CKA. Hidden-state slot semantics depend on the model architecture. Degenerate inputs produce an inconclusive result.

Scores depend on probe content, tokenization, layer, and model configuration. They have no calibrated cutoff and cannot establish lineage, ownership, or a particular training recipe. Self-comparison reuses one extraction and serves as a numerical control.

## Project layout

| Path | Contents |
| --- | --- |
| `evidence_audit/checkers/` | Endpoint, replay, and relation APIs |
| `evidence_audit/evidence/recorder.py` | Serialization for training-boundary evidence |
| `examples/` | Offline demos and generated fixtures |
| `tests/` | Positive controls, scope checks, and targeted tampering cases |
| `docs/RESULTS.md` | Reproduction settings, measured outcomes, and limitations |

## Reproducibility

The tests exercise successful checks before interpreting negative cases. A completed tensor mismatch or a specific recipe conflict is checked separately from initialization failure. Use trusted local inputs. The prototype runs in the caller's Python process, reads local artifacts, and has no isolation or evidence-authentication layer. Keep runtime output and real training evidence out of the project tree. The released reproduction scope is the synthetic CPU suite. [Coverage and historical evidence status](docs/RESULTS.md).

## Credits and license

- The endpoint checker independently implements ordered weighted summation. MergeKit's linear merge informed the historical reference runs. [MergeKit reference implementation](https://github.com/arcee-ai/mergekit/blob/a6e402884ba9/mergekit/merge_methods/linear.py).
- Replay uses the LoRA method of Hu et al. and the PyTorch, Transformers, and PEFT implementations for model execution and optimization. [LoRA paper](https://openreview.net/forum?id=nZeVKeeFYf9), [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft).
- The relation procedure is adapted from REEF and uses linear CKA as described by Kornblith et al. [REEF paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/77e830bd802f4b63b682038d090a97b6-Abstract-Conference.html), [linear CKA paper](https://proceedings.mlr.press/v97/kornblith19a.html).
- NumPy provides array operations, safetensors provides tensor serialization, and Tokenizers builds the synthetic tokenizer. [NumPy](https://numpy.org/), [safetensors](https://github.com/huggingface/safetensors), [Tokenizers](https://github.com/huggingface/tokenizers).

This project is a personal research portfolio example. Project-wide open-source licensing is deferred. Dependencies are installed separately under their upstream terms. The examples generate all checkpoint and probe data locally.
