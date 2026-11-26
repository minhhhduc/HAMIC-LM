# HAMIC JAX/TPU implementation plan

**Status:** implementation plan. No file under `src/hamic/`, the four GPU notebooks, or
`configs/` is modified.

**Backend:** JAX. The TPU package is `src/hamic_jax/`; the GPU package remains the
read-only oracle.

## Global constraints

- `hamic_jax` must import without `torch` and without importing `hamic`.
- `configs/models/*.yaml`, `tokenizer.json`, and tokenized manifests are read directly.
- TPU credentials use `HF_KEY`. `HF_TOKEN` belongs to the GPU path only.
- Parameter trees use the exact Hugging Face `state_dict` names.
- Tied output heads are one leaf plus alias metadata.
- Line length is 99; public modules carry docstrings and annotations.
- No command in this plan is executed by the authoring session. The TPU operator runs
  the tests on a JAX machine.

## File map

| File | Responsibility |
|---|---|
| `src/hamic_jax/spec.py` | Pure dataclasses for presets, special tokens, training config |
| `src/hamic_jax/presets.py` | Strict YAML loader for `configs/models/*.yaml` |
| `src/hamic_jax/params.py` | Shape-only and initialized flat parameter trees |
| `src/hamic_jax/rope.py` | RoPE tables and `rotate_half` application |
| `src/hamic_jax/norms.py` | LayerNorm (weight-only) and RMSNorm |
| `src/hamic_jax/attention.py` | Mask construction and fused `jax.nn.dot_product_attention` (native GQA) |
| `src/hamic_jax/mlp.py` | GeGLU and SwiGLU |
| `src/hamic_jax/model.py` | ModernBERT, Llama, and Qwen3 forwards plus losses |
| `src/hamic_jax/objectives.py` | CLM labels and dynamic MLM masking |
| `src/hamic_jax/data.py` | Tokenized-shard reader, deterministic packer, sharding |
| `src/hamic_jax/optim.py` | Optax AdamW, WSD/cosine schedules, global-norm clip |
| `src/hamic_jax/train.py` | JIT step, accumulation, train/eval loops |
| `src/hamic_jax/checkpoint.py` | Torch-free immutable checkpoints and object stores |
| `src/hamic_jax/export.py` | Safetensors + Hugging Face config export |
| `src/hamic_jax/downstream.py` | JAX task fine-tuning: task data, encoders, `ModernBertPredictionHead` + classifier, Optax loop, metrics |
| `src/hamic_jax/cli.py` | `hamic-jax doctor/params/train/evaluate/export` |
| `notebooks/tpu_*.ipynb` | Thin Jupyter operator surfaces |
| `tests/jax/*.py` | Static, CPU-JAX, and optional torch-oracle tests |

## Task order

1. **Spec and presets:** strict YAML parsing, all six released presets, special ids.
2. **Parameters:** exact HF key shapes, tied-head alias, measured counts.
3. **Model core:** RoPE, norms, attention, MLP, three architectures.
4. **Objectives and data:** MLM policy, packed batches, cursor, multi-host stream.
5. **Optimizer and trainer:** Optax AdamW/WSD, accumulation scan, jit step.
   Library-first: `optax.adamw` and `optax.join_schedules` with
   `optax.linear_schedule`/`optax.cosine_decay_schedule`; no hand-written AdamW
   or schedule arithmetic. The one deliberate exception is
   `hamic_jax.optim.clip_by_global_norm`: Optax omits the oracle's `+1e-6`
   denominator, so the final-audit rule requires the JAX form. Downstream
   fine-tuning uses the oracle's linear warmup + linear decay, not cosine.
6. **Checkpoint and export:** manifest/commit, `HF_KEY` store, safetensors export.
7. **CLI:** doctor, parameter report, train, evaluate, export.
8. **Notebooks:** drill, train, eval, downstream, each a thin surface over the CLI API.
9. **Tests:** CPU JAX tests always; torch oracle tests behind `pytest.importorskip`.

## Acceptance gates

- P1: package imports without torch or `hamic`; preset and parameter tests pass.
- P2: tiny-model logits/loss match the pinned torch oracle within the documented
  tolerances.
- P3: one TPU smoke run writes a committed checkpoint.
- P4: resume restores the same cursor, step, and optimizer state.
- P5: export loads with stock `transformers` classes.
- P6: production preset run.