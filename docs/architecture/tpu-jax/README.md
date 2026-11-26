# HAMIC on TPU with JAX — design set

**Status:** implementation written; static verification only. No TPU run, no measured
performance. The code lives in `src/hamic_jax/`, `notebooks/`, and `tests/jax/`.
**Audience:** JAX/TPU implementers, model auditors, run operators.
**Relationship to the current implementation:** `src/hamic` (PyTorch + Transformers, GPU) is the **reference oracle**. The TPU implementation must reproduce the same mathematical function, objective, optimizer, and data order; runtime, kernels, and checkpoint packaging differ.

> The current workstation is an **audit machine only**. It has no TPU, and the corpus must not be downloaded. Local verification uses synthetic fixtures and shape-only configurations; see [06-audit-plan.md](06-audit-plan.md).

## 0. Implementation status

| Surface | State |
|---|---|
| Package | `src/hamic_jax/` — 16 modules, Optax-based, torch-free |
| Operator surfaces | `notebooks/tpu_{drill,train,eval,downstream}.ipynb` |
| Tests | `tests/jax/test_{static_import,presets,forward,ngram}.py`; `test_ngram.py` runs with NumPy only, the rest need a JAX/Optax machine |
| Verification | `py_compile` clean; imports/forwards not executed here (audit machine rule) |
| Runtime deps | `jax[tpu]`, `optax`, `safetensors`, `huggingface_hub`, `numpy`, `tokenizers`, `datasets` for downstream |

`pyproject.toml` is intentionally untouched, so `hamic_jax` is not packaged by the wheel
target and `tests/jax` is outside the default pytest roots.

## 1. Feasibility verdict

**Yes, with conditions.** The encoder (`HAMIC-ES`/`HAMIC-EM`) and decoder (`HAMIC-D500M-M`/`HAMIC-D1B-M`) families are standard Transformer stacks with stock Hugging Face classes, so a JAX/TPU port can preserve the mathematical function. The reference decoder presets (`hamic-d500m`, `hamic-d1b`) are **not trainable on current data** because they declare a 48,000-token vocabulary while the tokenized corpora use 20,544; train the `-M` presets or produce a 48K corpus first (D-T15).

The costly parts are not the layers. They are: exact HF parameter mapping (fused QKV/MLP, tied embeddings, QK-norm order), the data plane (packing, masking, cursor), checkpoint resharding, and export. The design front-loads those into CPU parity gates before any TPU spend.

Not feasible in v1: the ED/seq2seq path (no matching stock HF class), bitwise PyTorch parity, and any performance claim before measurement.

## 2. Goals

1. Run the existing HAMIC model families on TPU with JAX, starting with the MLM encoders `HAMIC-ES`/`HAMIC-EM` and the CLM decoders `HAMIC-D500M-M`/`HAMIC-D1B-M`. The reference decoders `HAMIC-D500M`/`HAMIC-D1B` are in scope for shape/export parity only until D-T15 is resolved.
2. Preserve **computational parity** with the GPU implementation: same parameters, forward, loss, optimizer update, and batch order. The exact contract is [01-parity-contract.md](01-parity-contract.md).
3. Preserve the release path: a JAX checkpoint must export to safetensors + `config.json` + tokenizer and load through a **stock** `transformers` class, with no `trust_remote_code`.
4. Separate two planes: data preparation (CPU, Python/NumPy/Rust tokenizer) and training (JAX/TPU). The TPU never tokenizes, deduplicates, or crawls.
5. Support auditing on a machine with neither TPU nor real data.

## 3. Scope

| Area | In scope for v1 | Out of scope for v1 |
|---|---|---|
| MLM encoders | `HAMIC-ES`, `HAMIC-EM` (ModernBERT) | legacy E100M/E300M pilots |
| CLM decoders | `HAMIC-D500M-M`, `HAMIC-D1B-M` (training); `HAMIC-D500M`, `HAMIC-D1B` (shape/export parity) | ED/seq2seq, cross-attention |
| Masking | token, span, PMI, PMI-ngram, whole-word | spelling-variant corruption (`variant_share > 0`) |
| Tokenizer | reuse the GPU release's `tokenizer.json` | train a new tokenizer on TPU |
| Data | read existing `.npy` shards + manifest | crawl, dedup, filter, re-tokenize |
| Checkpoints | Optimizer-agnostic safetensors + HAMIC manifest | a new serving-only checkpoint format |
| Export | safetensors + HF config + tokenizer | serving engine, quantization, distillation |
| Operator flow | separate TPU notebooks for drill/train/eval/downstream; GPU notebooks untouched | shared notebooks or a backend-switch notebook |
| Hardware | TPU v5e-8 and newer | multi-node GPU, CPU cluster |

## 4. Non-goals

- Do not delete or replace the PyTorch path. Both paths coexist; PyTorch remains the oracle and the already-verified export path.
- Do not edit `src/hamic/`, the four GPU notebooks, GPU tests, or `configs/**` for the TPU port. Read them; add only TPU-owned files.
- Do not change architecture, depth, vocabulary, or objective to "fit TPU". Any such change creates a new model identity and needs its own decision record.
- Do not claim bitwise parity between PyTorch and JAX. Parity is **mathematical parity with numerical tolerances**, not identical RNG streams or kernels.
- Do not claim throughput, MFU, or cost before measuring on real TPU hardware.
- Do not train or download corpora on the audit machine.
- Do not use `trust_remote_code` in any released artifact.

## 5. Precedence when documents disagree

1. `configs/models/*.yaml`, `configs/train/**/*.yaml`, and `src/hamic/tokenization/spec.py` — source of truth for shapes, vocabulary, and special tokens.
2. [docs/architecture/model-specifications.md](../model-specifications.md) — canonical architecture specification.
3. [01-parity-contract.md](01-parity-contract.md) — parity contract for the TPU implementation.
4. The remaining implementation documents in this folder.

If a document here disagrees with (1) or (2), (1)/(2) wins and this folder must be corrected.

## 6. Document set

| Document | Content |
|---|---|
| [01-parity-contract.md](01-parity-contract.md) | Definition of parity; formulas, shapes, masks, loss, optimizer, schedule, data order, tolerances |
| [02-jax-model.md](02-jax-model.md) | JAX package layout, parameter tree, attention/RoPE/norm/MLP/head, dtype, jit/scan/remat |
| [03-data-plane.md](03-data-plane.md) | `.npy` shards, manifest, packing, masking, cursor, GCS reads, no local corpus |
| [04-tpu-runtime.md](04-tpu-runtime.md) | TPU selection, mesh/sharding, memory estimates, preemption, logging, runbook |
| [05-checkpoint-export.md](05-checkpoint-export.md) | structural safetensors + manifest/COMMITTED, safetensors conversion, HF export |
| [06-audit-plan.md](06-audit-plan.md) | Local-only audit: test matrix, fixtures, CI, gates, evidence |
| [07-roadmap.md](07-roadmap.md) | Phases P0–P7, exit criteria, open decisions, risks |
| [08-notebook-parity.md](08-notebook-parity.md) | Separate TPU notebook set mirroring the GPU flow; no shared source, no GPU edits |

## 7. Design decisions

| ID | Decision | Status | Rationale |
|---|---|---|---|
| D-T01 | Add `src/hamic_jax/` in the same repo; share `configs/`, tokenizer, shards, and manifests | **locked** | One source of truth; no config fork |
| D-T02 | Plain JAX pytrees + Optax; no Flax NNX; no Orbax in v1 | **locked for v1** | Flat parameter tree keyed by `state_dict` names; Optax owns the optimizer; structural safetensors restore |
| D-T03 | A pinned PyTorch + `transformers` environment is the parity oracle | **locked** | Any HF deviation is a contract deviation, not an optimization |
| D-T04 | Tokenization and data preparation stay outside the TPU | **locked** | Rust tokenizer runs on CPU; TPU only reads shards |
| D-T05 | Parameters are `dict[str, Array]` keyed by **exact** HF `state_dict` names | **locked** | Conversion, export, and comparison need no second naming table |
| D-T06 | bf16 compute; fp32 master weights, optimizer state, and loss | **locked** | Numerical stability; TPU is optimized for bf16 |
| D-T07 | v1 attention uses reference XLA + `jax.nn.dot_product_attention`; Pallas/splash only after parity | **locked** | A faster kernel must not hide a math bug |
| D-T08 | `sparse_prediction` is off in v1; enabling it requires static capacity | **provisional** | JAX dislikes dynamic shapes; the loss must stay identical |
| D-T09 | Checkpoints use structural safetensors + HAMIC `manifest.json` + `COMMITTED.json` | **locked** | Preserves publish-last, verification, and immutability without a broken Orbax dependency |
| D-T10 | Baseline TPU is v5e-8; D1B needs v5p/v6e or more chips | **provisional** | Pending quota and measurement |
| D-T11 | Parity tolerances in §13 of the contract; calibrate on tiny models first | **provisional** | No JAX measurements yet |
| D-T12 | v1 covers encoders and decoders only; ED/seq2seq later | **locked** | No matching stock HF class for the current ED path |
| D-T13 | The audit machine does not train and does not download corpora | **locked** | No hardware, no data egress |
| D-T14 | Export still targets stock HF classes; JAX is a training runtime only | **locked** | Preserves HAMIC's release invariant |
| D-T15 | Reference decoder presets (vocab 48,000) require a 48K corpus; v1 TPU runs use the `-M` presets (vocab 20,544) | **open** | The current GPU presets are internally inconsistent; see [07-roadmap.md](07-roadmap.md) |
| D-T16 | Preserve global `tokens_per_step` when porting to multiple chips; changing global batch creates a new run identity | **locked** | Otherwise the trajectory diverges at step 0 |
| D-T24 | TPU has its own four notebooks, `notebooks/tpu_*.ipynb`, mirroring GPU behavior but sharing no notebook source | **locked** | Prevents a backend switch and a second UX in one file; see [08-notebook-parity.md](08-notebook-parity.md) |
| D-T25 | The TPU port adds `src/hamic_jax/`, `notebooks/tpu_*.ipynb`, `tests/jax/`, and packaging extras only; it does not edit `src/hamic/`, GPU notebooks, GPU tests, or configs | **locked** | The GPU path stays the untouched oracle |

## 8. What "same logic" means

Given the same parameters and inputs, the TPU implementation must produce the same mathematical result:

- same parameter count and shape for every tensor;
- same forward formula, normalization order, RoPE, and masks;
- same loss and reduction domain;
- same AdamW step, schedule, clipping, and accumulation;
- same document/batch order and resumable cursor;
- same special-token ids and tokenizer hash.

Not required: same RNG bitstream, kernels, reduction order, speed, compilation, or memory measurements.

## 9. Gates before claiming "the TPU version works"

1. **P1 — static:** presets, parameter counts, special-token ids, and manifest schema match the GPU implementation.
2. **P2 — CPU parity:** forward, loss, gradients, and one optimizer step match the oracle within tolerance, on a tiny model and on real shapes with converted weights.
3. **P3 — TPU smoke:** 20–50 steps on TPU, finite loss, checkpoint publish/restore works.
4. **P4 — resume drill:** kill mid-run, resume, next batch and loss match the uninterrupted run within tolerance.
5. **P5 — export:** a JAX checkpoint exports to safetensors, loads through `AutoModelForMaskedLM`/`AutoModelForCausalLM`, and logits match the JAX model.
6. **P6 — real run:** a run with a real token budget, evaluation and BPB, artifacts, and a written report.

Until P2 passes, any TPU number is a smoke test, not a scientific result.