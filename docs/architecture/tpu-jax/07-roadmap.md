# 07 — Roadmap, open decisions, risks

This is the execution order after the design is accepted. No phase starts before the previous phase's exit criteria are met. The design is deliberately front-loaded: parity is cheaper to fix on a tiny CPU model than on a TPU.

## 1. Phases

### P0 — Design and decisions (this document set)

Deliverables:

- this folder, reviewed;
- decision register in [README §6](README.md#6-design-decisions);
- the oracle pin (`transformers`, `torch`) recorded;
- the vocab inconsistency (D-T15) resolved or explicitly scoped.

Exit criteria:

- no contradictory numbers across docs;
- every "locked" decision has an owner and a test;
- the user accepts the scope: encoders + decoders, no ED/seq2seq, no data download on the audit machine.

### P1 — Skeleton, config, parameter tree

Deliverables:

- `src/hamic_jax/` package with `config.py`, `presets.py`, `params.py`;
- YAML parser that rejects unknown keys;
- shape-only parameter tree for every preset;
- oracle → JAX converter (safetensors, no torch in the converter);
- parameter-count test;
- the `hamic_jax` calls the TPU notebook cells will use (`probe`, `train`, `resume`, `evaluate`).

Exit criteria:

- counts match the measured table for all six presets;
- key set and shapes match the oracle `state_dict`;
- tied weights have one leaf;
- converter round-trip is exact for a tiny model.

### P2 — Forward parity

Deliverables:

- `rope.py`, `norms.py`, `attention.py`, `mlp.py`, `heads.py`, `model.py`;
- reference attention path and `jax.nn.dot_product_attention` path;
- tiny encoder, Llama, and Qwen3 fixtures;
- logits/loss/gradient parity tests on CPU.

Exit criteria:

- all L2 tiny tests pass within tolerance;
- shape-only checks pass for all presets;
- the document mask and QK-norm order are covered by fixtures;
- no `trust_remote_code`.

### P3 — Optimizer and data plane

Deliverables:

- Optax AdamW, WSD schedule, clipping, accumulation;
- NumPy packer, cursor, masking policy, source weights;
- synthetic shard fixtures;
- 3-step integration test and resume test.

Exit criteria:

- one optimizer step matches the oracle within tolerance;
- the first 5 packed batches match the oracle;
- mask policy tests pass (not bitwise torch RNG);
- resume reproduces the next batch and loss.

### P4 — Checkpoint and export

Deliverables:

- Structural safetensors checkpoint with metadata and manifest/commit protocol;
- resharding support;
- JAX → safetensors + HF config + tokenizer export;
- HF load and logits parity test.

Exit criteria:

- publish/commit/resolve drills pass;
- corrupt payload is rejected;
- exported model loads through a stock HF class;
- logits match the JAX model.

### P5 — TPU smoke (v5e-8)

Deliverables:

- device/mesh bootstrap;
- `hamic-jax doctor`;
- 20–50 step smoke with real shapes;
- JSONL metrics and a checkpoint on GCS;
- `notebooks/tpu_drill.ipynb` and `notebooks/tpu_train.ipynb`, mirroring the GPU cell order.

Exit criteria:

- finite loss and grad norm;
- checkpoint publish/restore on the real store;
- no unexpected recompilation;
- memory within the measured budget;
- the drill stops at step 6, resumes in a new kernel, and reproduces the reference losses.

### P6 — D1B sharding and preemption

Deliverables:

- 2D mesh (`data`, `fsdp`) for D1B;
- preemption-safe loop and soft-deadline pause;
- resume drill, including resharding;
- memory/profile report;
- `notebooks/tpu_eval.ipynb`, mirroring the GPU metric names and masking views.

Exit criteria:

- D1B fits and runs on the chosen TPU slice;
- two kills (before/after commit) recover correctly;
- next-10-step comparison against the control run is explained;
- no silent fallback to a smaller model.

### P7 — Production runs and report

Deliverables:

- one real run per family in scope, with the agreed token budget;
- evaluation (loss, BPB, masking views where applicable);
- exported checkpoints;
- `notebooks/tpu_downstream.ipynb` run on the exported model;
- a written report with evidence and limitations.

Exit criteria:

- P1–P6 reports attached;
- export verified;
- claims match the measured artifacts;
- open decisions closed or explicitly deferred.

## 2. Open decisions

| ID | Decision | Blocks | Options | Default if unresolved |
|---|---|---|---|---|
| D-T15 | 48K vocab for reference decoders | training `hamic-d500m` / `hamic-d1b` | produce a 48K corpus; change presets to 20,544; train only `-M` | use `-M` presets |
| D-T17 | Attention kernel | TPU performance | reference XLA; `dot_product_attention`; Pallas/splash | `dot_product_attention` after parity |
| D-T18 | Checkpoint store adapter | publish/resume | GCS via `google-cloud-storage`; `gcsfs`; S3-compatible | GCS SDK |
| D-T19 | Sparse MLM head | memory/speed | dense v1; static-capacity gather later | dense |
| D-T20 | TPU type/quota | P5/P6 | v5e-8; v5p-8; v6e-8 | v5e-8 smoke, v5p for D1B |
| D-T21 | Export scope | release | only `-M`; both; reference shape-only | export what was trained |
| D-T22 | GPU mask replay | strict trajectory parity | own JAX stream; replay oracle masks; export mask artifact | own stream |
| D-T23 | Multi-host data stream | multi-host D1B | coordinator scatter; deterministic host-indexed stream | single host first |

D-T15 is the only decision that can invalidate a planned run. Resolve it before P7, preferably before P3.

## 3. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF version drift changes semantics | high | high | pin the parity environment; rerun P2 on upgrade |
| bf16 tolerance fails on deep models | medium | high | fp32 norms/softmax/loss; reference path; per-layer error budget |
| Fused QKV / MLP layout mismatch | medium | high | generated mapping; round-trip test; reject unknown keys |
| GQA or QK-norm order wrong | medium | high | dedicated fixtures; compare to HF intermediates |
| D1B does not fit on v5e-8 | medium | high | FSDP; v5p/v6e; reduce micro-batch |
| Dynamic shapes from sparse prediction | medium | medium | dense v1; static capacity if enabled |
| Checkpoint resharding corrupts state | low | high | reshard drill; post-restore logits fixture |
| Preemption loses work | high | medium | 10–30 min checkpoints; commit protocol; resume drill |
| GCS latency/egress | medium | medium | shard cache; prefetch; manifest-only download |
| Tokenizer/manifest mismatch | low | high | fail closed before step 0 |
| Vocab inconsistency (D-T15) | high | high | use `-M`; decide before production |
| Export tied weights duplicated/missing | medium | medium | write once + config tie; HF load test |
| Audit mistaken for a real run | medium | high | gates P3–P6; explicit non-claims in reports |

## 4. Definition of done for "parity"

A run may be described as "same model as the GPU run" only when:

1. the parameter tree matches the oracle's key set, shapes, dtypes, and count;
2. forward logits, loss, and gradients pass the tolerance on converted weights;
3. one optimizer step passes the tolerance with identical optimizer state;
4. the data order and cursor pass the packer fixtures;
5. the mask policy matches, and resume reproduces the mask stream;
6. the checkpoint restores on the target mesh and continues;
7. the export loads through a stock HF class and matches JAX logits;
8. the parity report is attached, with actual errors and the environment.

Anything less is "a JAX implementation inspired by HAMIC", not "the same model".

## 5. Immediate next actions

1. Freeze the oracle environment and record the exact `torch`/`transformers` versions.
2. Generate the HF → JAX key mapping from `state_dict`, including fused-tensor handling.
3. Create the tiny fixtures and the shape-only parameter tree.
4. Implement the reference encoder/decoder forward; run L2 on CPU.
5. Only after P2 passes, provision a TPU and run P3.

Do not start with TPU profiling. A fast wrong model is still wrong, and TPU time is the most expensive place to discover a parity bug.