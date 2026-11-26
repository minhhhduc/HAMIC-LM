# 04 — TPU runtime

This document covers hardware selection, mesh and sharding, memory budgeting, preemption, observability, and the operational runbook. It does not claim measured performance.

## 1. Target hardware

| Generation | HBM per chip | Typical slice | Role |
|---|---:|---|---|
| TPU v5e | 16 GiB | v5e-8 (8 chips, one host) | baseline for ES/EM/D500M; D1B with FSDP |
| TPU v5p | 95 GiB | v5p-8 / v5p-16 | D1B with less sharding, larger micro-batches |
| TPU v6e (Trillium) | 32 GiB | v6e-8 | preferred if quota is available |

These numbers are from Cloud TPU documentation at design time and **must be re-checked against the quota actually granted**. The runtime records the device type, chip count, HBM per chip, and libtpu version in the run manifest.

Baseline decision D-T10: v5e-8 for the first smoke and for ES/EM/D500M; D1B on v5p/v6e or v5e-8 with FSDP. This is provisional until measured.

## 2. Memory budget

Let $P$ be the parameter count. With fp32 master weights, a bf16 compute copy, bf16 gradients, and fp32 Adam moments:

$$
M_{\text{state}} = 4P + 2P + 2P + 8P = 16P \text{ bytes}.
$$

If gradients are kept in fp32, add $2P$; the runtime must record which policy is used, because it changes the fit.

| Model | $P$ | State (16P) | State (18P) |
|---|---:|---:|---:|
| `hamic-es` | 150,906,432 | 2.41 GB | 2.72 GB |
| `hamic-em` | 348,796,992 | 5.58 GB | 6.28 GB |
| `hamic-d500m` | 486,176,000 | 7.78 GB | 8.75 GB |
| `hamic-d500m-m` | 521,828,608 | 8.35 GB | 9.39 GB |
| `hamic-d1b` | 1,021,143,040 | 16.34 GB | 18.38 GB |
| `hamic-d1b-m` | 1,011,064,832 | 16.18 GB | 18.20 GB |

Activation memory is additional and dominates at large micro-batches. A rough per-tensor scale is

$$
M_{\text{act}} \approx B \cdot L \cdot d \cdot 2 \text{ bytes} \cdot k,
$$

where $k$ counts live tensors per layer (attention scores, projections, MLP intermediates, residuals). For `hamic-es` at $B=192$, $L=512$, $d=768$, one bf16 tensor is ~151 MB; unrematerialized attention/MLP can exceed a chip. Rematerialization (`jax.checkpoint`) trades compute for memory and is expected for the encoder's large micro-batch.

Rules:

- Fit the **state** first; then choose the micro-batch so activations fit with a 20% margin.
- Measure with `jax.profiler` / `jax.live_arrays()`; do not trust the formula for activations.
- If the state does not fit on one chip, shard it (FSDP); do not silently switch to a smaller model.
- Record `peak_hbm_gib` and `peak_host_rss_gib` per run.

## 3. Mesh and sharding

```python
devices = jax.devices()                     # 8 chips on v5e-8
mesh = jax.sharding.Mesh(devices, ("data",))
```

- ES/EM/D500M: data-parallel, parameters replicated; batch sharded over `data`.
- D1B: 2D mesh `("data", "fsdp")` (for example `(4, 2)` on 8 chips); parameters sharded over `fsdp`; batch over `data`.
- Embeddings/LM head: vocabulary-axis sharding when FSDP is active; otherwise replicated.
- Norms/biases: replicated.
- Optimizer state follows the parameter sharding.

Use a single sharding template and `jax.device_put`. Avoid per-tensor `device_put` in the step. GSPMD inserts collectives; `shard_map` is a later optimization and must pass the parity suite on the same mesh.

## 4. Global batch and sharding

The optimizer step must keep the GPU run's global token count if the run is meant to be comparable:

$$
B_{\text{global}} = B_{\text{micro/device}} \times A \times N_{\text{data}}.
$$

Examples for the current recipes:

| Run | GPU micro | GPU accum | Tokens/step | v5e-8 micro/device | Accum | Tokens/step |
|---|---:|---:|---:|---:|---:|---:|
| `hamic-es-v014` | 192 | 4 | 393,216 | 24 | 4 | 393,216 |
| `hamic-em-v05` | 64 | 8 | 262,144 | 8 | 8 | 262,144 |

The table is an existence proof, not a recommendation; the actual micro-batch is chosen from memory measurements. If the division is not integral, the run has a different global batch and is a **new identity** (D-T16). Record `global_batch_tokens` and `tokens_per_step` in the manifest.

## 5. Compilation and XLA flags

Start with defaults. Do not cargo-cult flags. Candidate flags are benchmark-gated and recorded with the run:

- async collective fusion;
- data-parallel all-reduce optimization;
- scoped VMEM limits;
- layout/precision knobs.

For each candidate, run the same 50-step smoke and compare step time and loss; keep a flag only if it improves the median without changing numerics. The parity suite must run with the final flag set, not just defaults.

`jax_default_matmul_precision` is not a substitute for bf16 inputs. Cast weights/activations explicitly; do not rely on a global precision flag to change semantics.

## 6. Preemption and the session clock

TPU spot capacity can be preempted. The runtime must be kill-safe:

- checkpoint every 10–30 minutes, and always before a known soft deadline;
- write to local SSD first, then upload; commit only after upload verification;
- on SIGTERM, stop accepting new steps, finish the current step if safe, checkpoint, exit with `paused`;
- on restart, resolve the newest `COMMITTED` checkpoint, verify manifest and tokenizer, restore optimizer/step/cursor/RNG, continue;
- never overwrite a committed checkpoint; a retry of the same step is an error (HAMIC immutability invariant).

The HAMIC `SessionClock` semantics are the model: a soft deadline pauses the run cleanly; a hard deadline may kill it, and the last committed checkpoint is the recovery point. The JAX runtime records `status ∈ {completed, paused, interrupted, failed}`.

## 7. Observability

Log JSONL records to GCS, one per logging interval:

```json
{"step": 1000, "loss": 2.31, "lr": 0.00078, "grad_norm": 0.84,
 "tokens": 393216000, "tokens_per_second": 12345.6,
 "peak_hbm_gib": 14.2, "device": "v5e-8", "commit": "..."}
```

Also record:

- compiled executable count and recompilation events;
- host RSS and shard cache size;
- data wait time (time spent packing/prefetching vs compute);
- checkpoint duration and upload bytes;
- eval loss and BPB at the configured interval.

Use `jax.profiler.trace` for a short window when step time regresses; do not leave profiling on in production.

## 8. Runbook

### 8.1 Provision

```bash
gcloud compute tpus tpu-vm create hamic-tpu \
  --zone=... --accelerator-type=v5e-8 --version=tpu-ubuntu2204-base
```

Record zone, accelerator type, TPU software version, and libtpu version.

### 8.2 Install

```bash
pip install -e ".[tpu]"
python -c "import jax; print(jax.devices())"
```

The `jax[tpu]` wheel must match the libtpu version installed on the VM. A mismatch is a hard failure, not a warning.

### 8.3 Stage artifacts

- export `HF_KEY` in the VM environment (or inject it from the platform secret manager); never echo it;
- checkout the pinned commit;
- fetch `tokenized_manifest.json` and `tokenizer.json` only;
- verify hashes;
- start the shard cache with the first window;
- do not download the full corpus.

### 8.4 Doctor

`hamic-jax doctor --config <train.yaml>` checks:

- device count and type;
- mesh construction;
- preset/vocab/special-id consistency;
- manifest and tokenizer hashes;
- one synthetic packed batch through the model;
- parameter count against the measured preset;
- checkpoint store read/write permission;
- credential presence: `HF_KEY` is set when the store requires auth, and only its presence is reported.

### 8.5 Smoke

Run 20–50 steps with the real shapes, a tiny token budget, and checkpointing every few steps. Verify:

- finite loss and grad norm;
- loss decreases;
- checkpoint publish, resolve, restore;
- resume produces the same next batch and a loss within tolerance.

### 8.6 Production run

Launch with the config, a run id, and a token budget. Monitor the JSONL, not the terminal. On pause/preemption, restart from the latest committed checkpoint. On completion, export and evaluate.

## 9. Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| OOM at compile | micro-batch too large, no remat | reduce micro-batch, enable remat, shard state |
| OOM at runtime | activation spike, cache growth | measure live arrays, cap shard cache |
| Step time jumps | recompilation from shape change | fix shapes, inspect executable count |
| Loss NaN | bf16 overflow, bad mask, LR spike | check logits/softmax fp32, mask rows, grad norm |
| Loss diverges from GPU at step 0 | weight mapping, loss reduction, LR schedule | run P2 parity before TPU |
| Checkpoint restore fails | code/config/tokenizer/hash mismatch | fail closed; never force-load |
| Preemption loop | spot capacity | move to on-demand or checkpoint more often |
| GCS read stalls | shard cache miss, throttling | prefetch, local SSD, reduce workers |
| Host OOM | packer holds too many rows | stream batches, free arrays, cap prefetch |
| Tokenizer mismatch | wrong artifact | abort before step 0 |

## 10. Security and cost

- GCS access through the TPU VM service account with least privilege; the bucket is read-only for training, write-only for the run prefix.
- The Hugging Face credential for the JAX/TPU backend is the environment variable `HF_KEY`. It is read at process start and passed directly to the Hub client; it never appears in YAML, CLI arguments, logs, metrics, manifests, checkpoints, or exported artifacts. If it is absent when the store requires it, fail before step 0.
- `HF_KEY` belongs to this backend only. The existing GPU PyTorch path keeps its current credential handling and is not modified by this design.
- No corpus or checkpoint in the repo. Secrets come from the VM environment or a secret manager, never from YAML committed to git.
- The tokenized corpus and released model artifacts are immutable; the run writes only under `runs/<run_id>/`.
- Cost is a quota decision, not a design assumption. Record chip-hours and storage egress per run; do not extrapolate from GPU numbers.