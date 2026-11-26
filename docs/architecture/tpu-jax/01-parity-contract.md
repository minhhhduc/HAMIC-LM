# 01 — Parity contract

**Normative.** This document defines what "the JAX/TPU model computes the same function as the GPU model" means. If an implementation choice conflicts with this contract, the contract wins unless `configs/` or [model-specifications.md](../model-specifications.md) says otherwise (see [README §4](README.md#4-precedence-when-documents-disagree)).

## 1. Definition

Parity relates two implementations $A$ (PyTorch/GPU, oracle) and $B$ (JAX/TPU) over the same model identity $M$, parameters $\theta$, and batch $x$:

$$
\| f_A(\theta, x) - f_B(\theta, x) \|_\infty \le \tau_{\text{fwd}},
$$

$$
| \mathcal{L}_A(\theta, x) - \mathcal{L}_B(\theta, x) | \le \tau_{\text{loss}},
$$

$$
\cos(\nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B) \ge \tau_{\text{grad}},
$$

and after one optimizer step with identical optimizer state,

$$
\| \theta'_A - \theta'_B \|_\infty \le \tau_{\text{step}}.
$$

Parity is **not** identical RNG streams, kernels, reduction order, compiled programs, throughput, memory, or checkpoint bytes. Tolerances are in §13; they are provisional until measured.

## 2. Oracle environment

The oracle is `src/hamic` with:

- the `transformers` version pinned in `requirements-parity.lock` (created at implementation start);
- the exact class used by `build_model`: `ModernBertForMaskedLM`, `LlamaForCausalLM`, or `Qwen3ForCausalLM`;
- PyTorch CPU for deterministic fixtures, GPU for end-to-end checks;
- deterministic settings where available (`torch.use_deterministic_algorithms(True)` for fixtures that support it).

Match the **pinned HF source**, not a remembered description. An HF upgrade is a parity event: rerun P2 and record the new pin.

## 3. Model identities and presets

| Identity | Family | Layers | Hidden | Heads (Q/KV) | Head dim | FFN | Vocab | Context | Params (measured) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hamic-es` | ModernBERT MLM | 24 | 768 | 12 | 64 | 1,408 | 20,544 | 2,048 | 150,906,432 |
| `hamic-em` | ModernBERT MLM | 28 | 1,024 | 16 | 64 | 2,432 | 20,544 | 2,048 | 348,796,992 |
| `hamic-d500m` | Llama CLM | 24 | 1,280 | 10 / 2 | 128 | 3,584 | 48,000 | 2,048 | 486,176,000 |
| `hamic-d500m-m` | Qwen3 CLM | 28 | 1,280 | 10 / 2 | 128 | 3,584 | 20,544 | 2,048 | 521,828,608 |
| `hamic-d1b` | Llama CLM | 22 | 2,048 | 16 / 4 | 128 | 5,120 | 48,000 | 2,048 | 1,021,143,040 |
| `hamic-d1b-m` | Qwen3 CLM | 24 | 2,048 | 16 / 4 | 128 | 4,864 | 20,544 | 2,048 | 1,011,064,832 |

Parameter counts are the **measured** values in the model specs. Never derive a release count from a formula; the JAX side reproduces the count by building the tree and summing leaf sizes.

RoPE theta, norm epsilon, bias flags, activation, and tying come from `configs/models/*.yaml` and the HF `config.json` generated for that preset. The JAX parser must reject unknown keys.

### 3.1 Vocab inconsistency (blocking for reference decoders)

`hamic-d500m` and `hamic-d1b` declare vocab 48,000, but no current HAMIC tokenized corpus has a 48K tokenizer; the current corpus uses 20,544. These presets can be shape-tested and exported, but cannot be trained on available data. v1 TPU training uses `hamic-d500m-m` and `hamic-d1b-m` unless a 48K tokenized corpus is produced. Decision D-T15 is open.

## 4. Tokenization and special ids

The tokenizer is the released `tokenizer.json` (plain `tokenizers` JSON, no custom code). The JAX data plane loads token ids from the tokenized manifest, never hard-codes them.

| Item | Contract |
|---|---|
| Vocabulary size | exactly `vocab_size` in the preset |
| Special ids | exactly the ids in the released tokenizer spec |
| `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>` | must match `src/hamic/tokenization/spec.py` |
| Document boundary | `<s>` at document start, `</s>` at document end |
| Shard contents | raw token ids only, no special tokens; the packer adds them |
| Dtype | `uint16` when `vocab_size <= 65,536`, else `uint32` |

Any mismatch aborts before the first step.

## 5. Packed sequence layout

`PackedBatches` produces rows of exactly `seq_len` tokens:

```
row = concat(piece(doc_i)[offset:], piece(doc_{i+1}), ..., piece(doc_j)[:take])
piece(doc) = [bos_id] + doc_tokens + [eos_id]
```

Properties to preserve:

- documents concatenate without padding;
- a document may be split across rows, including mid-document;
- a row may contain several documents and a trailing fragment;
- there is no attention padding mask; every position is a real token;
- the cursor is `(epoch, position, token_offset, consumed_tokens)` and advances per token taken;
- `consumed_tokens += batch_size * seq_len` after each batch;
- when `position` reaches the end of the epoch permutation, the next epoch permutation is generated and `position` resets.

The exact permutation is in [03-data-plane.md §4](03-data-plane.md#4-document-order-and-source-weights). The packer is CPU/NumPy and must be a line-by-line port, not a "similar" sampler.

## 6. Positions and attention masks

### 6.1 Position ids

The GPU path does not pass `position_ids`; HF therefore uses

$$
\text{position\_ids}[b, t] = t, \quad t = 0,\ldots,L-1.
$$

Positions do **not** reset at document boundaries inside a packed row. JAX must use the same row-relative positions. Verify with a fixture where a row contains two documents: the second document's first token has position equal to its column index, not 0.

For decoders, the same rule applies; causal attention uses row order, and document boundaries do not break causality.

### 6.2 Encoder document mask

When `cross_document_attention: false` (encoder default), the encoder uses a block-diagonal document mask:

```python
documents = cumsum(input_ids == bos_id, axis=1) - 1
mask[b, i, j] = documents[b, i] == documents[b, j]
```

The mask is boolean `[B, 1, L, L]`, or an equivalent additive mask with `0` allowed and a large negative value disallowed. Every row has at least one allowed position; a fully masked query must not produce NaN.

HF's ModernBERT expects an attention-mask dictionary; JAX maps the document mask to the same semantics as `DocumentMask`, not to a padding mask.

### 6.3 Decoder causal mask

The decoder uses a standard lower-triangular mask:

$$
\text{allowed}[i,j] = j \le i.
$$

There is no padding mask because rows are full. Packed documents may attend to earlier documents in the same row; this is the current GPU behavior when `cross_document_attention: true` and must not be "fixed" in the TPU port.

### 6.4 Mask dtype and softmax

- Build masks as boolean or additive in the activation dtype.
- Apply softmax in fp32, then cast back; this is the stable path and avoids bf16 overflow.
- A fully masked query is a bug; assert against it in debug mode.

## 7. Encoder forward (ModernBERT)

The normative implementation is HF `ModernBertForMaskedLM` at the pinned version. Mirror its module order exactly:

1. token embedding;
2. embedding norm;
3. per layer: attention norm → self-attention with RoPE and the document mask → residual → MLP norm → GeGLU MLP → residual;
4. final norm;
5. MLM head: dense → activation → norm → decoder;
6. decoder weight tied to the token embedding where the preset config says so.

Do not assume LayerNorm vs RMSNorm, bias presence, or activation from the family name. Read the HF config and `state_dict`; the converter rejects missing or unexpected tensors.

### 7.1 Attention

With input $x$:

$$
Q = x W_Q,\quad K = x W_K,\quad V = x W_V,
$$

$$
\text{head}_h = \text{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_h}} + M\right) V_h,
$$

$$
\text{attn} = \text{concat}(\text{head}_1,\ldots,\text{head}_H) W_O.
$$

ModernBERT may store a fused `Wqkv`; JAX keeps the same fused tensor or a tested split view. A split view must not change the math or the parameter count.

### 7.2 RoPE

Use HF's `rotate_half` convention:

$$
\text{rotate\_half}(x) = \text{concat}(-x_{d/2:}, x_{:d/2}),
$$

$$
q_{\text{rot}} = q \cos + \text{rotate\_half}(q) \sin,
$$

with

$$
\text{inv\_freq}_i = \theta^{-2i/d}, \quad
\text{freqs}_{t,i} = t \cdot \text{inv\_freq}_i.
$$

Compute `inv_freq` in fp32, compute cos/sin, then cast exactly as HF does. Apply RoPE to Q and K after the QKV projection (and after QK-norm where present), never to V.

### 7.3 MLP

GeGLU:

$$
\text{MLP}(x) = W_{\text{down}}\left(\text{gelu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right),
$$

where `gelu` is the exact erf-based GELU used by HF's `GELUActivation` unless the preset explicitly selects the tanh approximation. If HF stores a fused `Wi`, split along the correct axis with a tested view.

## 8. Decoder forward (Llama / Qwen3)

The normative implementation is HF `LlamaForCausalLM` or `Qwen3ForCausalLM` at the pinned version.

Per layer:

1. input RMSNorm;
2. self-attention with RoPE, GQA, causal mask;
3. residual add;
4. post-attention RMSNorm;
5. SwiGLU MLP;
6. residual add.

Then final RMSNorm and LM head. The LM head is tied to the token embedding where the preset says so.

### 8.1 GQA

With $H$ query heads and $G$ KV heads, $H$ must be divisible by $G$. Repeat each KV head $H/G$ times **logically** for the attention computation; do not materialize a repeated tensor if a broadcast/einsum view is possible. The result must be identical to HF's `repeat_kv`.

### 8.2 QK-Norm (Qwen3 `-M` presets)

Qwen3 applies RMSNorm to each query and key head **after** reshaping to `[B, H, L, D]` and **before** RoPE. Preserve this order:

```
q = q_proj(x).reshape(B, L, H, D).transpose(B, H, L, D)
q = rms_norm(q, q_norm_weight, eps)
q = rope(q, cos, sin)
```

The same for K. Do not apply QK-norm before the head reshape, and do not apply it to V.

### 8.3 RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot w.
$$

The mean is over the last dimension. Compute in fp32 and cast back. The epsilon comes from the preset; do not substitute `1e-6` by habit.

### 8.4 SwiGLU

$$
\text{MLP}(x) = W_{\text{down}}\left(\text{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right).
$$

`silu(x) = x * sigmoid(x)`. The gate and up projections are separate tensors in HF; keep them separate.

## 9. Objectives

### 9.1 Causal LM

Inputs:

```python
input_ids = batch                      # int32/int64 [B, L]
labels = batch                         # same tensor, cloned in the oracle
```

HF shifts internally and computes cross-entropy over positions $1 \ldots L-1$:

$$
\mathcal{L}_{\text{CLM}} = \frac{1}{B(L-1)} \sum_{b=1}^{B} \sum_{t=1}^{L-1} -\log p_\theta(x_{b,t} \mid x_{b,<t}).
$$

JAX may (a) pass full logits to a shift-aware loss, or (b) slice `logits[:, :-1]` against `input_ids[:, 1:]`. Both are acceptable only if the reduction count is exactly $B(L-1)$ and the loss matches. Do not ignore the first target twice or average per-row means.

### 9.2 Masked LM

The oracle's `MaskedLM` is the contract:

1. `maskable = ~isin(input_ids, special_ids)`.
2. Group positions into selection units (token, span, PMI, PMI-ngram, whole-word, variant).
3. For each unit, draw $u \sim U(0,1)$ at the unit's representative position; select the unit if $u < \text{rate}(\text{step})$.
4. A position is selected only if its unit is selected **and** it is maskable.
5. `labels = input_ids` at selected positions, `IGNORE_INDEX = -100` elsewhere.
6. For each selected position, draw an action $a \sim U(0,1)$:
   - $a < \text{mask\_share}$: replace with `<mask>`;
   - $\text{mask\_share} \le a < \text{mask\_share} + \text{random\_share}$: replace with a random id in `[first_regular_id, vocab_size)`;
   - next `variant_share` interval: replace with a spelling variant if available, else random;
   - remainder: keep the original token.
7. Defaults: `mask_share=0.8`, `random_share=0.1`, `variant_share=0.0`. When `variant_share > 0`, the oracle first takes it from `random_share`, then from `mask_share`; JAX must reproduce the adjusted shares, not the raw values.
8. `first_regular_id = max(special_ids) + 1`.

The loss is the mean cross-entropy over selected (non-`-100`) positions:

$$
\mathcal{L}_{\text{MLM}} = \frac{1}{N_{\text{sel}}} \sum_{(b,t):\,\text{label}\ne -100} -\log p_\theta(x_{b,t} \mid \tilde{x}_b).
$$

`N_sel` is counted after masking, not `B*L*rate`. The denominator must match the oracle exactly; this is a common parity bug.

### 9.2b Masking-unit implementation verdict

Implemented in `src/hamic_jax/objectives.py` with NumPy:

- `token`, `span`, `pmi`, and `pmi_ngram` selection units;
- `whole_word_masking`, applied as the outermost wrapper exactly like the oracle
  (`src/hamic/training/trainer.py:227–240`), including over `NgramUnits`;
- PMI-ngram mining: complete-syllable windows, unigram-through-`max_n+1` counters,
  document filtering, dominant-extension filtering, the minimum over all compositions of
  the PMI score, and the `discount_k` score, matching `src/hamic/training/ngram_units.py`;
- cache file `<tokenized>/pmi-ngrams-<n>-<min_count>-<min_doc_count>-<min_pmi>.npz`.

Not implemented in v1: the spelling-variant corruption table (`variant_share > 0`).
`VariantTable` in the oracle depends on `hamic.text` and the packaged teencode lexicon;
the JAX backend refuses `variant_share > 0` at construction instead of silently
approximating it. Configs `proxy-v0.2/e-v02-e7-variant{10,20}.yaml` are therefore out of
scope until that table is ported.

### 9.3 Evaluation masking

Evaluation uses the same objective with:

- `eval_mask_rate` is optional; when unset (or falsy) it falls back to the training
  `mask_rate`, matching `MaskRateSchedule(cfg.eval_mask_rate or cfg.mask_rate)`;
- a fixed seed offset `1 << 40` plus the batch index, so every checkpoint sees the same masks;
- `shuffle=False` for the validation loader;
- `ceil(eval_sequences / micro_batch_size)` batches, the same construction as the
  oracle. `eval_sequences` defaults to 256 and `eval_every_steps` to 500.

### 9.3b Sliding-window verdict

Resolved verdict: HAMIC's pinned `ModernBertConfig` sets
`layer_types=["full_attention"] * num_layers` and does **not** pass a sliding window
(`src/hamic/modeling/factory.py`), so every layer attends bidirectionally over the whole
document. The JAX encoder applies one global bidirectional mask and is therefore faithful.
`local_attention`/`sliding_window` in a preset are recorded for shape compatibility only;
they MUST NOT be applied until the oracle does.

### 9.4 Sparse prediction

The oracle can compute the MLM head only on labelled positions (`sparse_prediction=True` for ModernBERT). This is an **operational** optimization: it must produce the same loss and gradients as the dense head within tolerance, because unlabelled positions contribute zero to the loss and zero gradient.

v1 JAX uses the dense head. If sparse prediction is added later, it must use a static capacity (for example, `ceil(rate * B * L) + margin`), pad the gathered positions, and mask the padding out of the loss. Dynamic shapes are not allowed in the jitted train step.

## 10. Optimizer and schedule

### 10.1 AdamW

Match the oracle's `torch.optim.AdamW` semantics:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t},
$$
$$
\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda \theta_{t-1}\right).
$$

Defaults: `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`, `grad_clip=1.0`,
`min_lr_ratio=0.1`, `decay_ratio=0.2`. Weight decay applies only to parameters with
`ndim >= 2`; biases, norms, and scalars are excluded. The JAX mask is derived from the
parameter tree, not a hand-written list.

The optimizer step starts at 1 for the first update; bias correction uses the **global** step, not the local process step. On resume, restore `step` and the moments.

Implementation: `optax.chain(clip_by_global_norm(grad_clip), optax.adamw(...))`, where
`hamic_jax.optim.clip_by_global_norm` is the JAX form of §10.3. The pretraining
weight-decay mask is `leaf.ndim >= 2`; downstream fine-tuning additionally excludes
parameters whose name contains `norm`, matching `src/hamic/downstream/finetune.py`.
Downstream task heads mirror the stock classes: `ModernBertPredictionHead`
(dense → GELU → LayerNorm) followed by a linear classifier, applied after CLS pooling
for sequence/pair tasks and to every token for token/QA tasks. The pretrained MLM
`head.*` and `decoder.*` tensors are dropped before fine-tuning because
`from_pretrained` re-initialises the task head.
Optax's bias correction counts `optimizer.update` calls, so restoring the Optax state
restores the count; the checkpoint stores that state as flat leaves and rebuilds its
treedef with `optimizer.init(params)` at load time.

### 10.2 Gradient accumulation

The oracle divides the loss by `grad_accum_steps` before backward, or equivalently averages gradients over the window. JAX must use the same reduction:

$$
g = \frac{1}{A} \sum_{a=1}^{A} \nabla_\theta \mathcal{L}(\theta; x_a).
$$

Do not sum without dividing, and do not average the loss after the fact. The number of optimizer steps is `ceil(total_tokens / (seq_len * micro_batch_size * grad_accum_steps))`, matching the oracle's accounting. Resolved against `src/hamic/training/config.py:268`: the JAX `TrainingConfig.total_steps` uses `math.ceil`, and `warmup_steps` uses `max(1, round(total_steps * warmup_ratio))`.

### 10.3 Gradient clipping

Global norm clipping:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2 + \epsilon}\right), \quad c=1.0.
$$

The norm is over all trainable parameters, computed in fp32. The oracle's
`clip_grad_norm_` divides by `||g|| + 1e-6`; Optax's `clip_by_global_norm`
omits that epsilon and clips only when `||g|| >= c`. Because the final-audit
rule forbids substituting a different formula, `src/hamic_jax/optim.py` ships
`clip_by_global_norm` in JAX with the oracle's `+1e-6` factor. It stays a
single global reduction; on a TPU mesh XLA lowers the reduction to an
all-reduce. Downstream fine-tuning uses the same transform.

### 10.4 Learning-rate schedule

The oracle's WSD schedule:

- warmup for `warmup_steps` steps, linear from `learning_rate / warmup_steps` at
  step 0 to `learning_rate` at step `warmup_steps - 1`;
- stable at `learning_rate`;
- decay for `decay_ratio * total_steps` steps to `min_lr_ratio * learning_rate`.

The exact boundary rounding (floor/ceil) must match the oracle. Implement the schedule as a pure function of the global step; do not store a Python float that drifts on resume. JAX returns fp32 and casts at use.

Implementation uses `optax.linear_schedule` for warmup with
`init_value=peak/warmup` and `transition_steps=max(1, warmup - 1)`, so step 0 is
`peak/warmup` and step `warmup - 1` is `peak`, exactly the oracle's
`(step + 1) / warmup`. WSD then joins a constant hold and the linear decay above; cosine
joins the same warmup to `optax.cosine_decay_schedule` with
`decay_steps = total - warmup`, reproducing the oracle's
`floor + (1 - floor) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))`.

Current recipes:

| Run | LR | Warmup | Decay | Min LR | Tokens/step | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| `hamic-es-v014` | 8.0e-4 | 3% | 20% | 10% | 393,216 | 8.1e10 |
| `hamic-em-v05` | 5.0e-4 | 4% | 20% | 10% | 262,144 | 2.0e10 |

JAX reads these from the same YAML; it does not duplicate the numbers.

## 11. Data order

The loader is part of the model identity because it determines the gradient sequence. JAX must reproduce:

- the per-epoch document permutation, including source-weighted repeats;
- the `shuffle` flag;
- cursor semantics on resume;
- exact `bos_id`/`eos_id`;
- `seq_len` and `batch_size` per device (see [04-tpu-runtime.md §4](04-tpu-runtime.md#4-global-batch-and-sharding)).

It need not reproduce torch's RNG bitstream for **masking** if the JAX run uses its own documented masking RNG; masking is a stochastic regularizer, not a fixed function. However:

- the masking **policy** must be identical;
- a resumed JAX run must reproduce its own masks exactly;
- parity fixtures use precomputed masks so forward/loss comparisons are deterministic.

For strict trajectory parity with a specific GPU run, the mask stream is part of the identity and must be exported/replayed. This is open; see [07-roadmap.md](07-roadmap.md).

## 12. Initialization

The oracle uses HF's `_init_weights` (normal with std 0.02, plus architecture-specific residual scaling where applicable). JAX initialization must either:

1. consume converted oracle weights (preferred for parity and continuation), or
2. reproduce the same distribution and document that a from-scratch run is a **new identity**, not a bitwise continuation.

Parity tests always use converted weights. Distribution tests compare mean/std per tensor family, not individual draws.

## 13. Tolerances

Provisional gates, recalibrated after the tiny-model run:

| Quantity | fp32 CPU | bf16 TPU |
|---|---|---|
| Logits, max abs | `1e-5` | `2e-2` |
| Logits, max relative | `1e-4` | `2e-2` |
| Loss | `1e-4` | `5e-3` |
| Gradient cosine | `0.9999` | `0.999` |
| One-step parameter delta, max abs | `1e-6` | `1e-4` |
| Parameter count | exact | exact |
| Shapes/dtypes | exact | exact |

Rules:

- Compare in fp64 where the oracle allows it; otherwise fp32 to fp32.
- A tolerance failure is a bug until proven to be pure floating-point reduction order. "TPU is different" is not a waiver.
- Report the actual max error, not only pass/fail.
- Calibrate on a 2-layer, hidden-64 model first, then on each real preset.
- bf16 comparisons use converted weights and the same bf16 inputs; do not compare bf16 against fp32 and call the difference a bug.

## 14. Not required

The following differences are explicitly allowed:

- different attention kernel (reference XLA vs `dot_product_attention` vs Pallas), provided logits pass §13;
- different parameter layout inside a fused tensor, provided names/shapes map exactly at the boundary;
- different checkpoint serialization (flat safetensors vs torch), provided the manifest and
  conversion are verified;
- different masking RNG stream, provided the policy matches and resume is exact for that run;
- different compilation cache, mesh, and sharding, provided gradients and optimizer steps pass §13;
- different throughput and memory.

Everything else is a parity defect.