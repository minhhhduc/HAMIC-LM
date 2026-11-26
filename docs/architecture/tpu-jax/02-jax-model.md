# 02 — JAX model design

This document describes the JAX implementation of the HAMIC model families. The mathematical contract is [01-parity-contract.md](01-parity-contract.md); this document covers code structure, parameter layout, kernels, dtype, and compilation.

## 1. Package layout

Add a sibling package, not a subpackage that imports torch:

```
src/hamic_jax/
  __init__.py
  config.py          # YAML -> typed config; rejects unknown keys
  presets.py         # model identity -> shapes, RoPE, norm, tying
  params.py          # init, HF state_dict <-> pytree, parameter count
  rope.py            # RoPE tables and application
  norms.py           # LayerNorm, RMSNorm
  attention.py       # reference, dot_product_attention, masks
  mlp.py             # GeGLU, SwiGLU
  heads.py           # MLM head, LM head, tied views
  model.py           # encoder/decoder forward, layer stacks
  objectives.py      # MLM/CLM loss and reductions
  data.py            # NumPy packer, masking, cursor (port of hamic.training)
  optim.py           # Optax AdamW, WSD schedule, clipping, accumulation
  train.py           # jitted train step, eval step, loop
  checkpoint.py      # safetensors + HAMIC manifest/publish
  export.py          # safetensors + HF config/tokenizer
  cli.py             # `hamic-jax` entry point
tests/jax/
  fixtures/          # tiny configs, synthetic shards, oracle tensors
  test_*.py
```

Packaging changes (design, not applied yet):

- `pyproject.toml`: add `src/hamic_jax` to the package list;
- runtime dependencies: `jax[tpu]`, `optax`, `safetensors`, `huggingface_hub`, `numpy`,
  `tokenizers`; no Flax and no Orbax;
- optional extra `parity = ["torch", "transformers", "safetensors"]` for CPU fixtures;
- console script `hamic-jax = hamic_jax.cli:main`.

`hamic_jax` must be importable without torch and without importing `hamic`. It reads
`configs/*.yaml`, `tokenizer.json`, and tokenized manifests directly. If it needs a pure
dataclass, define it in `hamic_jax.spec`; do not add or extract a shared module in `src/hamic`,
because that would edit the GPU path.

## 2. Parameter tree

Parameters are a flat `dict[str, jax.Array]` whose keys are the **exact** names from the HF `state_dict`:

```
model.embeddings.tok_embeddings.weight
model.layers.0.attn.Wqkv.weight
model.layers.0.mlp.Wi.weight
...
```

Why flat:

- conversion from safetensors is a dictionary rename, not a structural rewrite;
- export is the inverse operation;
- every leaf has a stable name for sharding, weight decay, checkpointing, and error messages;
- comparing parameter sets with the oracle is a set comparison.

The cost is that layer code indexes by formatted keys; that is acceptable and explicit.

### 2.1 Tied embeddings

Tied weights are stored **once** under the canonical embedding name. The LM head / decoder reads the same array. Export writes the canonical tensor once and relies on the HF config (`tie_word_embeddings` or the architecture's equivalent) to reconstruct the tie; it does not write a duplicate alias key. The optimizer must see one leaf, not two aliases; otherwise the tied weight receives two updates and parity breaks.

The converter records aliases in a manifest:

```json
{"aliases": {"lm_head.weight": "model.embed_tokens.weight"}}
```

The JAX tree never contains the alias as a separate leaf.

### 2.2 Initialization

Two supported paths:

1. **Converted weights (default):** load oracle safetensors, rename keys, cast to fp32 master. This is the only path used for parity and for continuation.
2. **From scratch:** initialize each tensor with the HF distribution for its family (normal std 0.02, residual-output scaling where HF applies it), using an explicit `jax.random.PRNGKey` per tensor name. A from-scratch run is a new identity; it is not a bitwise continuation of a GPU run.

Parameter count is computed by `sum(x.size for x in tree.leaves())` and must equal the measured preset count. Do not use an analytic formula.

## 3. Forward API

```python
def encoder_forward(
    params: dict[str, Array],
    input_ids: Array,          # int32 [B, L]
    attention_mask: Array,     # bool [B, 1, L, L] document mask
) -> Array:                    # logits [B, L, vocab]

def decoder_forward(
    params: dict[str, Array],
    input_ids: Array,          # int32 [B, L]
) -> Array:                    # logits [B, L, vocab]

def loss_fn(
    params: dict[str, Array],
    batch: Batch,
) -> tuple[Array, dict[str, Array]]:
    ...
```

`Batch` is a frozen pytree:

```python
class Batch(NamedTuple):
    input_ids: Array        # masked input for MLM; raw for CLM
    labels: Array           # original ids at selected positions; -100 elsewhere
    attention_mask: Array | None
```

The model functions are pure; the train step closes over config and sharding, not over mutable module state.

## 4. Layers

### 4.1 Embedding and norms

- Token embedding: `take(params["..."], input_ids)`.
- LayerNorm: mean/variance over the last axis, `eps` from config, optional bias, fp32 compute, cast back.
- RMSNorm: mean of squares over the last axis, `eps` from config, no bias, fp32 compute, cast back.

Never share a norm implementation between LayerNorm and RMSNorm by a flag that changes broadcasting; keep two small functions and test both against the oracle.

### 4.2 Attention

The reference path is deliberately simple:

```python
scale = 1.0 / math.sqrt(head_dim)
scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)
out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
```

The reference path is a teaching aid only. The shipped path is
`jax.nn.dot_product_attention(query, key, value, mask=mask)`: it keeps logits in
float32, fills masked logits with a finite dtype minimum (so an all-masked row
does not become NaN in bf16), and lets XLA fuse the three matmuls and the
softmax on TPU. GQA is native — `key`/`value` stay `[B, L, G, D]` when
`H % G == 0`, so `repeat_kv` is not materialised. The mask convention is
verified: in JAX, `True` means **keep**.

The parity suite compares this path to a reference implementation before it
compares to PyTorch.

Pallas/splash attention is a later optimization. It must pass the same parity suite with the document mask; if splash cannot express the block-diagonal document mask, it is not usable for the encoder.

### 4.3 RoPE

```python
def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

`cos`/`sin` are `[L, D]` (or broadcast `[1, 1, L, D]`) and are computed once per sequence length, cached by `(theta, head_dim, dtype, L)`. Cache keys must include dtype; a bf16 table reused for fp32 is a parity bug.

Positions are `arange(L)` per row, as specified in the contract. Do not reset at document boundaries.

### 4.4 MLP

- GeGLU: split the fused `Wi` output into gate/up if HF stores it fused; apply exact GELU to the gate; multiply; down-project.
- SwiGLU: separate `gate_proj`/`up_proj`; apply SiLU to the gate; multiply; down-project.

The activation implementation must match HF's exact function. In particular, `jax.nn.gelu` defaults to the tanh approximation in some versions; pass `approximate=False` explicitly, or implement the erf form, after checking the oracle.

### 4.5 Heads

- LM head: `logits = x @ embed_weight.T` for tied embeddings; add bias only if the oracle has one.
- MLM head: dense → activation → norm → decoder, in exactly the HF order. The decoder is tied to the token embedding where configured; the dense/norm may have bias depending on the preset.

Do not fuse the MLM head into the embedding projection unless the fused form is algebraically identical and passes parity; the dense/norm between them usually prevents it.

## 5. Loss

### 5.1 CLM

Use `optax.softmax_cross_entropy_with_integer_labels` on shifted logits:

```python
logits = decoder_forward(params, input_ids)[:, :-1, :]
targets = input_ids[:, 1:]
loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
```

The mean is over all `B*(L-1)` positions. Do not use `reduction="mean"` per row and then average rows.

### 5.2 MLM

```python
logits = encoder_forward(params, input_ids, attention_mask)
loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
valid = labels != -100
loss = jnp.sum(jnp.where(valid, loss, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
```

The denominator is the number of selected positions. `jnp.maximum(..., 1)` prevents NaN on an empty batch in debug; in a real step the count is nonzero, and the oracle's behavior on zero must be matched explicitly.

Compute the cross-entropy in fp32 even when logits are bf16; cast logits to fp32 for the loss. This is the stability policy in D-T06.

## 6. Dtype policy

| Object | Dtype |
|---|---|
| Master parameters | fp32 |
| Compute copy of weights | bf16, derived from master |
| Activations | bf16 |
| Attention scores / softmax | fp32 |
| Norm statistics | fp32 |
| Loss | fp32 |
| Optimizer moments | fp32 |
| RNG keys | uint32 |

TPU matmuls take bf16 inputs and accumulate in fp32. Use `preferred_element_type=jnp.float32` where the API supports it, or cast the result; verify with the parity suite. Do not keep a second bf16 master; the fp32 master is the source of truth, and the bf16 copy is a cache.

## 7. JIT, scan, and rematerialization

### 7.1 Train step

```python
class TrainState(NamedTuple):
    params: dict[str, Array]
    opt_state: optax.OptState
    step: Array

@jax.jit
def train_step(state, batch, rng):
    (loss, grads) = jax.value_and_grad(loss_fn)(state.params, batch)
    grads = clip_by_global_norm(grads)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainState(params, opt_state, state.step + 1), loss
```

The real step includes gradient accumulation: scan over $A$ microbatches, average the gradients, then update once. The scanned function must not update the optimizer.

Donation is enabled for all four step arguments (`params`, optimizer state, ids,
labels) because the loop replaces each on return and no caller reads the donated
buffers. The scan carries only the gradient accumulator, not a stack of
per-microbatch gradients, so peak memory is one gradient tree plus one optimizer
state. If a caller ever reads the pre-step arrays, remove the donation before
adding that caller; donation misuse corrupts silently rather than raising.

### 7.2 Layer scan

v1 unrolls layers for parity and debuggability. v2 may stack same-shaped layer parameters into `[N, ...]` and use `jax.lax.scan`:

```python
def stack_layers(params, prefix, n_layers):
    ...
def unstack_layers(stacked, prefix, n_layers):
    ...
```

Stacking is a training-time layout optimization; the checkpoint and export still use per-layer names. The stack/unstack functions must be exact inverses and are covered by a round-trip test.

`jax.checkpoint` (remat) wraps each layer or the attention block when activation memory requires it. Remat changes compute, not math; it must pass the same parity tolerance.

### 7.3 Compilation stability

- Shapes are static: `[B, L]` fixed per run, `L = seq_len`.
- Do not let a Python integer depend on data (no dynamic gather sizes).
- Keep the train step and eval step as separate jits; eval uses a different batch shape only if `eval_sequences` requires it, and that triggers a separate compilation by design.
- Log the compiled executable count; an unexpected increase means a shape or sharding leak.

## 8. Sharding

### 8.1 Mesh

```python
mesh = jax.sharding.Mesh(devices, ("data", "fsdp"))
```

- ES/EM/D500M: `data` only, parameters replicated, batch sharded over `data`.
- D1B: `data` × `fsdp`; parameters sharded over `fsdp`, batch over `data`.
- The exact mesh shape is a runtime decision from the device count; it is recorded in the run manifest.

### 8.2 Parameter sharding

Use `NamedSharding` with `PartitionSpec` per leaf:

- embeddings / LM head: sharded on the vocabulary axis when FSDP is active, replicated otherwise;
- attention/MLP matrices: sharded on the output or input axis according to the FSDP convention, consistently for forward and backward;
- norms/biases: replicated (they are small).

Do not hand-place leaves with `jax.device_put` per tensor in the hot path. Build a sharding template once, `jax.device_put` the tree, and pass `in_shardings`/`out_shardings` to jit.

### 8.3 Collectives

Gradient all-reduce over `data` and reduce-scatter/all-gather over `fsdp` are inserted by GSPMD from the sharding annotations. v1 relies on GSPMD; `shard_map` is a later optimization for attention/MLP if profiling shows unnecessary collectives. Any `shard_map` rewrite must pass the parity suite on the same mesh.

### 8.4 Global batch

The global batch per optimizer step must remain

$$
B_{\text{global}} = B_{\text{micro}} \times A \times N_{\text{data}},
$$

where $B_{\text{micro}}$ is the per-device micro-batch, $A$ is gradient accumulation, and $N_{\text{data}}$ is the data-parallel degree. To preserve a GPU run's `tokens_per_step`, choose $B_{\text{micro}}$ and $A$ so the product equals the GPU product:

$$
B_{\text{micro}}^{\text{TPU}} \times A^{\text{TPU}} = \frac{B_{\text{micro}}^{\text{GPU}} \times A^{\text{GPU}}}{N_{\text{data}}}.
$$

If the division is not integral, the run has a different global batch and is a **new identity**; record it in the run manifest and do not compare loss curves step-for-step with the GPU run.

## 9. Optimizer integration

- Optax `adamw` with `weight_decay` mask derived from `ndim >= 2`; downstream
  fine-tuning adds `"norm" not in name`, as the oracle does.
- `hamic_jax.optim.clip_by_global_norm(1.0)` before the update: a JAX transform
  with the oracle's `+1e-6` denominator (`optax.clip_by_global_norm` omits it).
- WSD schedule as a pure function of `step`; use `optax.join_schedules` or a custom `Schedule` that matches the oracle's rounding.
- Bias correction uses the global step; Optax's `adam` uses the count in `opt_state`, so restore that count exactly.
- Accumulation averages gradients before clipping. Clipping after averaging is the oracle's order; do not clip each microbatch separately.

The optimizer state is Optax's own pytree (Adam's `mu`, `nu`, and count, wrapped by the
chain/mask transforms). It is saved as flattened leaves plus the Optax treedef rebuilt at
load time; the HAMIC manifest records the step and the parameter tree hash.

## 10. Evaluation

Evaluation is a separate jitted function with no gradient:

```python
@jax.jit
def eval_step(params, batch):
    logits = model_forward(params, batch)
    loss = ...
    return loss, nats, target_bytes
```

For CLM BPB, accumulate:

$$
\text{nats} = \sum \text{loss} \times N_{\text{targets}}, \quad
\text{BPB} = \frac{\text{nats}}{\ln(2) \sum \text{bytes}}.
$$

The per-token byte lengths are precomputed by the tokenizer/`token_byte_lengths` utility and shipped as an array; the TPU must not decode tokens or call the tokenizer during evaluation.

Evaluation masks use the fixed seed offset and `eval_mask_rate` from the config. Evaluation batch construction uses `shuffle=False` and the same `eval_sequences` count as the oracle.

## 11. Debugging hooks

- `jax.debug.print` for a small number of steps, gated by an env var; never in the hot loop by default.
- `jax.experimental.checkify` in debug builds for NaN/Inf, fully masked attention rows, and label-range checks.
- A `debug_checks` config block that is off in production and on in the parity suite.
- Save the first batch, the first loss, and the first gradient norm as fixtures; these catch most porting mistakes before TPU time.

## 12. Open implementation questions

| Question | How it is resolved |
|---|---|
| Fused vs split QKV layout in ModernBERT | Read the pinned HF `state_dict`; converter generates the mapping; round-trip test |
| Exact GELU variant | Compare against HF on a fixture; pass `approximate=False` if that is the oracle |
| LayerNorm bias presence | Read the HF config; reject a mismatch instead of guessing |
| `jax.nn.dot_product_attention` mask convention | Compare reference and optimized paths on the document-mask fixture |
| Sparse MLM head | Off in v1; static-capacity design if enabled |
| Layer scan vs unroll | Unroll first for parity; scan only after a round-trip test |
| Sharding template | Benchmark on the target TPU; record in the run manifest |

None of these changes the model identity; they are implementation choices constrained by the parity contract.