# 05 — Checkpoints and export

The TPU run must be resumable, auditable, and exportable to the existing Hugging Face release format. This document defines the checkpoint contents, the publish protocol, resharding, and the JAX ↔ safetensors/HF boundary.

## 1. Checkpoint contents

A committed checkpoint contains everything needed to continue **and** to explain the run:

```
step-000000123/
  params.safetensors      # flat dict[str, Array], HF state_dict names
  optimizer.safetensors   # Optax state leaves in flatten order
  meta.json               # step, cursor, optimizer leaf count, run metadata
  metrics.jsonl           # metrics since the previous checkpoint
  manifest.json           # every payload file: path, size, sha256
  COMMITTED.json          # manifest sha256, created last
```

`metadata.json` must include:

| Field | Why |
|---|---|
| `run_id`, `global_step` | identity and resume |
| `consumed_tokens` | budget accounting |
| `data_cursor` | `{epoch, position, token_offset, consumed_tokens}` |
| `mask_rng` | seed and micro-step for the mask stream |
| `config` | the full train config, not a path |
| `model` | the preset/config used to build the tree |
| `model_spec_sha256` | hash of the resolved preset |
| `tokenizer_sha256` | the released tokenizer |
| `tokenized_manifest_sha256` | the exact data artifact |
| `code_revision` | git commit and dirty flag |
| `environment` | JAX, libtpu, Python, NumPy versions |
| `mesh` | device type, chip count, mesh shape |
| `global_batch_tokens` | trajectory identity |
| `precision` | bf16 compute / fp32 master policy |
| `param_tree_sha256` | canonical hash of names/shapes/dtypes |

The parameter tree hash is computed over sorted `(name, shape, dtype)` tuples, not over values. It catches a silently changed architecture before restore.

Credentials are forbidden in checkpoint metadata. `HF_KEY` is read from the process environment at runtime and is never written to a checkpoint, manifest, metric, or export.

## 2. Payload layout

The implementation does **not** depend on Orbax: Orbax's checkpoint import is broken in the pinned
environment, and its API changes across JAX releases. The payload is two safetensors files,
written atomically inside the publish-last protocol:

- `params.safetensors` is the flat `dict[str, Array]`, keyed by exact HF `state_dict` names;
- `optimizer.safetensors` stores the Optax state's leaves in `jax.tree_util.tree_flatten`
  order, keyed `opt_00000`, `opt_00001`, ...;
- `meta.json` records `step`, `cursor`, `optimizer_leaves`, and the run metadata.

On restore the caller builds the Optax transform from the config and parameter tree,
initializes it to obtain the treedef, and substitutes the saved leaves by flat order. The
treedef therefore comes from Optax itself; no Optax internals are serialized and no custom
optimizer state format is invented. Dtype/shape checks are strict; a mismatch is an error.

Checkpointing is synchronous. The publish protocol below is the atomicity boundary: a
crash between payload write and `COMMITTED.json` leaves an incomplete prefix that resume
ignores. A future async writer can be added behind the same manifest without changing the
payload format.

## 3. Publish protocol

Reuse the HAMIC publish-last sequence, with safetensors as the payload writer:

1. Write the checkpoint to a local temporary directory.
2. Build `manifest.json` over every file with size and SHA-256.
3. Upload payload files and `manifest.json` in one batch where the store supports it.
4. Verify remote size and SHA-256 against the manifest.
5. Upload `COMMITTED.json` with the manifest hash.
6. Read back `COMMITTED.json`; only then update `LATEST.json`.
7. Delete the local temporary directory.

Invariants:

- `COMMITTED.json` is the only resumability marker; `LATEST.json` is a hint.
- A committed checkpoint is immutable. A retry that finds `COMMITTED.json` must raise, not overwrite.
- A partial upload is invisible to the resolver because it has no commit marker.
- The resolver picks the newest committed checkpoint whose metadata matches the requested run and, optionally, step.
- If the newest committed checkpoint fails verification, the resolver fails closed; it does not silently fall back to an older one unless the operator explicitly asks for a fallback.

The JAX runtime may use the existing `CheckpointPublisher`/`CheckpointResolver` logic if it can be made torch-free. Otherwise, port the protocol exactly and test it against the same fixtures. Do not invent a second manifest format.

## 4. Resharding

A checkpoint may be restored on a different mesh (for example, v5e-8 → v5p-8, or 8 → 16 chips). The requirements:

- the parameter tree names/shapes/dtypes match exactly;
- the optimizer state tree matches exactly;
- the loader accepts arrays on any device mesh; sharding is a runtime placement, not a
  payload property, so `jax.device_put` installs the tree on the target mesh;
- after restore, run a forward pass on a fixed fixture and compare logits to the pre-restore model;
- record the source and target mesh in the new checkpoint metadata.

Resharding must not change the mathematical state. If a parameter is replicated in the source and sharded in the target, the values must be identical after restore; the parity fixture catches a mis-stitched shard.

## 5. RNG and cursor

The JAX run has its own mask RNG, so the checkpoint stores:

- the base seed;
- the global micro-step (or the exact counter used to derive the per-step key);
- the data cursor.

On resume, the next mask is derived from `(seed, next_micro_step)`, exactly as an uninterrupted run would. The data cursor restores the packer position. Together they guarantee that resume reproduces the next batch and the next mask.

If a run must replay a GPU mask stream, the checkpoint additionally stores the mask stream identity or a pointer to a precomputed mask artifact. That is an open decision; the default JAX run does not.

## 6. PyTorch → JAX conversion

The converter is a one-way, deterministic transformation:

```
HF safetensors directory
  -> read tensors with safetensors.numpy (no torch required)
  -> rename keys by the mapping generated from the oracle's state_dict
  -> resolve tied aliases (drop the alias leaf)
  -> cast to fp32 master
  -> validate names, shapes, dtypes, parameter count
  -> write structural safetensors checkpoint + metadata
```

The mapping is **generated**, not hand-written from memory:

1. instantiate the oracle model on the meta device;
2. take `state_dict()` keys and shapes;
3. apply the architecture's rename table (for example, a fused `Wqkv` may stay fused; a split layout may need a concatenate/split op with a tested inverse);
4. assert the resulting set equals the JAX tree's key set;
5. assert the parameter count equals the measured preset.

If the converter needs a tensor transpose or a fused split, the operation is recorded in the mapping manifest and has a round-trip test.

## 7. JAX → HF export

The export target is the existing release layout, not a JAX-specific format:

```
config.json
model.safetensors            # or model-00001-of-000NN.safetensors + index
tokenizer.json
tokenizer_config.json
special_tokens_map.json
hamic_model.json             # HAMIC provenance, optional for HF load
```

Rules:

- `config.json` uses the stock architecture (`ModernBertForMaskedLM`, `LlamaForCausalLM`, `Qwen3ForCausalLM`) and the exact preset fields;
- `trust_remote_code` is never required;
- tied weights are written once, with `tie_word_embeddings` (or the architecture's equivalent) set so the HF loader reconstructs the tie; do not write duplicate safetensors keys;
- safetensors shards respect the size limit and include the standard index;
- tokenizer files are copied byte-for-byte from the released tokenizer; do not regenerate them;
- `hamic_model.json` records the source run, step, tokenizer hash, and parity report id;
- the export is deterministic given the checkpoint.

The exporter must not import torch. It writes safetensors + JSON; the verification step may use torch/transformers in a separate environment.

## 8. Export verification

For every exported model:

1. `AutoConfig.from_pretrained(path, trust_remote_code=False)` succeeds;
2. `AutoModelForMaskedLM.from_pretrained` or `AutoModelForCausalLM.from_pretrained` loads with no missing/unexpected keys (tied keys handled by the config);
3. parameter count equals the measured preset;
4. tokenizer loads and round-trips a fixed Vietnamese sample;
5. logits on a fixed fixture match the JAX model within §13 tolerance;
6. for CLM, a short greedy generation runs without error;
7. the model card/metadata records the parity report and the source checkpoint.

A mismatch in step 5 blocks the release. Do not "fix" it by relaxing the tolerance or by exporting the oracle's weights instead of the JAX weights.

## 9. Drills

Before a production run, execute these drills on the target store:

| Drill | Expected result |
|---|---|
| Kill after a checkpoint upload but before commit | checkpoint is invisible; resume uses the previous commit |
| Kill after commit | resume restores and continues |
| Corrupt one payload byte | verification fails; resolver rejects |
| Wrong tokenizer hash | abort before step 0 |
| Restore on a different mesh | logits match pre-restore fixture |
| Retry the same step | publish raises; no overwrite |
| Missing `LATEST.json` | resolver still finds the newest commit |

Each drill writes a short report with the checkpoint prefix, the failure injected, and the observed result.

## 10. Failure policy

- **Fail closed** on any hash, shape, dtype, tokenizer, config, or code mismatch.
- **Never** auto-repair a checkpoint by dropping tensors or casting dtypes.
- **Never** publish a checkpoint whose post-write verification has not completed.
- **Never** compare a resumed run to an uninterrupted run without checking the data cursor and mask step first.
- Keep the last $N$ committed checkpoints (configurable); deleting older ones is an explicit operation, not a side effect of publishing.