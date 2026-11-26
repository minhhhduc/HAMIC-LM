# 03 — Data plane

The TPU consumes artifacts produced by the existing CPU pipeline. It does not tokenize, deduplicate, crawl, or re-shard. This document defines the on-disk contract, the packing algorithm, masking, and how shards reach TPU hosts without downloading the corpus to the audit machine.

## 1. Principle

Data preparation and training are separate planes:

```
raw corpus -> filter/dedup -> frozen corpus -> tokenizer -> .npy shards + manifest
                                                               |
                                                               v
                                                     TPU host reads shards
```

The boundary is the tokenized directory. Everything upstream is unchanged HAMIC code; everything downstream is JAX. The tokenized directory is immutable; the TPU never writes to it.

## 2. Shard format

A tokenized directory contains:

```
tokenized_manifest.json
tokenizer.json
train/
  train-00000.tokens.npy
  train-00000.offsets.npy
  train-00000.sources.npy
  train-00001...
validation/
test/
```

Each shard is a triple of NumPy arrays:

| File | Dtype | Shape | Meaning |
|---|---|---|---|
| `<name>.tokens.npy` | `uint16` or `uint32` | `[N_tokens]` | concatenated document token ids, **no special tokens** |
| `<name>.offsets.npy` | `int64` | `[N_docs + 1]` | document boundaries; document `i` is `tokens[offsets[i]:offsets[i+1]]` |
| `<name>.sources.npy` | `uint8` | `[N_docs]` | source index into `manifest["source_ids"]`; optional for old directories |

Rules:

- `offsets[0] == 0`, `offsets[-1] == N_tokens`, strictly non-decreasing;
- no document is empty unless the source produced an empty document (the tokenizer pipeline decides; the reader must not silently drop it);
- `sources` is required when `source_weights` is used; otherwise the packer must fail closed;
- token ids are `< vocab_size`; special ids are not present in shards;
- the dtype is determined by `vocab_size <= 65,536`; a reader must reject a dtype that cannot represent the vocabulary.

The JAX reader uses `np.load(..., mmap_mode="r")` for local shards. It must not load a full split into RAM.

## 3. Manifest

`tokenized_manifest.json` is the authority. Top-level fields written by `tokenize_corpus`:

```json
{
  "schema_version": 1,
  "created_at_utc": "...",
  "format": "hamic-tokenized-npy-v1",
  "special_tokens_added": false,
  "dtype": "uint16",
  "tokenizer": {
    "name": "...",
    "vocab_size": 20544,
    "sha256": "...",
    "text_preparation": {"lookalikes": "..."}
  },
  "source_corpus": {
    "corpus_version": "...",
    "manifest_sha256": "..."
  },
  "source_ids": ["wikipedia-vi-20231101", "..."],
  "splits": {
    "train": {
      "documents": 0,
      "tokens": 0,
      "shards": [
        {
          "name": "train-00000",
          "documents": 0,
          "tokens": 0,
          "files": {
            "tokens": {"path": "train-00000.tokens.npy", "sha256": "..."},
            "offsets": {"path": "train-00000.offsets.npy", "sha256": "..."},
            "sources": {"path": "train-00000.sources.npy", "sha256": "..."}
          }
        }
      ]
    },
    "validation": {"...": "..."},
    "test": {"...": "..."}
  },
  "code": {"revision": "...", "dirty": false}
}
```

The JAX reader must:

1. parse the manifest, reject unknown `schema_version`/`format`;
2. verify `tokenizer.sha256` against `tokenizer.json` before any step;
3. verify `dtype` against the preset vocabulary;
4. verify each shard's file size and SHA-256 when `verify=True` (the default for run start; optional for every resume);
5. expose `source_ids` with stable order; source weights refer to these strings, not indices.

Do not invent a second manifest schema. If a field is missing, fail closed; do not default it silently.

## 4. Document order and source weights

The per-epoch order is exactly `PackedBatches._permutation`:

```python
rng = np.random.default_rng([seed, epoch])

if source_weights is None:
    order = rng.permutation(documents) if shuffle else np.arange(documents)
else:
    table = np.array([source_weights.get(s, 1.0) for s in source_ids], dtype=np.float64)
    weights = table[sources]                       # per document
    whole = np.floor(weights)
    extra = rng.random(documents) < (weights - whole)
    order = np.repeat(np.arange(documents), (whole + extra).astype(np.int64))
    if order.size == 0:
        raise ValueError("source_weights leave no documents to visit")
    order = rng.permutation(order) if shuffle else order
```

Notes:

- `source_weights` values are **visits per document per epoch**, not sampling probabilities;
- a weight of 2.0 means every document from that source is visited twice per epoch;
- a fractional part is a Bernoulli extra visit;
- unknown source names are an error, not a silent 1.0;
- the same `(seed, epoch)` must produce the same order in JAX; `np.random.default_rng` is the contract, not torch's generator.

The JAX data plane may reuse this NumPy code directly if it is torch-free. It must not reimplement it with a different RNG and call it "equivalent".

## 5. Packing

`PackedBatches.next_batch` is a state machine. The JAX port must preserve the state:

```python
state = (epoch, position, token_offset, consumed_tokens)

def next_batch(state):
    out = empty((batch_size, seq_len), int32)
    for row in range(batch_size):
        filled = 0
        while filled < seq_len:
            piece = piece_at(order[position])       # [bos] + doc + [eos]
            take = min(seq_len - filled, piece.size - token_offset)
            out[row, filled:filled+take] = piece[token_offset:token_offset+take]
            filled += take
            token_offset += take
            if token_offset >= piece.size:
                position += 1
                token_offset = 0
                if position >= order.size:
                    epoch += 1
                    position = 0
                    order = permutation(epoch)
    consumed_tokens += batch_size * seq_len
    return out, state
```

That is the whole algorithm. It is intentionally simple, and the bugs are in the boundaries:

- `take` uses the remaining piece length, not the remaining row length;
- a piece can be larger than a row, so a document can span many rows;
- the state advances after the take, not after the row;
- the epoch rollover happens when the position reaches the order length, not when the row ends;
- `consumed_tokens` counts output tokens, including `<s>`/`</s>`, not corpus tokens.

The packer returns `int32` on the host (vocab < 2^31); cast to `int32` for JAX, not `int64`, to halve host-to-device bytes. The model embedding `take` accepts int32.

## 6. Global batch and host/device distribution

Baseline v5e-8 is a single host with 8 chips. v1 packs the **global micro-batch** on the host:

```python
rows = B_micro_per_device * N_data
global_batch = packer.next_batch(rows)          # [rows, L]
per_device = global_batch.reshape(N_data, B_micro_per_device, L)
sharded = jax.device_put(per_device, sharding)
```

This preserves the exact document order of a single stream: optimizer step `s` consumes a contiguous block of `rows * A` rows, split across devices and accumulation. It also preserves `tokens_per_step` if `B_micro_per_device * N_data * A` equals the GPU product.

For multi-host pods, each host must not independently start at the same cursor. The design is a `HostBatchStream` that assigns consecutive global batches to hosts:

```
host h at step s consumes global batch index s * N_hosts + h
```

The stream is deterministic; the host cursor includes `host_index` and `N_hosts`. If the packer cannot seek to a batch index cheaply (variable document lengths make seeking non-trivial), the fallback is one coordinator host that packs and scatters, which is slower but exact. The choice is an implementation decision; it must be recorded in the run manifest and covered by the resume drill.

## 7. Masking

Masking is a CPU/NumPy operation, not a TPU operation. The model receives `input_ids` (already masked) and `labels` (original ids at selected positions, `-100` elsewhere).

### 7.1 Policy

The policy is the `MaskedLM` contract in [01-parity-contract.md §9.2](01-parity-contract.md#92-masked-lm):

- selection units, mask rate, mask/random/variant shares;
- special ids never selected;
- labels only at selected positions;
- random replacements in `[first_regular_id, vocab_size)`.

### 7.2 RNG

Use a NumPy `Generator`:

```python
rng = np.random.default_rng([seed, micro_step])
```

This is deterministic across processes and platforms for the same NumPy version. The JAX run's masks are its own stream; they are **not** bitwise equal to torch's masks. That is allowed by the parity contract, provided:

- the policy is identical;
- `(seed, micro_step)` is stored in the checkpoint cursor, so resume reproduces the same masks;
- parity fixtures use precomputed masks, not RNG replay.

If a specific GPU run must be replayed exactly, export its mask stream or run the oracle's masking on CPU for every step. That is an open decision; see [07-roadmap.md](07-roadmap.md).

### 7.3 Units

| Unit | Implementation |
|---|---|
| `TokenUnits` | each position is its own unit |
| `SpanUnits` | geometric span length, capped at 5, seeded |
| `PmiUnits` | units joined by high-PMI pairs, table mined on CPU |
| `PmiNgramUnits` | PMI units extended to n-grams |
| `WholeWordUnits` | tokenizer word boundaries; requires a word-start table |
| `VariantUnits` | spelling-variant table; falls back to random when unavailable |

PMI mining is CPU-only and cached by `(corpus, tokenizer, mining config)`. The TPU never mines. The cached table is an artifact with a hash recorded in the run manifest.

## 8. Reading from object storage

The tokenized directory lives in GCS (or an equivalent object store). The TPU host needs random access to shards, but must not copy the full corpus to every host.

Recommended v1:

1. Download `tokenized_manifest.json` and `tokenizer.json` to the host (small).
2. Verify the manifest and tokenizer hash.
3. For each assigned shard, download `tokens`, `offsets`, and `sources` to a local SSD cache; verify SHA-256; `np.load(..., mmap_mode="r")`.
4. Evict shards only after the cursor has passed them for the current epoch; keep a prefetch window of 1–2 shards.
5. Never write to the GCS prefix; open it read-only.

Alternatives:

- `gcsfuse` mount with a local page cache: simplest, but latency and memory behavior are less predictable;
- `tensorstore`/`zarr` range reads: good for random access, but adds a dependency and a conversion step;
- `fsspec` whole-shard reads: acceptable for 50M-token shards, poor for random document access.

The choice is a deployment decision; the data contract is the `.npy` triple regardless. GCS uses the VM service account. If the store is Hugging Face instead, the token is read from `HF_KEY`; it is never placed in the manifest or shard cache.

## 9. Validation checklist

Before the first step:

- [ ] `tokenizer.sha256` matches the released tokenizer;
- [ ] `vocab_size` equals the preset;
- [ ] special ids match `src/hamic/tokenization/spec.py`;
- [ ] `dtype` can represent `vocab_size`;
- [ ] every shard's `offsets` is monotonic and ends at `len(tokens)`;
- [ ] every token id is `< vocab_size` (sampled or full scan on first run);
- [ ] `sources` exists if `source_weights` is non-empty;
- [ ] `source_ids` covers every key in `source_weights`;
- [ ] the first packed batch matches a fixture generated by the oracle;
- [ ] the first mask batch matches a fixture for the same policy and seed (policy check, not bitwise torch RNG).

## 10. Local audit without data

The audit machine must not download the corpus. It uses:

- a synthetic tokenized directory with 2–3 tiny shards, generated from a fixed token list;
- a manifest with the same schema and hashes;
- a tiny tokenizer spec with the same special ids;
- fixtures for a row containing two documents, a document split across rows, and an empty document.

These fixtures are enough to test the packer, cursor, mask policy, manifest verification, and resume. They are not enough to test real corpus statistics; that is a TPU/run-time check.