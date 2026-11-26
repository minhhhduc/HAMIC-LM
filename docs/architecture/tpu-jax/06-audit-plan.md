# 06 — Audit plan

The audit machine has no TPU and must not download the corpus. This plan defines what can be proven locally, what must wait for TPU, and what evidence is required before any claim.

## 1. Constraint

Allowed on the audit machine:

- static analysis of configs, presets, manifests, and code;
- CPU JAX and CPU PyTorch on synthetic fixtures;
- shape-only construction (`jax.eval_shape`, meta-device PyTorch);
- tiny synthetic tokenized directories written to `tmp_path`;
- checkpoint/export round-trips on tiny models;
- lint, format, and strict type checks.

Not allowed:

- downloading HAMIC corpora or tokenized splits;
- training a real model;
- calling a TPU;
- claiming throughput, MFU, or convergence.

## 2. Test layers

| Layer | What it proves | Runs locally |
|---|---|---|
| L0 static | configs, presets, parameter counts, special ids, manifest schema | yes |
| L1 unit | RoPE, norms, masks, packing, cursor, loss reduction, optimizer math | yes |
| L2 parity | JAX vs PyTorch on tiny models and converted weights | yes, CPU |
| L3 integration | full train step on synthetic shards, checkpoint/resume | yes, CPU |
| L4 export | safetensors + HF load + logits parity | yes, CPU |
| L5 TPU smoke | real devices, real sharding, real store | TPU only |
| L6 production | real corpus, real token budget | TPU only |

L0–L4 are the audit deliverable. L5–L6 are run-time gates.

## 3. Test matrix

### 3.1 L0 — static

- `test_preset_parameter_count`: build the shape-only parameter tree for every preset; sum leaf sizes; compare with the measured counts in [01-parity-contract.md §3](01-parity-contract.md#3-model-identities-and-presets).
- `test_preset_shapes`: layer count, hidden size, head count, head dim, FFN size, vocab, context.
- `test_special_ids`: tokenizer spec vs `src/hamic/tokenization/spec.py`.
- `test_manifest_schema`: parse a fixture manifest; reject an unknown `schema_version`/`format`; reject a missing tokenizer hash.
- `test_config_unknown_key`: the JAX parser rejects an unknown YAML key.
- `test_vocab_dtype`: 20,544 → `uint16`; 48,000 → `uint16`; >65,536 → `uint32`.
- `test_tied_weights_single_leaf`: the tree has one leaf per tied parameter; aliases live in metadata only.

### 3.2 L1 — unit

- `test_rope_matches_reference`: HF `rotate_half` vs JAX on a fixed tensor, including dtype cast.
- `test_layernorm_bias_and_eps`: oracle vs JAX, with and without bias.
- `test_rmsnorm_eps`: Qwen3/Llama epsilon, fp32 compute.
- `test_document_mask`: two documents in one row; cross-document positions are masked; positions do not reset.
- `test_causal_mask`: lower triangular; first row attends only to itself.
- `test_packer_two_documents`: a row containing `[bos] doc A [eos] [bos] doc B ...`.
- `test_packer_split_document`: a document longer than a row spans rows; the second row starts mid-document.
- `test_cursor_roundtrip`: serialize/restore `(epoch, position, token_offset, consumed_tokens)`; next batch identical.
- `test_source_weight_visits`: weight 2.0 visits twice; fractional weight is Bernoulli; unknown source raises.
- `test_mask_shares`: `first_regular_id`; special ids never selected; `variant_share > 0`
  is rejected until the variant table is ported.
- `test_mask_labels`: labels only at selected positions; `-100` elsewhere; denominator equals the selected count.
- `test_mlm_loss_denominator`: compare with a hand-computed mean over selected positions.
- `test_clm_shift`: loss over `B*(L-1)` positions; first target excluded exactly once.
- `test_adamw_one_step`: the Optax step vs a scalar reference; bias correction at step 1
  and step 2, and the `ndim >= 2` decay mask.
- `test_weight_decay_mask`: `ndim >= 2` only; no decay on bias/norm.
- `test_clip_boundary`: gradient norm just below/above 1.0; epsilon convention matches.
- `test_wsd_boundaries`: first step, last warmup step, first decay step, final step.

### 3.3 L2 — parity

Use a tiny model: 2 layers, hidden 64, 4 heads, head dim 16, FFN 128, vocab 128, seq 16, batch 2. Both an encoder and a decoder variant.

- `test_encoder_logits_parity`: converted weights, fixed input, fp32 CPU; max abs error ≤ tolerance.
- `test_decoder_logits_parity`: same for Llama and Qwen3 (QK-norm path).
- `test_encoder_loss_parity`: dense vs sparse oracle head; same selected positions.
- `test_decoder_loss_parity`: shifted CE.
- `test_gradient_cosine`: one backward; cosine ≥ tolerance.
- `test_optimizer_step_parity`: one AdamW step from identical state; parameter delta ≤ tolerance.
- `test_gqa_parity`: H/G = 2 and 4; compare `jax.nn.dot_product_attention` on
  `[B, T, G, D]` K/V against a reference that explicitly tiles K/V.
- `test_mask_convention`: reference attention vs `jax.nn.dot_product_attention` with the same document mask.

Real-preset parity is **shape-only** locally unless memory permits: use `jax.eval_shape` and compare against the oracle's meta-device shapes; do not allocate 1B parameters on the audit machine. Full-weight parity for real presets runs on a machine with enough RAM or on TPU.

### 3.4 L3 — integration

- Synthetic tokenized directory with 3 shards, 10 documents, two sources.
- Build the JAX data stream; compare the first 5 batches with the oracle packer (same NumPy code or a fixture).
- Run 3 optimizer steps with a tiny model; loss finite; parameters change.
- Publish a checkpoint to `LocalObjectStore`; resolve it; restore; next batch and loss match the uninterrupted run.
- Kill the process after commit; resume from the committed step; repeat.
- Corrupt a payload byte; publish verification fails.

### 3.5 L4 — export

- Convert a tiny oracle checkpoint to JAX; convert back to safetensors.
- Load with `AutoModelForMaskedLM`/`AutoModelForCausalLM` (`trust_remote_code=False`).
- Compare logits JAX vs HF on a fixed fixture.
- Check the safetensors key set and tied-weight handling.
- Check tokenizer round-trip on a fixed Vietnamese sample.
- Check that a missing file or wrong config fails closed.

## 4. Fixtures

```
tests/jax/fixtures/
  tiny_encoder.json
  tiny_decoder_llama.json
  tiny_decoder_qwen3.json
  tiny_tokenizer.json
  tiny_shards/...
  oracle_encoder.safetensors
  oracle_decoder_llama.safetensors
  oracle_decoder_qwen3.safetensors
  batches.npz
  masks.npz
  checkpoint_manifest.json
```

Fixtures are small enough to commit (a few hundred KB total). Generate them with a script, review the diff, and pin the generator commit. Do not generate fixtures at test time with random seeds that can drift; use fixed seeds and checked-in arrays.

## 5. CI

Add jobs, not a second CI system:

| Job | Command | Network |
|---|---|---|
| lint | `ruff check . && ruff format --check .` | no |
| types | `mypy --strict src/hamic_jax` | no |
| core tests | `pytest tests/ -m "not tpu"` | no |
| jax tests | `pytest tests/jax -m "not tpu"` | no |
| parity | `pytest tests/jax -m "parity"` | no (fixtures checked in) |
| docs | link check + decision-table consistency | no |

`pyproject.toml` changes required at implementation time:

- `[tool.mypy] packages = ["hamic", "hamic_jax"]` (or a second mypy invocation);
- `[tool.pytest.ini_options] markers = ["tpu", "parity"]`;
- ruff extends to `src/hamic_jax` and `tests/jax`;
- optional extras `tpu` and `parity`.

TPU tests are never part of GitHub-hosted CI. They run on a TPU VM with `HAMIC_TPU=1`, and their reports are attached to the run, not to a PR.

## 6. Evidence

Every audit produces a JSON report:

```json
{
  "suite": "P2",
  "commit": "…",
  "timestamp_utc": "…",
  "environment": {"python": "…", "jax": "…", "torch": "…", "transformers": "…"},
  "cases": [
    {"name": "encoder_logits_parity", "status": "pass", "max_abs": 3.1e-6, "tolerance": 1e-5}
  ],
  "artifacts": ["…"],
  "notes": "…"
}
```

Rules:

- report actual errors, not only pass/fail;
- include the exact commit and environment;
- never overwrite a previous report; name it by commit + suite;
- a report with skipped cases is not a pass;
- a tolerance change is a code change with a rationale, not a test edit.

## 7. Gates

### P1 — static

All L0 tests pass. Parameter counts and special ids match. The manifest parser rejects malformed input.

### P2 — CPU parity

All L1/L2 tests pass on the tiny models, and the shape-only checks pass for every preset. This is the minimum bar for touching a TPU.

### P3 — TPU smoke

On the target TPU:

- 20–50 steps with real shapes;
- finite loss and grad norm;
- checkpoint publish, resolve, restore;
- resume produces the same next batch and a loss within tolerance;
- no unexpected recompilation after the first step.

### P4 — resume drill

Kill the process at least twice, once before commit and once after; resume; compare the next 10 steps with an uninterrupted control run. Differences must be explained by the RNG policy, not by data loss.

### P5 — export

The exported model loads through a stock HF class and matches the JAX logits on a fixed fixture. The tokenizer hash is unchanged.

### P6 — production

A real run with a token budget, evaluation, BPB, artifacts, and a written report. Only this gate supports scientific claims.

## 8. What the audit cannot prove

- That the model learns as well on TPU as on GPU. That requires a real run and evaluation.
- That the TPU kernel is fast, or that the memory estimate holds. That requires profiling.
- That the real corpus packer/mask statistics match. That requires real shards.
- That the exported model is useful. That requires downstream evaluation.

State these limits explicitly in the report; do not let a green P2 be read as a green P6.