# 08 — TPU operator notebooks

**Status:** implemented. The four TPU notebooks (`notebooks/tpu_*.ipynb`) are plain
Jupyter notebooks that call `hamic_jax` rather than reimplementing training math.

The TPU port is a **separate implementation** of the same logic, not a switch inside the GPU
path. It mirrors the GPU operator flow cell-for-cell, but its notebooks, backend calls, and
checkpoint format are its own. Nothing under `src/hamic/` or the four GPU notebooks is edited.

GPU notebooks (PyTorch, `HF_TOKEN`, unchanged):

```
notebooks/molab_drill.py
notebooks/molab_train.py
notebooks/molab_eval.py
notebooks/molab_downstream.py
```

TPU notebooks (JAX, `HF_KEY`, plain Jupyter, all new):

```
notebooks/tpu_drill.ipynb
notebooks/tpu_train.ipynb
notebooks/tpu_eval.ipynb
notebooks/tpu_downstream.ipynb
```

The two sets share **behavior**, not source. A change to the GPU notebook is not automatically
a TPU change; parity is enforced by the acceptance checklist in §6, not by importing one
notebook from the other.

## 1. Mapping

| GPU notebook | TPU notebook | Same behavior | TPU implementation |
|---|---|---|---|
| `molab_drill.py` | `notebooks/tpu_drill.ipynb` | setup → auth → device probe → smoke train → stop at step 6 → new session → `resume` → compare losses | pinned source checkout; `jax.devices()`; `hamic_jax.session.run_session`; structural safetensors checkpoint; `HF_KEY` |
| `molab_train.py` | `notebooks/tpu_train.ipynb` | config selector → shard check → bench/train → soft deadline → checkpoint to Hub → resume with the same config | `hamic_jax.session.run_session`; `runs/<run_id>/` manifest, metrics, checkpoints and status |
| `molab_eval.py` | `notebooks/tpu_eval.ipynb` | run selector → validation rows → token/span/PMI views → loss table | downloads a committed JAX checkpoint; `evaluate_views`; read-only |
| `molab_downstream.py` | `notebooks/tpu_downstream.ipynb` | task selector → protocol → fine-tune → report | exports, re-imports, then calls the JAX `run_task`; no `hamic` import |

`molab_downstream.py` is duplicated too. Reusing the GPU notebook would make the TPU path depend
on the PyTorch environment and violate the separation rule.

## 2. Cell contract

Every TPU notebook keeps the GPU operator order as ordered Jupyter cells:

1. title and warnings (markdown);
2. editable constants: repository names, run id, phase/mode, and budgets;
3. pinned source checkout and runtime probe;
4. credential cell: `HF_KEY` from the environment or `getpass`, never a saved value;
5. stores built from the credential;
6. configuration discovery or run selection;
7. data/manifest check;
8. run cell with a `tqdm` progress bar;
9. metrics, tables, and charts.

The notebook is a **thin operator surface**. Training, evaluation, checkpointing, and resume
live in `hamic_jax`, not in notebook cells. A cell may format output, but it must not contain a
second implementation of the loss, mask policy, optimizer, or data order. Jupyter has no
reactive widget graph: cells run top to bottom, choices are plain constants the operator edits
before running the next cell, and a stale kernel is rejected by the bootstrap import check.

## 3. Shared artifacts

Both backends use the same Hub layout and naming:

```
runs/<run_id>/
  checkpoints/step-000000123/
    manifest.json          # step, tokens, tokenizer hash, config identity
    COMMITTED.json         # written last; the checkpoint is complete
    params.safetensors
    optimizer.safetensors  # when the optimizer state is included
  metrics.jsonl
  manifest.json            # backend, preset, raw config, tokenized_prefix
  COMPLETE.json            # terminal success (or STATUS.json when stopped)
  STATUS.json              # stopped / paused with the last step
```

The payload differs: PyTorch `state_dict` files versus flat safetensors arrays. `manifest.json`
records `backend: "torch" | "jax"`, `architecture`, `preset`, `tokenizer_sha256`,
`tokenized_prefix`, `param_count`, and the token budget. Eval and downstream locate the corpus
from `tokenized_prefix`, never from the writer's local path.

A run started on one backend MUST NOT resume on the other. The resolver fails closed when
`backend` does not match the runtime; cross-backend continuation is a new run.

## 4. Credentials

| Path | Environment variable | Notebook source |
|---|---|---|
| GPU/PyTorch | `HF_TOKEN` | password input, used when non-empty |
| TPU/JAX | `HF_KEY` | `getpass` prompt, falling back to `HF_KEY` |

Each notebook reads only its own variable. `HF_KEY` is never read by `src/hamic` or the GPU
notebooks; `HF_TOKEN` is never read by `hamic_jax` or the TPU notebooks.

Tokens are never written to a cell, a config, a manifest, a metric, or a checkpoint.

## 5. Isolation rule

The TPU port may add:

```
src/hamic_jax/**
notebooks/tpu_*.ipynb
tests/jax/**
docs/architecture/tpu-jax/**
```

It may also add an optional `tpu` extra and package entry for `src/hamic_jax` in
`pyproject.toml`. That is packaging metadata, not a change to GPU behavior.

It MUST NOT edit:

```
src/hamic/**
notebooks/molab_drill.py
notebooks/molab_train.py
notebooks/molab_eval.py
notebooks/molab_downstream.py
tests/test_*.py
configs/**
```

`configs/`, `tokenizer.json`, and tokenized manifests are read-only inputs. If `hamic_jax`
needs a pure dataclass, it defines it in `hamic_jax`; it does not extract one into `hamic.spec`.

**Packaging caveat:** the wheel target in `pyproject.toml` lists only `src/hamic`, so
`pip install git+…` does not ship `src/hamic_jax`. The four notebooks therefore resolve
`HAMIC_REF` to a commit, extract that commit's archive under the work root, prepend
`<checkout>/src` to `sys.path`, install only the runtime dependencies, and assert that
`hamic_jax.__file__` resolves inside the checkout. If `pyproject.toml` later adds
`src/hamic_jax` to the wheel packages, the notebooks can switch to the GPU pattern
(`pip install git+…` plus the `direct_url.json` commit check) without changing any cell after
the install.

## 6. Acceptance checklist

A TPU notebook is accepted only when:

1. it has the same sections and choices as its GPU counterpart;
2. it runs on a TPU VM without editing source cells;
3. the drill stops at step 6, resumes in a new kernel, and reproduces the reference losses
   within the contract tolerance;
4. train publishes and resolves a checkpoint through the same Hub layout;
5. eval produces the same metric names and masking views as the GPU notebook;
6. downstream produces the same task list, protocol, and report fields;
7. no token appears in output, logs, or the notebook file;
8. the notebook imports no `hamic` module and contains no training math of its own.

## 7. Roadmap placement

- **P1:** define the `hamic_jax` calls the notebook cells use (`run_session`, `run_smoke`,
  `evaluate_views`, `run_task`). Done.
- **P5:** `notebooks/tpu_drill.ipynb` and `notebooks/tpu_train.ipynb` are implemented; the
  drill still has to pass on a real TPU before the first production run.
- **P6:** `notebooks/tpu_eval.ipynb` is implemented and reads only committed runs.
- **P7:** `notebooks/tpu_downstream.ipynb` is implemented; run it on an exported model.

Until the drill passes on a real TPU, a TPU run is not evidence.