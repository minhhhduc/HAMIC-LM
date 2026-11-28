"""Notebook workflows for the TPU lab.

The TPU notebooks under ``notebooks`` are thin control surfaces. Configuration
discovery, benchmark isolation, throughput math, completed-run evaluation, and
downstream fine-tuning live here so the same operations are testable without a notebook.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hamic_jax.checkpoint import (
    CheckpointResolver,
    LocalObjectStore,
    ObjectStore,
    checkpoint_name,
    load_checkpoint,
)
from hamic_jax.data import TokenizedCorpus, fetch_tokenized, tokenized_prefix
from hamic_jax.downstream import (
    TASKS,
    FinetuneConfig,
    build_tokenizer,
    load_task,
    run_task,
)
from hamic_jax.export import export_pretrained, import_weights
from hamic_jax.presets import load_preset, load_training_config
from hamic_jax.runs import RunRecord, list_runs
from hamic_jax.session import SessionLimits, SessionReport, run_session
from hamic_jax.spec import (
    ModelPreset,
    TrainingConfig,
    model_preset_from_mapping,
    training_config_from_mapping,
)
from hamic_jax.train import evaluate_views

__all__ = [
    "BenchResult",
    "EvaluationRow",
    "Throughput",
    "TrainingCandidate",
    "complete_runs",
    "default_candidate",
    "discover_training_configs",
    "evaluate_completed_run",
    "fetch_corpus",
    "measure_throughput",
    "run_bench",
    "run_downstream",
    "run_training_session",
]

_VIEWS = ("token", "span", "pmi")


@dataclass(frozen=True)
class TrainingCandidate:
    """One YAML training config that loaded successfully."""

    name: str
    config: TrainingConfig
    preset: ModelPreset


@dataclass(frozen=True)
class BenchResult:
    """A benchmark run and the temporary config that produced it."""

    config: TrainingConfig
    report: SessionReport


@dataclass(frozen=True)
class Throughput:
    """Observed throughput and the remaining wall-clock estimate."""

    tokens_per_second: float
    tflops: float
    hours_left: float


@dataclass(frozen=True)
class EvaluationRow:
    """One validation view for one completed run."""

    run_id: str
    step: int
    objective: str
    view: str
    eval_loss: float


def discover_training_configs(
    config_root: str | Path,
) -> tuple[list[TrainingCandidate], list[dict[str, str]]]:
    """Load every ``*.yaml`` under ``config_root`` and report the rejected ones."""
    root = Path(config_root)
    candidates: list[TrainingCandidate] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            config, preset = load_training_config(path)
        except Exception as exc:  # noqa: BLE001 - one bad YAML must not hide the rest
            rejected.append({"config": path.name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        name = path.relative_to(root.parent.parent).as_posix()
        candidates.append(TrainingCandidate(name=name, config=config, preset=preset))
    return candidates, rejected


def default_candidate(
    candidates: Sequence[TrainingCandidate], needle: str = "e-v02-base"
) -> TrainingCandidate:
    """Return the candidate whose name contains ``needle``, else the first one."""
    if not candidates:
        raise ValueError("no training config loaded")
    for candidate in candidates:
        if needle in candidate.name:
            return candidate
    return candidates[0]


def fetch_corpus(
    config: TrainingConfig,
    data_store: ObjectStore,
    work_root: str | Path,
    run_id: str,
    *,
    prefix: str | None = None,
) -> tuple[TrainingConfig, TokenizedCorpus]:
    """Fetch and verify a published tokenized corpus, returning a local config."""
    remote = prefix or tokenized_prefix(config.tokenized)
    destination = Path(work_root) / "corpora" / run_id / Path(config.tokenized).name
    corpus = fetch_tokenized(data_store, remote, destination)
    return replace(config, tokenized=str(destination)), corpus


def complete_runs(store: ObjectStore) -> list[RunRecord]:
    """Return finished runs, newest first."""
    return [record for record in list_runs(store) if record.complete]


def run_bench(
    base: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    work_root: str | Path,
    *,
    steps: int,
    learning_rate: float | None = None,
    warmup_ratio: float | None = None,
    source_commit: str = "",
    progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
    show_progress: bool = False,
) -> BenchResult:
    """Run a short local benchmark that can never resume or touch Hub runs."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overrides: dict[str, float] = {}
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if warmup_ratio is not None:
        overrides["warmup_ratio"] = warmup_ratio
    bench_config = replace(
        base,
        run_id=f"{base.run_id}-bench-{stamp}",
        total_tokens=base.tokens_per_step * steps,
        checkpoint_every_steps=0,
        checkpoint_every_minutes=0.0,
        eval_every_steps=0,
        log_every_steps=max(1, steps // 5),
        **overrides,
    )
    root = Path(work_root)
    store = LocalObjectStore(root / "bench-store" / stamp)
    report = run_session(
        bench_config,
        preset,
        corpus,
        store,
        root / "bench" / stamp,
        SessionLimits(
            steps=steps,
            checkpoint_every_steps=0,
            checkpoint_every_seconds=0.0,
            log_every_steps=max(1, steps // 5),
            checkpoint_at_end=True,
        ),
        source_commit=source_commit,
        raw_config=asdict(bench_config),
        progress=progress,
        log=log,
        show_progress=show_progress,
    )
    return BenchResult(config=bench_config, report=report)


def measure_throughput(
    report: SessionReport, config: TrainingConfig, preset: ModelPreset
) -> Throughput:
    """Compute tokens/s, TFLOP/s, and remaining hours from a session report."""
    start = report.resumed_from_step or 0
    rows = [
        record
        for record in report.metrics
        if int(record.get("step", 0)) > start
        and "tokens" in record
        and "elapsed_seconds" in record
    ]
    rate = 0.0
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        elapsed = float(last["elapsed_seconds"]) - float(first["elapsed_seconds"])
        tokens = float(last["tokens"]) - float(first["tokens"])
        if elapsed > 0:
            rate = tokens / elapsed
    remaining = max(config.total_steps - report.step, 0)
    hours_left = remaining * config.tokens_per_step / max(rate, 1.0) / 3600.0
    tflops = 6.0 * preset.num_parameters * rate / 1e12
    return Throughput(
        tokens_per_second=rate,
        tflops=tflops,
        hours_left=hours_left,
    )


def run_training_session(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    store: ObjectStore,
    workdir: str | Path,
    *,
    seconds: float = 10.5 * 3600.0,
    source_commit: str = "",
    raw_config: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
    handle_preemption: bool = False,
    fetcher: Any = None,
    show_progress: bool = False,
) -> SessionReport:
    """Run or resume a real training session against the configured Hub store."""
    limits = SessionLimits(
        seconds=seconds,
        checkpoint_every_steps=config.checkpoint_every_steps,
        checkpoint_every_seconds=config.checkpoint_every_minutes * 60.0,
        log_every_steps=config.log_every_steps,
        eval_every_steps=config.eval_every_steps,
        eval_sequences=config.eval_sequences,
        checkpoint_at_end=True,
    )
    return run_session(
        config,
        preset,
        corpus,
        store,
        workdir,
        limits,
        source_commit=source_commit,
        raw_config=dict(raw_config) if raw_config is not None else asdict(config),
        progress=progress,
        log=log,
        handle_preemption=handle_preemption,
        fetcher=fetcher,
        show_progress=show_progress,
    )


def _resolve_preset(
    source_dir: str | Path, raw: Mapping[str, Any], manifest: Mapping[str, Any]
) -> ModelPreset:
    """Load the source preset, falling back to the shape recorded in the manifest."""
    model_preset = raw.get("model_preset")
    if isinstance(model_preset, str):
        try:
            return load_preset(Path(source_dir) / model_preset)
        except (OSError, ValueError):
            pass
    return model_preset_from_mapping(manifest["preset"])


def _download_checkpoint(
    record: RunRecord, run_store: ObjectStore, work_root: str | Path
) -> tuple[Path, dict[str, np.ndarray[Any, Any]], int]:
    """Download the checkpoint that matches the run's recorded step."""
    resolver = CheckpointResolver(run_store, record.prefix)
    committed = resolver.committed()
    if not committed:
        raise ValueError(f"no committed checkpoint under {record.prefix}")
    chosen = next((item for item in committed if item.step == record.step), committed[-1])
    step = int(chosen.step)
    destination = Path(work_root) / "checkpoints" / record.run_id / checkpoint_name(step)
    if destination.exists():
        shutil.rmtree(destination)
    resolver.download(chosen, destination)
    params, _, _, _, _ = load_checkpoint(destination)
    return destination, params, step


def evaluate_completed_run(
    record: RunRecord,
    run_store: ObjectStore,
    data_store: ObjectStore,
    work_root: str | Path,
    *,
    source_dir: str | Path,
    sequences: int,
    views: Sequence[str],
    mask_rate: float,
    log: Callable[[str], None] = print,
    show_progress: bool = False,
) -> list[EvaluationRow]:
    """Evaluate one finished run under the requested masking views."""
    if not record.complete:
        raise ValueError(f"{record.run_id} is not complete")
    if sequences <= 0:
        raise ValueError("sequences must be positive")
    selected = tuple(view for view in views if view in _VIEWS)
    if len(selected) != len(tuple(views)):
        raise ValueError(f"views must be drawn from {_VIEWS}")
    raw = record.manifest["config"]
    config = training_config_from_mapping(raw)
    preset = _resolve_preset(source_dir, raw, record.manifest)
    prefix = record.manifest.get("tokenized_prefix") or tokenized_prefix(Path(config.tokenized))
    config, corpus = fetch_corpus(config, data_store, work_root, record.run_id, prefix=prefix)
    if config.objective == "clm":
        selected = ("token",)
    if not selected:
        raise ValueError("select at least one masking view")
    _, params, step = _download_checkpoint(record, run_store, work_root)
    log(f"evaluating {record.run_id} at step {step} with {selected}")
    metrics = evaluate_views(
        config,
        preset,
        corpus,
        params,
        views=selected,
        sequences=sequences,
        mask_rate=mask_rate,
        show_progress=show_progress,
    )
    return [
        EvaluationRow(
            run_id=record.run_id,
            step=step,
            objective=config.objective,
            view=view,
            eval_loss=float(loss),
        )
        for view, loss in metrics.items()
    ]


def run_downstream(
    record: RunRecord,
    run_store: ObjectStore,
    data_store: ObjectStore,
    work_root: str | Path,
    *,
    source_dir: str | Path,
    token: str,
    protocol: str,
    tasks: Sequence[str],
    progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Export one encoder, re-import it, and run the requested Vietnamese tasks."""
    if not record.complete:
        raise ValueError(f"{record.run_id} is not complete")
    if protocol not in ("quick", "full"):
        raise ValueError("protocol must be 'quick' or 'full'")
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    if not tasks:
        raise ValueError("select at least one task")
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    raw = record.manifest["config"]
    config = training_config_from_mapping(raw)
    preset = _resolve_preset(source_dir, raw, record.manifest)
    prefix = record.manifest.get("tokenized_prefix") or tokenized_prefix(Path(config.tokenized))
    _, corpus = fetch_corpus(config, data_store, work_root, record.run_id, prefix=prefix)
    _, params, step = _download_checkpoint(record, run_store, work_root)
    export_dir = Path(work_root) / "exports" / f"{record.run_id}-step-{step:09d}"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_pretrained(
        preset,
        params,
        export_dir,
        tokenizer_dir=corpus.root,
        lineage={
            "run_id": record.run_id,
            "step": step,
            "backend": "jax",
            "source_commit": record.manifest.get("source_commit", ""),
        },
    )
    encoder = import_weights(export_dir, preset)
    if protocol == "quick":
        finetune = FinetuneConfig(
            learning_rates=(5e-5,),
            seeds=(42,),
            max_epochs=3,
            patience=1,
        )
    else:
        finetune = FinetuneConfig()
    tokenizer = build_tokenizer(export_dir / "tokenizer.json", max_length=finetune.max_length)
    cache_dir = str(Path(work_root) / "hf-cache")
    results: dict[str, Any] = {}
    for index, name in enumerate(tasks, start=1):
        log(f"fine-tuning {name} ({protocol})")
        data = load_task(TASKS[name], cache_dir=cache_dir)
        result = run_task(
            preset,
            encoder,
            data,
            tokenizer,
            config=finetune,
            log=log,
        )
        results[name] = asdict(result)
        if progress is not None:
            progress(index, len(tasks), name)
    report_path = Path(work_root) / "downstream" / (
        f"downstream-{record.run_id}-{protocol}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": record.run_id,
        "step": step,
        "source_commit": record.manifest.get("source_commit", ""),
        "protocol": protocol,
        "config": asdict(finetune),
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {**report, "report_path": str(report_path)}