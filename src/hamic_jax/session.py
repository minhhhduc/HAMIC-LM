"""High-level train, resume and drill entry points for the TPU notebooks.

``Trainer`` is deliberately a single-process training loop. This module adds the parts an
operator needs on a preemptible TPU host: strict resume, periodic checkpoint publishing,
metrics and completion records, and a drill that proves a resumed run reproduces the
uninterrupted trajectory.
"""

from __future__ import annotations

import json
import math
import shutil
import signal
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hamic_jax.checkpoint import (
    CheckpointPublisher,
    CheckpointResolver,
    ObjectStore,
    checkpoint_name,
    load_checkpoint,
    save_checkpoint,
)
from hamic_jax.data import TokenizedCorpus, tokenized_prefix
from hamic_jax.optim import learning_rate_schedule
from hamic_jax.runs import (
    COMPLETE_NAME,
    mark_complete,
    publish_metrics,
    run_prefix,
    write_run_manifest,
    write_status,
)
from hamic_jax.sharding import device_label, peak_memory_gib
from hamic_jax.spec import ModelPreset, TrainingConfig
from hamic_jax.train import Trainer, evaluate_metrics

__all__ = [
    "SessionLimits",
    "SessionReport",
    "SmokeReport",
    "run_session",
    "run_smoke",
]


@dataclass(frozen=True)
class SessionLimits:
    """When to stop a training session and when to checkpoint."""

    steps: int | None = None
    seconds: float | None = None
    checkpoint_every_steps: int = 100
    checkpoint_every_seconds: float = 600.0
    log_every_steps: int = 10
    eval_every_steps: int = 0
    eval_sequences: int | None = None
    checkpoint_at_end: bool = True


@dataclass(frozen=True)
class SessionReport:
    """What one training session did."""

    run_id: str
    step: int
    resumed_from_step: int | None
    losses: tuple[float, ...]
    metrics: tuple[dict[str, Any], ...]
    checkpoints: tuple[str, ...]
    completed: bool
    paused: bool
    seconds: float
    interrupted: bool = False
    already_complete: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """Terminal status of the session."""
        if self.already_complete:
            return "already_complete"
        if self.completed:
            return "completed"
        if self.interrupted:
            return "interrupted"
        return "paused" if self.paused else "stopped"

    @property
    def final_step(self) -> int:
        """Step reached by the session."""
        return self.step

    @property
    def published_steps(self) -> list[int]:
        """Steps whose checkpoints this session committed."""
        return [
            int(Path(name).name.removeprefix("step-"))
            for name in self.checkpoints
            if Path(name).name.startswith("step-")
        ]

    @property
    def evaluations(self) -> dict[int, float]:
        """Evaluation loss keyed by step for this session only."""
        start = self.resumed_from_step or 0
        return {
            int(record["step"]): float(record["eval_loss"])
            for record in self.metrics
            if "eval_loss" in record and int(record.get("step", 0)) > start
        }

    @property
    def eval_bits_per_byte(self) -> dict[int, float]:
        """Bits-per-byte by step for this session; CLM runs with a tokenizer copy."""
        start = self.resumed_from_step or 0
        return {
            int(record["step"]): float(record["eval_bpb"])
            for record in self.metrics
            if "eval_bpb" in record and int(record.get("step", 0)) > start
        }


@dataclass(frozen=True)
class SmokeReport:
    """Comparison of a stopped-and-resumed run with an uninterrupted reference."""

    fresh: tuple[float, ...]
    resumed: tuple[float, ...]
    reference: tuple[float, ...]
    resumed_from_step: int | None
    max_abs_diff: float
    identical: bool


def run_session(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    store: ObjectStore | None,
    workdir: str | Path,
    limits: SessionLimits | None = None,
    *,
    source_commit: str = "",
    raw_config: Mapping[str, Any] | None = None,
    publish: bool = True,
    progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
    handle_preemption: bool = False,
    fetcher: Any = None,
) -> SessionReport:
    """Train until the step/time limit, resuming from the latest committed checkpoint.

    Args:
        config: Resolved training config.
        preset: Model shape matching the config.
        corpus: Verified local tokenized corpus.
        store: Published runs live here; required when ``publish`` is true.
        workdir: Local scratch area for restored and new checkpoints.
        limits: Stop/checkpoint cadence. Defaults to the config's checkpoint interval.
        source_commit: Repository commit that produced the run.
        raw_config: Original YAML mapping stored in ``manifest.json``.
        publish: Publish manifests, checkpoints, metrics and markers.
        progress: Called as ``(step, target, loss)`` after every training step.
        log: Progress sink.
        handle_preemption: Turn SIGTERM into a clean checkpoint and ``paused`` status.
            A preemptible v5e host sends SIGTERM before it reclaims the VM.

    Returns:
        What the session did, including the loss sequence produced now.
    """
    if publish and store is None:
        raise ValueError("publish=True needs an object store")
    if limits is None:
        limits = SessionLimits(
            checkpoint_every_steps=0,
            checkpoint_every_seconds=config.checkpoint_every_minutes * 60.0,
        )
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = run_prefix(config.run_id)
    started = time.monotonic()
    losses: list[float] = []
    metrics: list[dict[str, Any]] = _load_metrics(root, store, prefix, publish)
    checkpoints: list[str] = []
    resumed_from_step: int | None = None
    publisher: CheckpointPublisher | None = None

    if publish:
        manifest = {
            "run_id": config.run_id,
            "backend": "jax",
            "identity": config.identity(),
            "config": dict(raw_config) if raw_config is not None else asdict(config),
            "preset": preset.to_dict(),
            "source_commit": source_commit,
            "param_count": preset.num_parameters,
            "tokenizer_sha256": corpus.tokenizer_sha256,
            "corpus_sha256": corpus.manifest_sha256,
            "tokenized_prefix": tokenized_prefix(config.tokenized),
            "total_steps": config.total_steps,
            "total_tokens": config.total_tokens,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        assert store is not None
        write_run_manifest(store, manifest, f"open run {config.run_id}")
        if store.read_bytes(f"{prefix}/{COMPLETE_NAME}") is not None:
            return SessionReport(
                run_id=config.run_id,
                step=config.total_steps,
                resumed_from_step=None,
                losses=(),
                metrics=(),
                checkpoints=(),
                completed=True,
                paused=False,
                seconds=0.0,
                already_complete=True,
            )
        publisher = CheckpointPublisher(store, prefix)
        if config.objective == "mlm" and config.masking == "pmi":
            local_pmi = corpus.root / "pmi_pairs.npy"
            if not local_pmi.is_file() and store.read_bytes(f"{prefix}/pmi_pairs.npy") is not None:
                store.download(f"{prefix}/pmi_pairs.npy", local_pmi)

    trainer = Trainer(config, preset, corpus, fetcher=fetcher)
    if publish and store is not None and config.objective == "mlm" and config.masking == "pmi":
        local_pmi = corpus.root / "pmi_pairs.npy"
        if local_pmi.is_file() and store.read_bytes(f"{prefix}/pmi_pairs.npy") is None:
            store.put_files({f"{prefix}/pmi_pairs.npy": local_pmi}, f"pmi pairs {config.run_id}")
    warnings: list[str] = []
    if publish:
        assert store is not None
        resolver = CheckpointResolver(store, prefix)
        restore_dir = root / f"restore-{uuid.uuid4().hex}"
        try:
            remote = resolver.restore_latest(restore_dir)
            if remote is not None:
                params, optimizer_state, step, cursor, metadata = load_checkpoint(
                    restore_dir, trainer.optimizer
                )
                if metadata.get("backend") != "jax":
                    raise RuntimeError(
                        f"checkpoint backend is {metadata.get('backend')!r}, not 'jax'"
                    )
                if metadata.get("identity") != config.identity():
                    raise RuntimeError("checkpoint identity does not match the training config")
                if metadata.get("corpus_sha256") != corpus.manifest_sha256:
                    raise RuntimeError("checkpoint corpus does not match the tokenized corpus")
                saved_commit = str(metadata.get("source_commit") or "")
                running_commit = str(source_commit or "")
                if saved_commit and running_commit and saved_commit != running_commit:
                    raise RuntimeError(
                        "checkpoint code commit "
                        f"{saved_commit[:12]} does not match running commit "
                        f"{running_commit[:12]}"
                    )
                if not saved_commit or not running_commit:
                    warnings.append(
                        "code commit unknown on one side; continuing without a code check"
                    )
                trainer.restore(params, optimizer_state)
                del params, optimizer_state
                trainer.step = step
                trainer.batches.restore(cursor)
                resumed_from_step = step
                log(f"resumed {config.run_id} at step {step}")
        finally:
            shutil.rmtree(restore_dir, ignore_errors=True)
    metrics = [
        record
        for record in metrics
        if int(record.get("step", 0)) <= (resumed_from_step or 0)
    ]

    total_steps = config.total_steps
    target = total_steps if limits.steps is None else min(total_steps, limits.steps)
    deadline = None if limits.seconds is None else started + limits.seconds
    last_checkpoint_step = resumed_from_step
    last_checkpoint_time = time.monotonic()
    last_tokens = float(trainer.batches.consumed_tokens)
    last_elapsed = 0.0
    paused = False
    schedule = learning_rate_schedule(config)
    device = device_label()
    pending: list[tuple[int, jax.Array, jax.Array, int, float]] = []
    stop = {"requested": False}
    previous_handler: Any = None

    def flush() -> list[dict[str, Any]]:
        """Read one log window's device scalars and append its metric records."""
        nonlocal last_tokens, last_elapsed
        if not pending:
            return []
        values = np.asarray(
            jnp.stack(
                [
                    jnp.stack([item[1] for item in pending]),
                    jnp.stack([item[2] for item in pending]),
                ]
            )
        )
        elapsed = time.monotonic() - started
        peak = peak_memory_gib()
        records: list[dict[str, Any]] = []
        for index, (step, _, _, tokens, step_elapsed) in enumerate(pending):
            last_in_window = index == len(pending) - 1
            if last_in_window:
                step_elapsed = elapsed
            record: dict[str, Any] = {
                "step": step,
                "loss": float(values[0, index]),
                "lr": float(schedule(max(0, step - 1))),
                "grad_norm": float(values[1, index]),
                "tokens": tokens,
                "elapsed_seconds": round(step_elapsed, 3),
                "device": device,
                "commit": source_commit,
            }
            if peak is not None:
                record["peak_hbm_gib"] = peak
            if last_in_window and step_elapsed > last_elapsed:
                record["tokens_per_second"] = (tokens - last_tokens) / (
                    step_elapsed - last_elapsed
                )
            records.append(record)
        losses.extend(record["loss"] for record in records)
        metrics.extend(records)
        last_tokens = float(pending[-1][3])
        last_elapsed = elapsed
        pending.clear()
        return records

    def _request_stop(signum: int, frame: Any) -> None:
        stop["requested"] = True

    if handle_preemption:
        try:
            previous_handler = signal.signal(signal.SIGTERM, _request_stop)
        except (ValueError, OSError):
            warnings.append("SIGTERM handler unavailable; preemption will not checkpoint")

    try:
        with trainer.prefetch(depth=2):
            while trainer.step < target:
                if stop["requested"]:
                    paused = True
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    paused = True
                    break
                loss, norm = trainer.train_step_metrics()
                now = time.monotonic()
                pending.append(
                    (
                        trainer.step,
                        loss,
                        norm,
                        trainer.batches.consumed_tokens,
                        now - started,
                    )
                )
                due_eval = bool(
                    limits.eval_every_steps
                    and trainer.step % limits.eval_every_steps == 0
                )
                due_log = bool(
                    limits.log_every_steps
                    and trainer.step % limits.log_every_steps == 0
                )
                records = (
                    flush()
                    if due_eval or due_log or progress is not None
                    else []
                )
                current = records[-1] if records else None
                if due_eval:
                    assert current is not None
                    eval_loss, eval_bpb = evaluate_metrics(
                        config,
                        preset,
                        corpus,
                        trainer.params,
                        sequences=limits.eval_sequences,
                    )
                    current["eval_loss"] = eval_loss
                    if eval_bpb is not None:
                        current["eval_bpb"] = eval_bpb
                if progress is not None:
                    assert current is not None
                    progress(trainer.step, target, float(current["loss"]))
                if due_log:
                    assert current is not None
                    log(
                        f"step {trainer.step}/{target} loss {current['loss']:.5f} "
                        f"lr {current['lr']:.3g} |g| {current['grad_norm']:.3f} "
                        f"tokens {current['tokens']}"
                    )
                now = time.monotonic()
                due_steps = bool(
                    limits.checkpoint_every_steps
                    and trainer.step % limits.checkpoint_every_steps == 0
                )
                due_time = bool(
                    limits.checkpoint_every_seconds
                    and now - last_checkpoint_time >= limits.checkpoint_every_seconds
                )
                due_window = bool(getattr(trainer.batches, "has_completed_window", False))
                if publish and (due_steps or due_time or due_window):
                    flush()
                    checkpoint = _persist(
                        trainer,
                        config,
                        preset,
                        corpus,
                        store,
                        publisher,
                        root,
                        source_commit,
                        log,
                    )
                    if checkpoint is not None:
                        checkpoints.append(checkpoint)
                        last_checkpoint_step = trainer.step
                        last_checkpoint_time = now
                        _publish_metrics(root, store, config.run_id, metrics, publish)
                        evicted = trainer.evict_completed_windows()
                        if evicted:
                            log(f"evicted {len(evicted)} completed shard(s)")
                elif not publish and due_window:
                    trainer.evict_completed_windows()
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
    flush()

    completed = trainer.step >= total_steps
    interrupted = bool(
        limits.steps is not None
        and trainer.step >= limits.steps
        and trainer.step < total_steps
    )
    if (
        limits.checkpoint_at_end
        and trainer.step > 0
        and trainer.step != last_checkpoint_step
    ):
        checkpoint = _persist(
            trainer,
            config,
            preset,
            corpus,
            store,
            publisher,
            root,
            source_commit,
            log,
        )
        if checkpoint is not None:
            checkpoints.append(checkpoint)
            last_checkpoint_step = trainer.step
            trainer.evict_completed_windows()
    if publish:
        _publish_metrics(root, store, config.run_id, metrics, publish)
        assert store is not None
        payload = {
            "run_id": config.run_id,
            "backend": "jax",
            "step": trainer.step,
            "tokens": trainer.batches.consumed_tokens,
            "total_steps": total_steps,
            "source_commit": source_commit,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if completed:
            mark_complete(
                store,
                config.run_id,
                {**payload, "status": "completed"},
                f"complete {config.run_id}",
            )
        else:
            status = "paused" if paused else ("interrupted" if interrupted else "stopped")
            write_status(
                store,
                config.run_id,
                {**payload, "status": status},
                f"status {config.run_id}",
            )

    return SessionReport(
        run_id=config.run_id,
        step=trainer.step,
        resumed_from_step=resumed_from_step,
        losses=tuple(losses),
        metrics=tuple(metrics),
        checkpoints=tuple(checkpoints),
        completed=completed,
        paused=paused,
        seconds=time.monotonic() - started,
        interrupted=interrupted,
        warnings=tuple(warnings),
    )


def run_smoke(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    store: ObjectStore,
    workdir: str | Path,
    *,
    steps: int = 12,
    stop_after: int = 6,
    checkpoint_every: int = 4,
    source_commit: str = "",
    log: Callable[[str], None] = print,
) -> SmokeReport:
    """Stop a tiny run, resume it from the Hub, and compare with a fresh reference.

    The drill is intentionally local-store friendly: pass ``LocalObjectStore`` for a
    notebook that has no Hub token, or ``HuggingFaceObjectStore`` to exercise publishing.
    The run id gets a unique suffix so reruns always start from zero.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if not 0 < stop_after <= steps:
        raise ValueError("stop_after must be between 1 and steps")
    smoke_config = replace(
        config,
        run_id=f"{config.run_id}-drill-{stamp}",
        total_tokens=config.tokens_per_step * steps,
    )
    raw = asdict(smoke_config)
    root = Path(workdir)
    fresh = run_session(
        smoke_config,
        preset,
        corpus,
        store,
        root / "fresh",
        SessionLimits(
            steps=stop_after,
            checkpoint_every_steps=checkpoint_every,
            checkpoint_every_seconds=0.0,
            log_every_steps=1,
            checkpoint_at_end=False,
        ),
        source_commit=source_commit,
        raw_config=raw,
        log=log,
    )
    resumed = run_session(
        smoke_config,
        preset,
        corpus,
        store,
        root / "resumed",
        SessionLimits(
            steps=steps,
            checkpoint_every_steps=checkpoint_every,
            checkpoint_every_seconds=0.0,
            log_every_steps=1,
            checkpoint_at_end=True,
        ),
        source_commit=source_commit,
        raw_config=raw,
        log=log,
    )
    reference: list[float] = []
    trainer = Trainer(smoke_config, preset, corpus)
    for _ in range(steps):
        reference.append(float(trainer.train_step()))
    split = resumed.resumed_from_step or 0
    observed = list(fresh.losses[:split]) + list(resumed.losses)
    expected = (stop_after // checkpoint_every) * checkpoint_every
    diff = max(
        (abs(left - right) for left, right in zip(observed, reference)),
        default=float("inf"),
    )
    identical = (
        resumed.resumed_from_step == expected
        and len(observed) == len(reference)
        and math.isclose(diff, 0.0, rel_tol=0.0, abs_tol=1e-6)
    )
    return SmokeReport(
        fresh=fresh.losses,
        resumed=resumed.losses,
        reference=tuple(reference),
        resumed_from_step=resumed.resumed_from_step,
        max_abs_diff=diff,
        identical=identical,
    )


def _persist(
    trainer: Trainer,
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    store: ObjectStore | None,
    publisher: CheckpointPublisher | None,
    root: Path,
    source_commit: str,
    log: Callable[[str], None],
) -> str | None:
    name = checkpoint_name(trainer.step)
    local = root / "checkpoints" / name
    metadata = {
        "step": trainer.step,
        "backend": "jax",
        "run_id": config.run_id,
        "identity": config.identity(),
        "preset": preset.to_dict(),
        "source_commit": source_commit,
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "corpus_sha256": corpus.manifest_sha256,
        "param_count": preset.num_parameters,
    }
    if not local.exists():
        save_checkpoint(
            local,
            trainer.params,
            trainer.optimizer_state,
            trainer.step,
            trainer.batches.cursor(),
            metadata,
        )
    if publisher is not None:
        remote = f"{publisher.run_prefix}/checkpoints/{name}/COMMITTED.json"
        assert store is not None
        if store.read_bytes(remote) is None:
            publisher.publish(local, metadata)
            log(f"published checkpoint {name}")
        for older in (root / "checkpoints").glob("step-*"):
            if older != local and older.is_dir():
                shutil.rmtree(older, ignore_errors=True)
    return str(local)


def _load_metrics(
    root: Path,
    store: ObjectStore | None,
    prefix: str,
    publish: bool,
) -> list[dict[str, Any]]:
    if not publish:
        return []
    local = root / "metrics.jsonl"
    raw = local.read_text(encoding="utf-8") if local.is_file() else None
    if raw is None and store is not None:
        remote = store.read_bytes(f"{prefix}/metrics.jsonl")
        raw = remote.decode() if remote is not None else ""
    records: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _publish_metrics(
    root: Path,
    store: ObjectStore | None,
    run_id: str,
    metrics: list[dict[str, Any]],
    publish: bool,
) -> None:
    if not publish or store is None:
        return
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in metrics)
    (root / "metrics.jsonl").write_text(text, encoding="utf-8")
    publish_metrics(store, run_id, text, f"metrics {run_id}")