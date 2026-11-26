"""High-level train, resume and drill entry points for the TPU notebooks.

``Trainer`` is deliberately a single-process training loop. This module adds the parts an
operator needs on a preemptible TPU host: strict resume, periodic checkpoint publishing,
metrics and completion records, and a drill that proves a resumed run reproduces the
uninterrupted trajectory.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from hamic_jax.checkpoint import (
    CheckpointPublisher,
    CheckpointResolver,
    ObjectStore,
    checkpoint_name,
    load_checkpoint,
    save_checkpoint,
)
from hamic_jax.data import TokenizedCorpus, tokenized_prefix
from hamic_jax.runs import (
    mark_complete,
    publish_metrics,
    run_prefix,
    write_run_manifest,
    write_status,
)
from hamic_jax.spec import ModelPreset, TrainingConfig
from hamic_jax.train import Trainer, evaluate

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

    @property
    def status(self) -> str:
        """Terminal status of the session."""
        if self.completed:
            return "completed"
        return "paused" if self.paused else "stopped"

    @property
    def final_step(self) -> int:
        """Step reached by the session."""
        return self.step

    @property
    def published_steps(self) -> int:
        """Number of checkpoints published by the session."""
        return len(self.checkpoints)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal warnings; the JAX session currently emits none."""
        return ()

    @property
    def evaluations(self) -> dict[int, float]:
        """Evaluation loss keyed by step."""
        return {
            int(record["step"]): float(record["eval_loss"])
            for record in self.metrics
            if "eval_loss" in record
        }

    @property
    def eval_bits_per_byte(self) -> dict[int, float]:
        """Bits-per-byte by step; not tracked by the JAX evaluator."""
        return {}


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
            "tokenized_prefix": tokenized_prefix(config.tokenized),
            "total_steps": config.total_steps,
            "total_tokens": config.total_tokens,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        assert store is not None
        write_run_manifest(store, manifest, f"open run {config.run_id}")
        publisher = CheckpointPublisher(store, prefix)

    trainer = Trainer(config, preset, corpus)
    if publish:
        assert store is not None
        resolver = CheckpointResolver(store, prefix)
        restore_dir = root / f"restore-{uuid.uuid4().hex}"
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
            trainer.params = jax.tree.map(
                lambda leaf: jnp.asarray(leaf, dtype=jnp.float32), params
            )
            trainer.optimizer_state = optimizer_state
            trainer.step = step
            trainer.batches.restore(cursor)
            resumed_from_step = step
            log(f"resumed {config.run_id} at step {step}")
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

    while trainer.step < target:
        if deadline is not None and time.monotonic() >= deadline:
            paused = True
            break
        losses.append(float(trainer.train_step()))
        now = time.monotonic()
        elapsed = now - started
        record = {
            "step": trainer.step,
            "loss": losses[-1],
            "tokens": trainer.batches.consumed_tokens,
            "elapsed_seconds": round(elapsed, 3),
        }
        if elapsed > last_elapsed:
            record["tokens_per_second"] = (record["tokens"] - last_tokens) / (
                elapsed - last_elapsed
            )
        last_tokens = float(record["tokens"])
        last_elapsed = elapsed
        if limits.eval_every_steps and trainer.step % limits.eval_every_steps == 0:
            record["eval_loss"] = evaluate(
                config,
                preset,
                corpus,
                trainer.params,
                sequences=limits.eval_sequences,
            )
        metrics.append(record)
        if progress is not None:
            progress(trainer.step, target, record["loss"])
        if limits.log_every_steps and trainer.step % limits.log_every_steps == 0:
            log(
                f"step {trainer.step}/{target} loss {record['loss']:.5f} "
                f"tokens {record['tokens']}"
            )
        due_steps = bool(
            limits.checkpoint_every_steps
            and trainer.step % limits.checkpoint_every_steps == 0
        )
        due_time = bool(
            limits.checkpoint_every_seconds
            and now - last_checkpoint_time >= limits.checkpoint_every_seconds
        )
        if due_steps or due_time:
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

    completed = trainer.step >= total_steps
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
            write_status(
                store,
                config.run_id,
                {**payload, "status": "paused" if paused else "stopped"},
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