"""Two-phase TPU drill exposed to the notebooks.

A drill publishes a run, stops at a checkpoint boundary, and resumes it in a new kernel.
The notebook only renders the controls and calls :func:`run_drill_phase` and
:func:`compare_drill`; the limits and trajectory comparison live here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from hamic_jax.checkpoint import ObjectStore
from hamic_jax.data import TokenizedCorpus
from hamic_jax.session import SessionLimits, SessionReport, run_session
from hamic_jax.spec import ModelPreset, TrainingConfig
from hamic_jax.train import Trainer

__all__ = [
    "DrillComparison",
    "build_drill_config",
    "compare_drill",
    "run_drill_phase",
]


@dataclass(frozen=True)
class DrillComparison:
    """Loss trajectory of a resumed drill against an uninterrupted reference."""

    start: int
    observed: tuple[float, ...]
    reference: tuple[float, ...]
    max_abs_diff: float
    passed: bool


def build_drill_config(
    base: TrainingConfig,
    *,
    run_id: str,
    steps: int,
    checkpoint_every: int,
) -> TrainingConfig:
    """Return a tiny deterministic config whose budget is exactly ``steps`` steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    return replace(
        base,
        run_id=run_id,
        total_tokens=base.tokens_per_step * steps,
        checkpoint_every_steps=checkpoint_every,
        checkpoint_every_minutes=0.0,
        eval_every_steps=0,
        log_every_steps=1,
    )


def run_drill_phase(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    store: ObjectStore,
    workdir: str | Path,
    *,
    phase: str,
    steps: int,
    stop_after: int,
    checkpoint_every: int,
    source_commit: str = "",
    raw_config: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    log: Callable[[str], None] = print,
) -> SessionReport:
    """Run phase one (stop early) or phase two (resume and finish) of a drill."""
    if phase not in ("train", "resume"):
        raise ValueError("phase must be 'train' or 'resume'")
    if not 0 < stop_after <= steps:
        raise ValueError("stop_after must be between 1 and steps")
    if phase == "train":
        limits = SessionLimits(
            steps=stop_after,
            checkpoint_every_steps=checkpoint_every,
            checkpoint_every_seconds=0.0,
            log_every_steps=1,
            checkpoint_at_end=False,
        )
    else:
        limits = SessionLimits(
            steps=steps,
            checkpoint_every_steps=checkpoint_every,
            checkpoint_every_seconds=0.0,
            log_every_steps=1,
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
    )


def compare_drill(
    config: TrainingConfig,
    preset: ModelPreset,
    corpus: TokenizedCorpus,
    *,
    steps: int,
    stop_after: int,
    checkpoint_every: int,
    resumed_losses: tuple[float, ...],
    resumed_from_step: int | None,
) -> DrillComparison:
    """Compare the resumed trajectory with a fresh uninterrupted trainer."""
    if not 0 < stop_after <= steps:
        raise ValueError("stop_after must be between 1 and steps")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    reference: list[float] = []
    trainer = Trainer(config, preset, corpus)
    for _ in range(steps):
        reference.append(float(trainer.train_step()))
    start = resumed_from_step or 0
    observed = list(resumed_losses)
    expected = (stop_after // checkpoint_every) * checkpoint_every
    diff = max(
        (abs(left - right) for left, right in zip(observed, reference[start:], strict=False)),
        default=float("inf"),
    )
    passed = (
        start == expected
        and len(observed) == steps - start
        and math.isclose(diff, 0.0, rel_tol=0.0, abs_tol=1e-6)
    )
    return DrillComparison(
        start=start,
        observed=tuple(observed),
        reference=tuple(reference),
        max_abs_diff=diff,
        passed=passed,
    )