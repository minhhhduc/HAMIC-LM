"""Matplotlib charts for the TPU notebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from hamic_jax.drill import DrillComparison
from hamic_jax.lab import EvaluationRow
from hamic_jax.session import SessionReport

__all__ = ["downstream_bars", "drill_comparison", "evaluation_bars", "loss_curve"]


def loss_curve(report: SessionReport, *, title: str = "Training loss") -> Figure:
    """Plot the loss of the steps produced by this session, with eval points."""
    start = report.resumed_from_step or 0
    rows = [
        record
        for record in report.metrics
        if int(record.get("step", 0)) > start and "loss" in record
    ]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if rows:
        steps = [int(record["step"]) for record in rows]
        losses = [float(record["loss"]) for record in rows]
        ax.plot(steps, losses, linewidth=1.2, label="train loss")
        evaluations = [record for record in rows if "eval_loss" in record]
        if evaluations:
            ax.scatter(
                [int(record["step"]) for record in evaluations],
                [float(record["eval_loss"]) for record in evaluations],
                color="tab:orange",
                label="eval loss",
                zorder=3,
            )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No steps in this session", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def drill_comparison(comparison: DrillComparison) -> Figure:
    """Plot resumed losses against the uninterrupted reference trajectory."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if comparison.observed:
        steps = range(comparison.start + 1, comparison.start + len(comparison.observed) + 1)
        reference = [
            comparison.reference[comparison.start + index]
            for index in range(len(comparison.observed))
        ]
        ax.plot(steps, comparison.observed, marker="o", label="resumed")
        ax.plot(steps, reference, marker="x", linestyle="--", label="reference")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No resumed steps to compare", ha="center", va="center")
        ax.set_axis_off()
    verdict = "passed" if comparison.passed else "failed"
    ax.set_title(f"Drill {verdict} · max |diff| = {comparison.max_abs_diff:.3e}")
    fig.tight_layout()
    return fig


def evaluation_bars(rows: Sequence[EvaluationRow]) -> Figure:
    """Plot validation loss for every evaluated run and masking view."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if rows:
        labels = [f"{row.run_id}\n{row.view} @ {row.step}" for row in rows]
        losses = [row.eval_loss for row in rows]
        ax.bar(range(len(rows)), losses, color="tab:blue")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("validation loss")
        ax.set_title("Evaluation by view")
    else:
        ax.text(0.5, 0.5, "No evaluation rows", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def downstream_bars(results: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Plot the best development score for each fine-tuned task."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    items = [(name, result) for name, result in results.items()]
    labels = [name for name, _ in items]
    scores = []
    metrics = []
    for _, result in items:
        rates = result.get("dev_by_rate") or {}
        scores.append(max(rates.values()) if rates else float("nan"))
        metrics.append(str(result.get("metric", "score")))
    if items:
        ax.bar(range(len(items)), scores, color="tab:green")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        for position, score in enumerate(scores):
            if math.isfinite(score):
                ax.text(position, score, f"{score:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("best dev " + " / ".join(sorted(set(metrics))))
        ax.set_title("Downstream tasks")
    else:
        ax.text(0.5, 0.5, "No downstream tasks", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig