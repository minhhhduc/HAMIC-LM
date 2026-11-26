"""Published run records shared by TPU train, eval and downstream notebooks.

A run lives at ``runs/<run_id>/`` beside the checkpoint folders::

    manifest.json     immutable identity: config, preset, backend, source commit
    metrics.jsonl     one JSON object per logged step
    COMPLETE.json     written only when the configured step budget is reached
    STATUS.json       last state for a paused or deliberately stopped run

The checkpoints themselves are written by :class:`hamic_jax.checkpoint.CheckpointPublisher`
under ``runs/<run_id>/checkpoints/``. This module never trains and never decides which
parameters are valid; it only writes and lists the records the notebooks select from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hamic_jax.checkpoint import ObjectStore

__all__ = [
    "COMPLETE_NAME",
    "MANIFEST_NAME",
    "METRICS_NAME",
    "RUNS_ROOT",
    "STATUS_NAME",
    "RunRecord",
    "list_runs",
    "mark_complete",
    "publish_metrics",
    "run_prefix",
    "write_run_manifest",
    "write_status",
]

RUNS_ROOT = "runs"
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.jsonl"
COMPLETE_NAME = "COMPLETE.json"
STATUS_NAME = "STATUS.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunRecord:
    """One entry in the run list, parsed from ``manifest.json`` and markers."""

    run_id: str
    prefix: str
    manifest: dict[str, Any]
    complete: bool
    step: int | None
    status: str
    created: str


def run_prefix(run_id: str) -> str:
    """Return the Hub prefix for one run, rejecting path traversal."""
    clean = run_id.strip()
    if not clean or clean in (".", "..") or "/" in clean or "\\" in clean:
        raise ValueError(f"invalid run id: {run_id!r}")
    return f"{RUNS_ROOT}/{clean}"


def write_run_manifest(
    store: ObjectStore, manifest: dict[str, Any], message: str
) -> dict[str, Any]:
    """Create the immutable run manifest, or verify an existing one.

    The backend and training identity must match the stored record. A resume with a
    different config is a programming error and fails here instead of silently training
    a different objective under the same run id.

    Args:
        store: Object store that contains ``runs/``.
        manifest: Record with at least ``run_id``, ``backend`` and ``identity``.
        message: Commit message used when the manifest is first written.

    Returns:
        The stored manifest.
    """
    run_id = str(manifest["run_id"])
    path = f"{run_prefix(run_id)}/{MANIFEST_NAME}"
    existing = store.read_bytes(path)
    if existing is not None:
        stored = json.loads(existing)
        if stored.get("backend") != manifest.get("backend"):
            raise RuntimeError(
                f"run {run_id!r} was written by backend {stored.get('backend')!r}"
            )
        if stored.get("identity") != manifest.get("identity"):
            raise RuntimeError(f"run {run_id!r} exists with a different training identity")
        if stored.get("corpus_sha256") != manifest.get("corpus_sha256"):
            raise RuntimeError(f"run {run_id!r} exists with a different tokenized corpus")
        old_commit = stored.get("source_commit")
        new_commit = manifest.get("source_commit")
        if old_commit and new_commit and old_commit != new_commit:
            raise RuntimeError(
                f"run {run_id!r} was produced by commit {old_commit}, not {new_commit}"
            )
        return stored
    payload = {"schema_version": SCHEMA_VERSION, **manifest}
    store.put_bytes(path, json.dumps(payload, indent=2).encode(), message)
    return payload


def publish_metrics(store: ObjectStore, run_id: str, text: str, message: str) -> None:
    """Publish the complete JSONL metrics file for one run."""
    store.put_bytes(f"{run_prefix(run_id)}/{METRICS_NAME}", text.encode(), message)


def mark_complete(
    store: ObjectStore, run_id: str, payload: dict[str, Any], message: str
) -> None:
    """Write ``COMPLETE.json`` after the configured step budget is reached."""
    store.put_bytes(
        f"{run_prefix(run_id)}/{COMPLETE_NAME}",
        json.dumps(payload, indent=2).encode(),
        message,
    )


def write_status(
    store: ObjectStore, run_id: str, payload: dict[str, Any], message: str
) -> None:
    """Write the current state of a paused or deliberately stopped run."""
    store.put_bytes(
        f"{run_prefix(run_id)}/{STATUS_NAME}",
        json.dumps(payload, indent=2).encode(),
        message,
    )


def list_runs(store: ObjectStore) -> list[RunRecord]:
    """List every run that has a manifest, newest first.

    A run without ``COMPLETE.json`` is still returned so the train notebook can show
    progress; eval and downstream filter on :attr:`RunRecord.complete`.
    """
    paths = store.list_files(f"{RUNS_ROOT}/")
    run_ids = sorted(
        {
            path.split("/")[1]
            for path in paths
            if path.startswith(f"{RUNS_ROOT}/") and len(path.split("/")) > 1
        }
    )
    records: list[RunRecord] = []
    for run_id in run_ids:
        prefix = run_prefix(run_id)
        raw = store.read_bytes(f"{prefix}/{MANIFEST_NAME}")
        if raw is None:
            continue
        manifest = json.loads(raw)
        complete_raw = store.read_bytes(f"{prefix}/{COMPLETE_NAME}")
        status_raw = store.read_bytes(f"{prefix}/{STATUS_NAME}")
        complete_payload = json.loads(complete_raw) if complete_raw else {}
        status_payload = json.loads(status_raw) if status_raw else {}
        records.append(
            RunRecord(
                run_id=run_id,
                prefix=prefix,
                manifest=manifest,
                complete=bool(complete_payload),
                step=complete_payload.get("step"),
                status=str(status_payload.get("status", "running")),
                created=str(manifest.get("created", "")),
            )
        )
    return sorted(records, key=lambda record: record.created, reverse=True)