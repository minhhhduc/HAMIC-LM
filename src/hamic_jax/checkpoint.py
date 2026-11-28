"""Torch-free immutable checkpoints.

A checkpoint is a directory of safetensors payloads plus ``manifest.json`` and
``COMMITTED.json``, written last. The protocol is intentionally the same shape as the
GPU publisher so a run can be inspected with the same tooling, but this module never
imports ``hamic`` or ``torch``.

Credentials: the Hugging Face store reads ``HF_KEY`` only. ``HF_TOKEN`` belongs to the
GPU backend and is deliberately ignored here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import jax
import numpy as np
from safetensors.numpy import load_file, save_file

from hamic_jax.data import DataCursor

__all__ = [
    "CheckpointError",
    "CheckpointPublisher",
    "CheckpointResolver",
    "HuggingFaceObjectStore",
    "LocalObjectStore",
    "ObjectStore",
    "checkpoint_name",
    "load_checkpoint",
    "save_checkpoint",
]

MANIFEST_NAME = "manifest.json"
COMMIT_NAME = "COMMITTED.json"
LATEST_NAME = "LATEST.json"
PARAMS_NAME = "params.safetensors"
OPTIMIZER_NAME = "optimizer.safetensors"
META_NAME = "meta.json"
SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, corrupt, or already committed."""


def checkpoint_name(step: int) -> str:
    """Canonical checkpoint directory name."""
    return f"step-{step:09d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_manifest(directory: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not (path.parent == directory and path.name in (MANIFEST_NAME, COMMIT_NAME))
    ]
    entries = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix())
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "files": entries,
    }
    _atomic_json(directory / MANIFEST_NAME, manifest)
    return manifest


def _verify(directory: Path) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_NAME
    commit_path = directory / COMMIT_NAME
    if not manifest_path.is_file() or not commit_path.is_file():
        raise CheckpointError(f"{directory} is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(commit_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != _sha256(manifest_path):
        raise CheckpointError("manifest hash does not match COMMITTED.json")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise CheckpointError(f"missing or truncated payload: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise CheckpointError(f"sha256 mismatch: {entry['path']}")
    if not any(entry["path"] == META_NAME for entry in manifest["files"]):
        raise CheckpointError("checkpoint manifest has no state metadata")
    return json.loads((directory / META_NAME).read_bytes())


def save_checkpoint(
    directory: str | Path,
    params: Mapping[str, np.ndarray[Any, Any]],
    optimizer_state: Any,
    step: int,
    cursor: DataCursor,
    metadata: Mapping[str, Any],
) -> Path:
    """Write and commit a checkpoint directory (must not exist)."""
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()
        save_file(
            {name: np.asarray(value) for name, value in params.items()}, staging / PARAMS_NAME
        )
        leaves, _ = jax.tree_util.tree_flatten(optimizer_state)
        save_file(
            {f"opt_{index:05d}": np.asarray(leaf) for index, leaf in enumerate(leaves)},
            staging / OPTIMIZER_NAME,
        )
        meta = {
            "step": int(step),
            "cursor": cursor.to_dict(),
            "optimizer_leaves": len(leaves),
            "metadata": dict(metadata),
        }
        _atomic_json(staging / META_NAME, meta)
        manifest = _write_manifest(staging, meta)
        _atomic_json(staging / COMMIT_NAME, {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": _sha256(staging / MANIFEST_NAME),
            "files": len(manifest["files"]),
        })
        staging.rename(root)
    return root


def load_checkpoint(
    directory: str | Path,
    optimizer: Any = None,
) -> tuple[dict[str, np.ndarray[Any, Any]], Any, int, DataCursor, dict[str, Any]]:
    """Verify and load a committed checkpoint.

    ``optimizer`` is the Optax transform built for the same parameter tree. Its freshly
    initialized state supplies the structure; the saved leaves are substituted by flat
    order, so no Optax internals are serialized.
    """
    root = Path(directory)
    meta = _verify(root)
    params = load_file(root / PARAMS_NAME)
    optimizer_state: Any = None
    if optimizer is not None:
        template = jax.eval_shape(optimizer.init, params)
        treedef = jax.tree_util.tree_flatten(template)[1]
        saved = load_file(root / OPTIMIZER_NAME)
        count = int(meta["optimizer_leaves"])
        if len(saved) != count:
            raise CheckpointError("optimizer leaf count does not match metadata")
        leaves = [saved[f"opt_{index:05d}"] for index in range(count)]
        optimizer_state = jax.tree_util.tree_unflatten(treedef, leaves)
    cursor = DataCursor.from_dict(meta["cursor"])
    return params, optimizer_state, int(meta["step"]), cursor, dict(meta["metadata"])


class ObjectStore(Protocol):
    """The object-store operations needed by publishing and resolving."""

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        """Upload a batch of files."""
        ...

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        """Upload a small object."""
        ...

    def read_bytes(self, remote_path: str) -> bytes | None:
        """Read a small object."""
        ...

    def download(self, remote_path: str, destination: Path) -> Path:
        """Download one object."""
        ...

    def download_prefix(self, prefix: str, destination: Path) -> Path:
        """Download a directory prefix."""
        ...

    def list_files(self, prefix: str) -> list[str]:
        """List paths under a prefix."""
        ...

    def stat(self, remote_paths: list[str]) -> dict[str, int]:
        """Return sizes for existing paths."""
        ...


class LocalObjectStore:
    """A directory used as an object store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, remote: str) -> Path:
        target = (self.root / remote).resolve()
        if self.root.resolve() != target and self.root.resolve() not in target.parents:
            raise ValueError(f"path escapes store: {remote}")
        return target

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        del message
        for remote, local in files.items():
            target = self._path(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        del message
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def read_bytes(self, remote_path: str) -> bytes | None:
        target = self._path(remote_path)
        return target.read_bytes() if target.is_file() else None

    def download(self, remote_path: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), destination)
        return destination

    def download_prefix(self, prefix: str, destination: Path) -> Path:
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(self._path(prefix), destination)
        return destination

    def list_files(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        return sorted(path.relative_to(self.root).as_posix() for path in base.rglob("*") if path.is_file())

    def stat(self, remote_paths: list[str]) -> dict[str, int]:
        found: dict[str, int] = {}
        for remote in remote_paths:
            path = self._path(remote)
            if path.is_file():
                found[remote] = path.stat().st_size
        return found


class HuggingFaceObjectStore:
    """A private Hugging Face repository used as an object store.

    The token comes from ``HF_KEY`` (TPU backend) or the explicit ``token`` argument.
    ``HF_TOKEN`` is intentionally ignored.
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = True,
        create: bool = True,
        token: str | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        chosen = token or os.environ.get("HF_KEY")
        if not chosen:
            raise RuntimeError("set HF_KEY (TPU token) or pass token=")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._token = chosen
        self._api = HfApi(token=chosen)
        if create and not self._api.repo_exists(repo_id, repo_type=repo_type):
            self._api.create_repo(repo_id, repo_type=repo_type, private=private)

    def whoami(self) -> str:
        """Return the token's account name."""
        return str(self._api.whoami()["name"])

    def put_files(self, files: Mapping[str, Path], message: str) -> None:
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for remote, local in files.items()
        ]
        self._api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=operations,
            commit_message=message,
        )

    def put_bytes(self, remote_path: str, data: bytes, message: str) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=message,
        )

    def read_bytes(self, remote_path: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError

        try:
            path = hf_hub_download(
                self.repo_id,
                remote_path,
                repo_type=self.repo_type,
                token=self._token,
                force_download=True,
            )
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            return None
        return Path(path).read_bytes()

    def download(self, remote_path: str, destination: Path) -> Path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            self.repo_id,
            remote_path,
            repo_type=self.repo_type,
            token=self._token,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return destination

    def download_prefix(
        self, prefix: str, destination: Path, max_workers: int = 8
    ) -> Path:
        """Download every file under ``prefix`` into ``destination``.

        Lists the exact recursive tree under ``prefix`` rather than relying on
        ``snapshot_download`` glob patterns, then downloads every file into a
        sibling staging directory and atomically renames ``<staging>/<prefix>``
        to ``destination``.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        from huggingface_hub.hf_api import RepoFile
        from tqdm.auto import tqdm as auto_tqdm

        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        clean = prefix.rstrip("/")
        revision = self._api.repo_info(
            self.repo_id, repo_type=self.repo_type
        ).sha
        try:
            entries = self._api.list_repo_tree(
                self.repo_id,
                path_in_repo=clean,
                recursive=True,
                revision=revision,
                repo_type=self.repo_type,
            )
            file_entries = sorted(
                (entry for entry in entries if isinstance(entry, RepoFile)),
                key=lambda entry: entry.path,
            )
        except (EntryNotFoundError, RemoteEntryNotFoundError) as exc:
            raise FileNotFoundError(
                f"Nothing found under {prefix!r} in {self.repo_id}"
            ) from exc
        if not file_entries:
            raise FileNotFoundError(
                f"Nothing found under {prefix!r} in {self.repo_id}"
            )
        remote_files = [entry.path for entry in file_entries]
        total_bytes = sum(int(entry.size or 0) for entry in file_entries)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        lock = threading.Lock()
        done_files = 0
        bar = auto_tqdm(
            total=total_bytes or len(remote_files),
            unit="B" if total_bytes else "file",
            unit_scale=bool(total_bytes),
            desc=f"Downloading {clean}",
        )

        class _AggregatedTqdm:
            disabled = False

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _AggregatedTqdm:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def close(self) -> None:
                pass

            def refresh(self) -> None:
                pass

            def set_description(self, *args: Any, **kwargs: Any) -> None:
                pass

            def set_description_str(self, *args: Any, **kwargs: Any) -> None:
                pass

            def set_postfix_str(self, postfix: str = "", refresh: bool = False) -> None:
                del postfix, refresh

            def set_transfer_postfix_str(
                self, postfix: str = "", refresh: bool = False
            ) -> None:
                del postfix, refresh

            def update_transfer(self, n: int | float | None = 1) -> None:
                del n

            def update(self, n: int | float | None = 1) -> None:
                if total_bytes and n:
                    with lock:
                        bar.update(n)

        try:
            def _fetch(remote_path: str) -> None:
                nonlocal done_files
                hf_hub_download(
                    self.repo_id,
                    filename=remote_path,
                    repo_type=self.repo_type,
                    revision=revision,
                    local_dir=staging,
                    token=self._token,
                    tqdm_class=_AggregatedTqdm,
                )
                with lock:
                    done_files += 1
                    bar.set_postfix_str(
                        f"{done_files}/{len(remote_files)} files", refresh=False
                    )
                    if not total_bytes:
                        bar.update(1)

            try:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    list(pool.map(_fetch, remote_files))
            finally:
                bar.close()

            source = staging / clean
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Nothing found under {prefix!r} in {self.repo_id}"
                )
            missing = [
                remote_path
                for remote_path in remote_files
                if not (staging / remote_path).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete download under {prefix!r} in {self.repo_id}: "
                    f"missing {len(missing)}/{len(remote_files)} file(s), "
                    f"first={missing[0]!r}"
                )
            os.replace(source, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return destination

    def list_files(self, prefix: str) -> list[str]:
        return [
            item
            for item in self._api.list_repo_files(self.repo_id, repo_type=self.repo_type)
            if item.startswith(prefix)
        ]

    def stat(self, remote_paths: list[str]) -> dict[str, int]:
        from huggingface_hub import HfApi

        api = HfApi(token=self._token)
        found: dict[str, int] = {}
        try:
            infos = api.get_paths_info(
                self.repo_id,
                remote_paths,
                repo_type=self.repo_type,
                expand=True,
            )
        except Exception:
            return found
        for info in infos:
            size = getattr(info, "size", None)
            path = getattr(info, "path", None)
            if path is not None and size is not None:
                found[str(path)] = int(size)
        return found


@dataclass(frozen=True)
class RemoteCheckpoint:
    """A committed checkpoint in a store."""

    prefix: str
    metadata: dict[str, Any]

    @property
    def step(self) -> int:
        """Global step from the metadata."""
        return int(self.metadata["step"])


class CheckpointPublisher:
    """Publish a local directory with the commit-last protocol."""

    def __init__(self, store: ObjectStore, run_prefix: str) -> None:
        self.store = store
        self.run_prefix = run_prefix.rstrip("/")

    def publish(self, local_dir: str | Path, metadata: Mapping[str, Any]) -> RemoteCheckpoint:
        """Publish ``local_dir`` as ``step-<step>``."""
        root = Path(local_dir)
        name = checkpoint_name(int(metadata["step"]))
        prefix = f"{self.run_prefix}/checkpoints/{name}"
        if self.store.read_bytes(f"{prefix}/{COMMIT_NAME}") is not None:
            raise CheckpointError(f"{prefix} is already committed")
        manifest_path = root / MANIFEST_NAME
        manifest_hash = _sha256(manifest_path)
        marker = json.loads((root / COMMIT_NAME).read_bytes())
        if marker.get("manifest_sha256") != manifest_hash:
            raise CheckpointError("local manifest hash does not match COMMITTED.json")
        manifest = json.loads(manifest_path.read_bytes())
        saved = manifest["metadata"]
        if saved["step"] != int(metadata["step"]) or saved["metadata"] != dict(metadata):
            raise CheckpointError("local checkpoint metadata does not match the run")
        files = {f"{prefix}/{entry['path']}": root / entry["path"] for entry in manifest["files"]}
        files[f"{prefix}/{MANIFEST_NAME}"] = manifest_path
        self.store.put_files(files, message=f"checkpoint payload {name}")
        sizes = self.store.stat(list(files))
        for remote, local in files.items():
            if sizes.get(remote) != local.stat().st_size:
                raise CheckpointError(f"remote file missing or truncated: {remote}")
        marker = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": manifest_hash,
        }
        self.store.put_bytes(
            f"{prefix}/{COMMIT_NAME}", json.dumps(marker).encode(), f"commit {name}"
        )
        latest = {"checkpoint": prefix, "step": int(metadata["step"])}
        self.store.put_bytes(
            f"{self.run_prefix}/{LATEST_NAME}",
            json.dumps(latest).encode(),
            f"latest -> {name}",
        )
        return RemoteCheckpoint(prefix=prefix, metadata=dict(metadata))


class CheckpointResolver:
    """Find and restore committed checkpoints."""

    def __init__(self, store: ObjectStore, run_prefix: str) -> None:
        self.store = store
        self.run_prefix = run_prefix.rstrip("/")

    def committed(self) -> list[RemoteCheckpoint]:
        """Committed checkpoints, oldest first."""
        found: list[RemoteCheckpoint] = []
        for path in self.store.list_files(f"{self.run_prefix}/checkpoints/"):
            if not path.endswith(f"/{COMMIT_NAME}"):
                continue
            prefix = path[: -len(COMMIT_NAME) - 1]
            raw = self.store.read_bytes(f"{prefix}/{MANIFEST_NAME}")
            if raw is None:
                continue
            saved = json.loads(raw)["metadata"]
            # Older remote manifests stored the run metadata without the state wrapper.
            metadata = {**saved.get("metadata", saved), "step": saved["step"]}
            found.append(RemoteCheckpoint(prefix=prefix, metadata=metadata))
        return sorted(found, key=lambda item: item.step)

    def download(self, checkpoint: RemoteCheckpoint, destination: str | Path) -> Path:
        """Download and verify one checkpoint."""
        target = Path(destination)
        self.store.download_prefix(checkpoint.prefix, target)
        _verify(target)
        return target

    def restore_latest(self, destination: str | Path) -> RemoteCheckpoint | None:
        """Download the newest valid checkpoint, or ``None`` if no checkpoints exist."""
        final = Path(destination)
        final.parent.mkdir(parents=True, exist_ok=True)
        candidates = self.committed()
        for candidate in reversed(candidates):
            with tempfile.TemporaryDirectory(dir=final.parent) as staging:
                try:
                    staged = self.download(candidate, Path(staging) / "ckpt")
                except CheckpointError:
                    continue
                if final.exists():
                    raise FileExistsError(final)
                staged.rename(final)
                return candidate
        if candidates:
            raise CheckpointError(f"No valid committed checkpoint under {self.run_prefix!r}")
        return None