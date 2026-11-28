"""Tokenized shards, the deterministic packer, and multi-host row sharding.

The on-disk format is the GPU backend's ``hamic-tokenized-npy-v1``:
``<name>.tokens.npy``, ``<name>.offsets.npy``, optional ``<name>.sources.npy`` and a
``tokenized_manifest.json``. Ids carry no special tokens; the packer adds bos/eos.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hamic_jax.spec import TrainingConfig

__all__ = [
    "DataCursor",
    "PackedBatches",
    "ShardFetcher",
    "TokenizedCorpus",
    "TokenizedShard",
    "WindowedPackedBatches",
    "fetch_corpus_metadata",
    "fetch_tokenized",
    "shard_rows",
    "tokenized_prefix",
]

MANIFEST_NAME = "tokenized_manifest.json"
TOKENIZER_NAME = "tokenizer.json"


@dataclass(frozen=True)
class DataCursor:
    """Exact position of the next token to serve."""

    offset: int = 0
    consumed_tokens: int = 0
    epoch: int = 0
    token_offset: int = 0
    window: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-friendly view."""
        return {
            "offset": self.offset,
            "consumed_tokens": self.consumed_tokens,
            "epoch": self.epoch,
            "token_offset": self.token_offset,
            "window": self.window,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataCursor:
        """Rebuild a cursor from checkpoint metadata."""
        return cls(
            offset=int(data.get("offset", 0)),
            consumed_tokens=int(data.get("consumed_tokens", 0)),
            epoch=int(data.get("epoch", 0)),
            token_offset=int(data.get("token_offset", 0)),
            window=int(data.get("window", 0)),
        )


@dataclass(frozen=True)
class TokenizedShard:
    """One memory-mapped shard."""

    tokens: np.ndarray[Any, Any]
    offsets: np.ndarray[Any, Any]
    sources: np.ndarray[Any, Any] | None

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def document(self, index: int) -> np.ndarray[Any, Any]:
        """Return the token ids of one document."""
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokenizedCorpus:
    """A directory of memory-mapped tokenized shards."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{self.root} has no {MANIFEST_NAME}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "hamic-tokenized-npy-v1":
            raise ValueError("unsupported tokenized format")
        self.source_ids: list[str] | None = self.manifest.get("source_ids")
        self._cache: dict[str, tuple[TokenizedShard, ...]] = {}

    @property
    def tokenizer_sha256(self) -> str | None:
        """SHA-256 recorded for the tokenizer copy, if the manifest has one."""
        tokenizer = self.manifest.get("tokenizer") or {}
        value = tokenizer.get("sha256")
        return str(value) if value else None

    @property
    def manifest_sha256(self) -> str:
        """Fingerprint the corpus layout and recorded shard checksums for strict resume."""
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size recorded in the manifest."""
        return int((self.manifest.get("tokenizer") or {})["vocab_size"])

    def verify(self) -> None:
        """Raise ``ValueError`` when a shard or tokenizer differs from its checksum."""
        for split, entry in self.manifest["splits"].items():
            for shard in entry["shards"]:
                for record in shard["files"].values():
                    path = self.root / split / record["path"]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if _file_sha256(path) != record["sha256"]:
                        raise ValueError(f"Checksum mismatch: {path}")
        expected = self.tokenizer_sha256
        if expected is not None:
            path = self.tokenizer_file()
            if _file_sha256(path) != expected:
                raise ValueError(f"Checksum mismatch: {path}")

    def shard_count(self, split: str) -> int:
        """Number of shards recorded for ``split``."""
        return len(self.manifest["splits"][split]["shards"])

    def shard_entries(self, split: str) -> list[dict[str, Any]]:
        """Manifest shard records for ``split``."""
        return list(self.manifest["splits"][split]["shards"])

    def load_shard(self, split: str, index: int) -> TokenizedShard:
        """Memory-map one shard without requiring sibling shards on disk."""
        record = self.manifest["splits"][split]["shards"][index]
        directory = self.root / split
        files = record["files"]
        tokens = np.load(directory / files["tokens"]["path"], mmap_mode="r")
        offsets = np.load(directory / files["offsets"]["path"], mmap_mode="r")
        sources = (
            np.load(directory / files["sources"]["path"], mmap_mode="r")
            if "sources" in files
            else None
        )
        return TokenizedShard(tokens=tokens, offsets=offsets, sources=sources)

    def verify_shard(self, split: str, index: int) -> None:
        """Verify existence and SHA-256 of every file belonging to one shard."""
        record = self.manifest["splits"][split]["shards"][index]
        for entry in record["files"].values():
            path = self.root / split / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if _file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")

    def evict_shard(self, split: str, index: int) -> None:
        """Delete local files for one shard to reclaim disk space."""
        self._cache.pop(split, None)
        record = self.manifest["splits"][split]["shards"][index]
        for entry in record["files"].values():
            path = self.root / split / entry["path"]
            if path.is_file():
                path.unlink()

    def shards(self, split: str) -> tuple[TokenizedShard, ...]:
        """Return the split's shards, loading the memmaps once."""
        if split in self._cache:
            return self._cache[split]
        shards = [self.load_shard(split, index) for index in range(self.shard_count(split))]
        self._cache[split] = tuple(shards)
        return self._cache[split]

    def documents(self, split: str) -> Iterator[np.ndarray[Any, Any]]:
        """Yield train or eval documents in corpus order."""
        for shard in self.shards(split):
            for index in range(len(shard)):
                yield shard.document(index)

    def tokenizer_file(self) -> Path:
        """Return the tokenizer file that travels with the shards."""
        path = self.root / TOKENIZER_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{self.root} has no {TOKENIZER_NAME}")
        expected = self.tokenizer_sha256
        if expected is not None and _file_sha256(path) != expected:
            raise ValueError(f"{path} does not match the tokenizer recorded in the manifest")
        return path

    def token_byte_lengths(self) -> np.ndarray[Any, Any]:
        """UTF-8 bytes each token id decodes to (``0`` for special and added tokens).

        Bits per byte is comparable across tokenizers, unlike loss per token. The
        HAMIC tokenizers count their virtual spaces as bytes, so their bits per byte
        read about 0.4% low against a tokenizer without virtual spaces.

        Raises:
            FileNotFoundError: The directory has no tokenizer copy.
            ValueError: The copy does not match the manifest, or is not byte-level BPE.
        """
        data = json.loads(self.tokenizer_file().read_text(encoding="utf-8"))
        decoder = data.get("decoder") or {}
        kinds = {entry.get("type") for entry in decoder.get("decoders", [])}
        kinds.add(decoder.get("type"))
        if "ByteLevel" not in kinds:
            raise ValueError("byte lengths need a byte-level tokenizer")
        lengths = np.zeros(self.vocab_size, dtype=np.int64)
        for token, index in data["model"]["vocab"].items():
            lengths[index] = len(token)
        for added in data.get("added_tokens", []):
            lengths[added["id"]] = 0
        return lengths


def tokenized_prefix(path: str | Path) -> str:
    """Return the Hub prefix of a tokenized directory (``tokenized/<version>/<name>``)."""
    local = Path(path)
    return f"tokenized/{local.parent.name}/{local.name}"


def fetch_tokenized(
    store: Any, prefix: str, destination: str | Path
) -> TokenizedCorpus:
    """Download a published tokenized corpus and verify every checksum.

    An existing destination is reused only when it verifies. A partial or corrupt copy is
    removed and downloaded again, matching the GPU ``fetch_tokenized`` contract.

    Args:
        store: An object store with ``read_bytes`` and ``download_prefix``.
        prefix: Remote folder such as ``tokenized/v0.14/t1-bbpe-20544``.
        destination: Local directory that receives the shards.

    Raises:
        FileNotFoundError: The manifest is absent or a listed file is missing.
        ValueError: A shard or tokenizer hash differs from the manifest.
    """
    target = Path(destination)
    if (target / MANIFEST_NAME).is_file():
        try:
            corpus = TokenizedCorpus(target)
            corpus.verify()
            return corpus
        except (FileNotFoundError, ValueError):
            pass
    if target.exists():
        shutil.rmtree(target)
    if store.read_bytes(f"{prefix}/{MANIFEST_NAME}") is None:
        raise FileNotFoundError(f"No tokenized manifest under {prefix!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    store.download_prefix(prefix, target)
    corpus = TokenizedCorpus(target)
    try:
        corpus.verify()
    except FileNotFoundError as exc:
        present = sorted(
            p.relative_to(target).as_posix()
            for p in target.rglob("*")
            if p.is_file()
        )
        shutil.rmtree(target, ignore_errors=True)
        raise FileNotFoundError(
            f"{exc} (downloaded {len(present)} file(s) under {prefix!r}: {present[:10]})"
        ) from exc
    return corpus


class ShardFetcher:
    """On-demand downloader, verifier, and evictor for individual corpus shards."""

    def __init__(self, store: Any, prefix: str, corpus: TokenizedCorpus) -> None:
        self.store = store
        self.prefix = prefix.rstrip("/")
        self.corpus = corpus
        self._lock = threading.Lock()

    def ensure_shards(self, split: str, indices: Sequence[int]) -> None:
        """Download and verify any missing or corrupt shards in ``indices``."""
        with self._lock:
            entries = self.corpus.shard_entries(split)
            missing: list[str] = []
            for idx in indices:
                try:
                    self.corpus.verify_shard(split, idx)
                except (FileNotFoundError, ValueError):
                    for record in entries[idx]["files"].values():
                        missing.append(f"{self.prefix}/{split}/{record['path']}")
            if missing:
                missing = sorted(set(missing))
                if hasattr(self.store, "download_files"):
                    self.store.download_files(
                        missing,
                        self.corpus.root,
                        strip_prefix=self.prefix,
                        desc=f"Downloading {self.prefix}/{split}",
                    )
                else:
                    for remote in missing:
                        rel = remote[len(self.prefix) + 1 :]
                        self.store.download(remote, self.corpus.root / rel)
                for idx in indices:
                    self.corpus.verify_shard(split, idx)

    def ensure_tokens(self, split: str, min_tokens: int) -> list[int]:
        """Ensure a prefix of ``split`` covering at least ``min_tokens`` is on disk."""
        entries = self.corpus.shard_entries(split)
        if not entries:
            raise ValueError(f"Split {split!r} has no shards")
        chosen: list[int] = []
        total = 0
        for idx, entry in enumerate(entries):
            chosen.append(idx)
            total += int(entry.get("tokens", 0))
            if total >= min_tokens:
                break
        self.ensure_shards(split, chosen)
        return chosen

    def evict_shards(self, split: str, indices: Sequence[int]) -> None:
        """Delete local shard files in ``indices``."""
        with self._lock:
            for idx in indices:
                self.corpus.evict_shard(split, idx)
            if indices and hasattr(self.store, "purge_cache"):
                self.store.purge_cache()


def fetch_corpus_metadata(
    store: Any,
    prefix: str,
    destination: str | Path,
    *,
    eval_tokens: int = 0,
) -> tuple[TokenizedCorpus, ShardFetcher]:
    """Download manifest, tokenizer, and minimal validation shards for windowed streaming."""
    clean_prefix = prefix.rstrip("/")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        raw = store.read_bytes(f"{clean_prefix}/{MANIFEST_NAME}")
        if raw is None:
            raise FileNotFoundError(f"No tokenized manifest under {prefix!r}")
        manifest_path.write_bytes(raw)
    corpus = TokenizedCorpus(target)
    expected_tok = corpus.tokenizer_sha256
    tok_path = target / TOKENIZER_NAME
    if expected_tok is not None:
        valid_tok = tok_path.is_file() and _file_sha256(tok_path) == expected_tok
        if not valid_tok:
            store.download(f"{clean_prefix}/{TOKENIZER_NAME}", tok_path)
            corpus.tokenizer_file()
    elif not tok_path.is_file():
        if store.read_bytes(f"{clean_prefix}/{TOKENIZER_NAME}") is not None:
            store.download(f"{clean_prefix}/{TOKENIZER_NAME}", tok_path)
    fetcher = ShardFetcher(store, clean_prefix, corpus)
    if eval_tokens > 0 and "validation" in corpus.manifest.get("splits", {}):
        fetcher.ensure_tokens("validation", eval_tokens)
    return corpus, fetcher


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PackedBatches:
    """Deterministic, resumable packed sequences.

    Documents are visited in a seeded per-epoch permutation; each contributes
    ``[bos] + tokens + [eos]``. Rows are filled across document boundaries, so there is
    no padding. ``source_weights`` repeat documents within an epoch exactly as the GPU
    packer does.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus,
        split: str,
        seq_len: int,
        batch_size: int,
        bos_id: int,
        eos_id: int,
        seed: int = 0,
        shuffle: bool = True,
        cursor: DataCursor | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if seq_len < 2 or batch_size < 1:
            raise ValueError("seq_len must be >= 2 and batch_size >= 1")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.seed = seed
        self.shuffle = shuffle
        self._shards = list(corpus.shards(split))
        self._starts = np.cumsum([0] + [len(shard) for shard in self._shards])
        self.documents = int(self._starts[-1])
        if self.documents == 0:
            raise ValueError(f"Split {split!r} has no documents")
        self._visits = (
            self._document_weights(corpus, source_weights) if source_weights else None
        )
        start = cursor or DataCursor()
        self.epoch = start.epoch
        self.position = start.offset
        self.token_offset = start.token_offset
        self.consumed_tokens = start.consumed_tokens
        self._order = self._permutation(self.epoch)

    def restore(self, cursor: DataCursor) -> None:
        """Restore every cursor field and rebuild the epoch permutation."""
        self.epoch = cursor.epoch
        self.position = cursor.offset
        self.token_offset = cursor.token_offset
        self.consumed_tokens = cursor.consumed_tokens
        self._order = self._permutation(self.epoch)

    def _document_weights(
        self, corpus: TokenizedCorpus, source_weights: dict[str, float]
    ) -> np.ndarray[Any, Any]:
        source_ids = corpus.source_ids
        if source_ids is None or any(shard.sources is None for shard in self._shards):
            raise ValueError("source_weights need per-document sources")
        unknown = set(source_weights) - set(source_ids)
        if unknown:
            raise ValueError(f"Unknown sources in source_weights: {sorted(unknown)}")
        table = np.asarray(
            [source_weights.get(name, 1.0) for name in source_ids], dtype=np.float64
        )
        sources = np.concatenate(
            [shard.sources for shard in self._shards if shard.sources is not None]
        )
        return table[sources.astype(np.int64)]

    def _permutation(self, epoch: int) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng([self.seed, epoch])
        if self._visits is not None:
            whole = np.floor(self._visits)
            extra = rng.random(self.documents) < self._visits - whole
            order = np.repeat(np.arange(self.documents), (whole + extra).astype(np.int64))
            if order.size == 0:
                raise ValueError("source_weights leave no documents to visit")
            return rng.permutation(order) if self.shuffle else order
        if not self.shuffle:
            return np.arange(self.documents)
        return rng.permutation(self.documents)

    def _piece(self, global_index: int) -> np.ndarray[Any, Any]:
        shard_index = int(np.searchsorted(self._starts, global_index, side="right")) - 1
        tokens = self._shards[shard_index].document(
            global_index - int(self._starts[shard_index])
        )
        piece = np.empty(tokens.size + 2, dtype=np.int64)
        piece[0], piece[-1] = self.bos_id, self.eos_id
        piece[1:-1] = tokens
        return piece

    def _advance_document(self) -> None:
        self.position += 1
        self.token_offset = 0
        if self.position >= self._order.size:
            self.epoch += 1
            self.position = 0
            self._order = self._permutation(self.epoch)

    def next_batch(self) -> np.ndarray[Any, Any]:
        """Return the next ``[batch_size, seq_len]`` int64 block."""
        out = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for row in range(self.batch_size):
            filled = 0
            while filled < self.seq_len:
                piece = self._piece(int(self._order[self.position]))
                take = min(self.seq_len - filled, piece.size - self.token_offset)
                out[row, filled : filled + take] = piece[
                    self.token_offset : self.token_offset + take
                ]
                filled += take
                self.token_offset += take
                if self.token_offset >= piece.size:
                    self._advance_document()
        self.consumed_tokens += out.size
        return out

    def cursor(self) -> DataCursor:
        """Return the position of the next token to serve."""
        return DataCursor(
            offset=self.position,
            consumed_tokens=self.consumed_tokens,
            epoch=self.epoch,
            token_offset=self.token_offset,
        )


class WindowedPackedBatches:
    """Deterministic, resumable packed sequences streamed one shard window at a time."""

    def __init__(
        self,
        corpus: TokenizedCorpus,
        split: str,
        seq_len: int,
        batch_size: int,
        bos_id: int,
        eos_id: int,
        shard_window: int,
        *,
        fetcher: ShardFetcher | None = None,
        seed: int = 0,
        shuffle: bool = True,
        cursor: DataCursor | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if seq_len < 2 or batch_size < 1 or shard_window < 1:
            raise ValueError("seq_len >= 2, batch_size >= 1, shard_window >= 1 required")
        self._corpus = corpus
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.shard_window = shard_window
        self._fetcher = fetcher
        self.seed = seed
        self.shuffle = shuffle
        self.num_shards = corpus.shard_count(split)
        if self.num_shards == 0:
            raise ValueError(f"Split {split!r} has no shards")
        self.num_windows = math.ceil(self.num_shards / shard_window)
        self._source_weights = dict(source_weights) if source_weights else None
        self._source_table: np.ndarray[Any, Any] | None = None
        if self._source_weights:
            source_ids = corpus.source_ids
            if source_ids is None:
                raise ValueError("source_weights need per-document sources")
            unknown = set(self._source_weights) - set(source_ids)
            if unknown:
                raise ValueError(f"Unknown sources in source_weights: {sorted(unknown)}")
            self._source_table = np.asarray(
                [self._source_weights.get(name, 1.0) for name in source_ids],
                dtype=np.float64,
            )
        start = cursor or DataCursor()
        self.epoch = start.epoch
        self.window = start.window % self.num_windows
        self.position = start.offset
        self.token_offset = start.token_offset
        self.consumed_tokens = start.consumed_tokens
        self._loaded_key: tuple[int, int] | None = None
        self._active_indices: list[int] = []
        self._shards: list[TokenizedShard] = []
        self._starts = np.zeros(1, dtype=np.int64)
        self._order = np.zeros(0, dtype=np.int64)
        self._completed_shards: list[list[int]] = []

    @property
    def has_completed_window(self) -> bool:
        """Whether at least one shard window finished since the last eviction."""
        return bool(self._completed_shards)

    def window_indices(self, epoch: int, window: int) -> list[int]:
        """Shard indices assigned to ``(epoch, window)``."""
        rng = np.random.default_rng([self.seed, epoch])
        order = (
            rng.permutation(self.num_shards)
            if self.shuffle
            else np.arange(self.num_shards)
        )
        start = window * self.shard_window
        return [int(i) for i in order[start : start + self.shard_window]]

    def restore(self, cursor: DataCursor) -> None:
        """Restore every cursor field without eagerly downloading an unused window."""
        self.epoch = cursor.epoch
        self.window = cursor.window % self.num_windows
        self.position = cursor.offset
        self.token_offset = cursor.token_offset
        self.consumed_tokens = cursor.consumed_tokens
        self._loaded_key = None
        self._completed_shards.clear()

    def _ensure_loaded(self) -> None:
        if self._loaded_key == (self.epoch, self.window):
            return
        checked = 0
        while True:
            indices = self.window_indices(self.epoch, self.window)
            if self._fetcher is not None:
                self._fetcher.ensure_shards(self.split, indices)
            shards = [self._corpus.load_shard(self.split, idx) for idx in indices]
            starts = np.cumsum([0] + [len(shard) for shard in shards])
            total_docs = int(starts[-1])
            rng = np.random.default_rng([self.seed, self.epoch, self.window])
            if self._source_table is not None:
                if any(shard.sources is None for shard in shards):
                    raise ValueError("source_weights need per-document sources")
                sources = np.concatenate(
                    [shard.sources for shard in shards if shard.sources is not None]
                )
                visits = self._source_table[sources.astype(np.int64)]
                whole = np.floor(visits)
                extra = rng.random(total_docs) < visits - whole
                base = np.repeat(np.arange(total_docs), (whole + extra).astype(np.int64))
                order = rng.permutation(base) if self.shuffle else base
            elif not self.shuffle:
                order = np.arange(total_docs)
            else:
                order = rng.permutation(total_docs)
            if order.size > 0:
                self._active_indices = indices
                self._shards = shards
                self._starts = starts
                self._order = order
                self._loaded_key = (self.epoch, self.window)
                return
            checked += 1
            if checked >= self.num_windows:
                raise ValueError("source_weights leave no documents to visit")
            self._completed_shards.append(indices)
            self.window += 1
            if self.window >= self.num_windows:
                self.epoch += 1
                self.window = 0
            self.position = 0
            self.token_offset = 0

    def _piece(self, local_index: int) -> np.ndarray[Any, Any]:
        shard_index = int(np.searchsorted(self._starts, local_index, side="right")) - 1
        tokens = self._shards[shard_index].document(
            local_index - int(self._starts[shard_index])
        )
        piece = np.empty(tokens.size + 2, dtype=np.int64)
        piece[0], piece[-1] = self.bos_id, self.eos_id
        piece[1:-1] = tokens
        return piece

    def _advance_document(self) -> None:
        self.position += 1
        self.token_offset = 0
        if self.position >= self._order.size:
            self._completed_shards.append(list(self._active_indices))
            self.window += 1
            if self.window >= self.num_windows:
                self.epoch += 1
                self.window = 0
            self.position = 0
            self._loaded_key = None

    def next_batch(self) -> np.ndarray[Any, Any]:
        """Return the next ``[batch_size, seq_len]`` int64 block."""
        out = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for row in range(self.batch_size):
            filled = 0
            while filled < self.seq_len:
                self._ensure_loaded()
                piece = self._piece(int(self._order[self.position]))
                take = min(self.seq_len - filled, piece.size - self.token_offset)
                out[row, filled : filled + take] = piece[
                    self.token_offset : self.token_offset + take
                ]
                filled += take
                self.token_offset += take
                if self.token_offset >= piece.size:
                    self._advance_document()
        self.consumed_tokens += out.size
        return out

    def cursor(self) -> DataCursor:
        """Return the position of the next token to serve."""
        return DataCursor(
            offset=self.position,
            consumed_tokens=self.consumed_tokens,
            epoch=self.epoch,
            token_offset=self.token_offset,
            window=self.window,
        )

    def evict_completed(self) -> list[int]:
        """Delete shards from finished windows that are not needed right now."""
        if not self._completed_shards:
            return []
        candidates = {
            idx for batch in self._completed_shards for idx in batch
        }
        self._completed_shards.clear()
        protected = set(self.window_indices(self.epoch, self.window))
        next_epoch, next_window = (
            (self.epoch + 1, 0)
            if self.window + 1 >= self.num_windows
            else (self.epoch, self.window + 1)
        )
        protected |= set(self.window_indices(next_epoch, next_window))
        evict = sorted(candidates - protected)
        if evict and self._fetcher is not None:
            self._fetcher.evict_shards(self.split, evict)
        return evict


def shard_rows(
    rows: np.ndarray[Any, Any], host_id: int, host_count: int
) -> np.ndarray[Any, Any]:
    """Contiguous row slice owned by one data-parallel host."""
    if not 0 <= host_id < host_count:
        raise ValueError("host_id must be in [0, host_count)")
    per_host = rows.shape[0] // host_count
    if per_host < 1:
        raise ValueError("batch is smaller than the host count")
    return rows[host_id * per_host : (host_id + 1) * per_host]


def config_batches(
    corpus: TokenizedCorpus,
    config: TrainingConfig,
    split: str,
    cursor: DataCursor | None = None,
    source_weights: dict[str, float] | None = None,
) -> PackedBatches:
    """Build a packer from a training config."""
    from hamic_jax.spec import SpecialTokens

    special = SpecialTokens()
    return PackedBatches(
        corpus,
        split,
        seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        bos_id=special.bos_id,
        eos_id=special.eos_id,
        seed=config.seed,
        shuffle=config.shuffle,
        cursor=cursor,
        source_weights=source_weights,
    )