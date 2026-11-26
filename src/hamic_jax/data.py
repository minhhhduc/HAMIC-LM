"""Tokenized shards, the deterministic packer, and multi-host row sharding.

The on-disk format is the GPU backend's ``hamic-tokenized-npy-v1``:
``<name>.tokens.npy``, ``<name>.offsets.npy``, optional ``<name>.sources.npy`` and a
``tokenized_manifest.json``. Ids carry no special tokens; the packer adds bos/eos.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hamic_jax.spec import TrainingConfig

__all__ = [
    "DataCursor",
    "PackedBatches",
    "TokenizedCorpus",
    "TokenizedShard",
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

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-friendly view."""
        return {
            "offset": self.offset,
            "consumed_tokens": self.consumed_tokens,
            "epoch": self.epoch,
            "token_offset": self.token_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataCursor:
        """Rebuild a cursor from checkpoint metadata."""
        return cls(
            offset=int(data.get("offset", 0)),
            consumed_tokens=int(data.get("consumed_tokens", 0)),
            epoch=int(data.get("epoch", 0)),
            token_offset=int(data.get("token_offset", 0)),
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

    def shards(self, split: str) -> tuple[TokenizedShard, ...]:
        """Return the split's shards, loading the memmaps once."""
        if split in self._cache:
            return self._cache[split]
        entry = self.manifest["splits"][split]
        directory = self.root / split
        shards: list[TokenizedShard] = []
        for record in entry["shards"]:
            files = record["files"]
            tokens = np.load(directory / files["tokens"]["path"], mmap_mode="r")
            offsets = np.load(directory / files["offsets"]["path"], mmap_mode="r")
            sources = (
                np.load(directory / files["sources"]["path"], mmap_mode="r")
                if "sources" in files
                else None
            )
            shards.append(TokenizedShard(tokens=tokens, offsets=offsets, sources=sources))
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
    corpus.verify()
    return corpus


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