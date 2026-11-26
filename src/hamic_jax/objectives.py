"""Masked-LM and causal-LM objectives.

Selection follows HAMIC's GPU policy: units (tokens, geometric spans, or PMI
collocations) are selected with probability ``mask_rate``; every position in a selected
unit receives one action: mask, random regular token, spelling variant, or unchanged.
The default shares are BERT's 80/10/0.

Why these are NumPy and not a library: no dependency in HAMIC's set owns masking-unit
selection or PMI mining. Optax owns the loss elsewhere; this module only builds the
``input_ids``/``labels`` pair and the selection units.
"""

from __future__ import annotations

import itertools
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from hamic_jax.spec import SpecialTokens, read_yaml

__all__ = [
    "CLMObjective",
    "MLMObjective",
    "MaskedBatch",
    "NgramTables",
    "NgramUnits",
    "SpanUnits",
    "TokenClasses",
    "TokenUnits",
    "VariantTable",
    "WholeWordUnits",
    "mine_pmi_ngrams",
    "mine_pmi_pairs",
    "word_tokens",
]

_IGNORE_INDEX = -100


def _in_sorted(table: np.ndarray[Any, Any], keys: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if table.size == 0 or keys.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    where = np.searchsorted(table, keys).clip(max=table.size - 1)
    return table[where] == keys


@dataclass(frozen=True)
class MaskedBatch:
    """The two arrays consumed by the model for one microbatch."""

    input_ids: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]


class TokenUnits:
    """One selection unit per position."""

    def __call__(
        self, ids: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        del rng
        return np.broadcast_to(np.arange(ids.shape[1]), ids.shape).copy()


class SpanUnits:
    """Geometric spans truncated to ``[1, max_length]``."""

    def __init__(self, p: float = 0.2, max_length: int = 5) -> None:
        if not 0 < p <= 1 or max_length < 1:
            raise ValueError("need 0 < p <= 1 and max_length >= 1")
        self.p = p
        self.max_length = max_length

    def __call__(
        self, ids: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        batch, length = ids.shape
        uniform = rng.random((batch, length))
        if self.p >= 1:
            lengths = np.ones((batch, length), dtype=np.int64)
        else:
            tail = 1.0 - (1.0 - self.p) ** self.max_length
            lengths = np.floor(np.log1p(-uniform * tail) / np.log1p(-self.p)).astype(np.int64) + 1
            lengths = np.clip(lengths, 1, self.max_length)
        ends = np.cumsum(lengths, axis=1)
        starts = np.zeros((batch, length + 1), dtype=bool)
        starts[np.arange(batch)[:, None], np.clip(ends - lengths, 0, length)] = True
        starts[:, 0] = True
        return np.cumsum(starts[:, :length], axis=1) - 1


class PmiUnits:
    """Join neighbouring tokens whose pair key is in a mined collocation table."""

    def __init__(
        self,
        pairs: np.ndarray[Any, Any],
        vocab_size: int,
        special_ids: Iterable[int],
        keep: float = 1.0,
        max_length: int | None = None,
    ) -> None:
        if not 0 < keep <= 1:
            raise ValueError("keep must be in (0, 1]")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.pairs = np.sort(np.asarray(pairs, dtype=np.int64))
        self.vocab_size = vocab_size
        self.special = np.asarray(sorted(set(special_ids)), dtype=np.int64)
        lut = np.zeros(vocab_size, dtype=bool)
        lut[self.special] = True
        self._special_lut = lut
        self.keep = keep
        self.max_length = max_length

    def __call__(
        self, ids: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        regular = ~self._special_lut[ids]
        keys = ids[:, :-1].astype(np.int64) * self.vocab_size + ids[:, 1:].astype(np.int64)
        cand = regular[:, :-1] & regular[:, 1:]
        joined = np.zeros(keys.shape, dtype=bool)
        if cand.any():
            joined[cand] = _in_sorted(self.pairs, keys[cand])
        if self.keep < 1:
            joined &= rng.random(keys.shape) < self.keep
        starts = np.ones_like(ids, dtype=bool)
        starts[:, 1:] = ~joined
        if self.max_length is not None:
            positions = np.broadcast_to(np.arange(ids.shape[1]), ids.shape)
            last_start = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
            offset = positions - last_start
            starts |= (offset > 0) & (offset % self.max_length == 0)
        return np.cumsum(starts, axis=1) - 1



@dataclass(frozen=True)
class TokenClasses:
    """Per-id flags derived from the tokenizer vocabulary.

    ``word`` is letters/digits with at most one leading space, ``alpha`` is letters only,
    and ``spaced`` records the leading space.
    """

    word: np.ndarray[Any, Any]
    spaced: np.ndarray[Any, Any]
    alpha: np.ndarray[Any, Any]

    @classmethod
    def from_tokenizer_file(cls, path: str | Any) -> TokenClasses:
        """Classify every id of a byte-level ``tokenizer.json``."""
        return cls.from_texts(_decoded_vocabulary(path))

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> TokenClasses:
        """Classify decoded token texts (index = id)."""
        word = np.zeros(len(texts), dtype=bool)
        spaced = np.zeros(len(texts), dtype=bool)
        alpha = np.zeros(len(texts), dtype=bool)
        for index, text in enumerate(texts):
            core = text[1:] if text.startswith(" ") else text
            word[index] = bool(core) and core.isalnum()
            alpha[index] = bool(core) and core.isalpha()
            spaced[index] = text.startswith(" ")
        return cls(word=word, spaced=spaced, alpha=alpha)


def _continuation(
    ids: np.ndarray[Any, Any], word: np.ndarray[Any, Any], spaced: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Positions that continue the word of the previous token."""
    is_word = word[ids]
    cont = is_word & ~spaced[ids]
    if ids.ndim == 1:
        cont[1:] &= is_word[:-1]
        cont[:1] = False
    else:
        cont[:, 1:] &= is_word[:, :-1]
        cont[:, 0] = False
    return cont


def _complete_starts(
    ids: np.ndarray[Any, Any], classes: TokenClasses, special: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Positions holding a whole syllable (letter token, not continued, not continuing)."""
    cont = _continuation(ids, classes.word, classes.spaced)
    complete = classes.alpha[ids] & ~cont & ~special[ids]
    complete[:-1] &= ~cont[1:]
    return complete


def _ngram_keys(ids: np.ndarray[Any, Any], n: int, base: int) -> np.ndarray[Any, Any]:
    keys = np.zeros(ids.size - n + 1, dtype=np.int64)
    for offset in range(n):
        keys = keys * base + ids[offset : ids.size - n + 1 + offset]
    return keys


def _valid_windows(complete: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
    valid = complete[: complete.size - n + 1].copy()
    for offset in range(1, n):
        valid &= complete[offset : complete.size - n + 1 + offset]
    return valid


class _NgramCounter:
    """Exact (key, count, document count) totals with periodic in-memory compaction."""

    def __init__(self, compact_rows: int) -> None:
        self.keys: list[np.ndarray[Any, Any]] = []
        self.counts: list[np.ndarray[Any, Any]] = []
        self.docs: list[np.ndarray[Any, Any]] = []
        self.rows = 0
        self.compact_rows = compact_rows

    def add_document(self, keys: np.ndarray[Any, Any]) -> None:
        if keys.size == 0:
            return
        unique, counts = np.unique(keys, return_counts=True)
        self.keys.append(unique)
        self.counts.append(counts.astype(np.int64))
        self.docs.append(np.ones(unique.size, dtype=np.int64))
        self.rows += unique.size
        if self.rows >= self.compact_rows:
            self._compact()

    def _compact(self) -> None:
        if len(self.keys) <= 1:
            return
        keys = np.concatenate(self.keys)
        counts = np.concatenate(self.counts)
        docs = np.concatenate(self.docs)
        order = np.argsort(keys, kind="stable")
        keys, counts, docs = keys[order], counts[order], docs[order]
        first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        self.keys = [keys[first]]
        self.counts = [np.add.reduceat(counts, first)]
        self.docs = [np.add.reduceat(docs, first)]
        self.rows = int(first.size)
        self.compact_rows = max(self.compact_rows, 2 * self.rows)

    def totals(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if not self.keys:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty
        self.compact_rows = 0
        self._compact()
        return self.keys[0], self.counts[0], self.docs[0]


@dataclass(frozen=True)
class NgramTables:
    """Mined n-gram keys per length (sorted ``int64``, ids packed in base ``vocab_size``)."""

    vocab_size: int
    tables: dict[int, np.ndarray[Any, Any]]

    def save(self, path: str | Any) -> None:
        """Write an ``.npz`` file."""
        arrays: dict[str, Any] = {f"n{n}": keys for n, keys in self.tables.items()}
        arrays["vocab_size"] = np.int64(self.vocab_size)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Any) -> NgramTables:
        """Read a file written by :meth:`save`."""
        with np.load(path) as data:
            tables = {int(name[1:]): data[name] for name in data.files if name.startswith("n")}
            return cls(vocab_size=int(data["vocab_size"]), tables=tables)

    def __len__(self) -> int:
        return sum(keys.size for keys in self.tables.values())


def _compositions(n: int) -> list[tuple[int, ...]]:
    """Every split of ``n`` into at least two positive contiguous part lengths."""
    result = []
    for cuts in range(1, n):
        for positions in itertools.combinations(range(1, n), cuts):
            bounds = (0, *positions, n)
            result.append(tuple(b - a for a, b in itertools.pairwise(bounds)))
    return result


def _dominant_extension(
    keys: np.ndarray[Any, Any],
    counts: np.ndarray[Any, Any],
    longer: np.ndarray[Any, Any],
    longer_counts: np.ndarray[Any, Any],
    n: int,
    base: int,
) -> np.ndarray[Any, Any]:
    """Largest share of an n-gram's occurrences taken by one left or right neighbour."""
    dominant = np.zeros(keys.size)
    if longer.size == 0:
        return dominant
    for part in (longer // base, longer % base**n):
        order = np.argsort(part, kind="stable")
        part, part_counts = part[order], longer_counts[order]
        first = np.flatnonzero(np.r_[True, part[1:] != part[:-1]])
        groups, best = part[first], np.maximum.reduceat(part_counts, first)
        where = np.searchsorted(groups, keys).clip(max=groups.size - 1)
        hit = groups[where] == keys
        dominant[hit] = np.maximum(dominant[hit], best[where[hit]] / counts[hit])
    return dominant


def mine_pmi_ngrams(
    documents: Iterable[np.ndarray[Any, Any]],
    vocab_size: int,
    special_ids: Iterable[int],
    classes: TokenClasses,
    max_n: int = 3,
    max_tokens: int = 100_000_000,
    min_count: int = 50,
    min_doc_count: int = 5,
    min_pmi: float = 3.0,
    max_units: int = 200_000,
    discount_k: float = 0.0,
    max_extension_share: float = 0.6,
    compact_rows: int = 20_000_000,
) -> NgramTables:
    """Mine cohesive n-grams of complete syllables (PMI-Masking, Levine et al. 2021)."""
    if not 2 <= max_n <= 3:
        raise ValueError("max_n must be in [2, 3]")
    if vocab_size ** (max_n + 1) >= 2**63:
        raise ValueError("vocab_size ** (max_n + 1) does not fit in int64 keys")
    if min(max_tokens, min_count, min_doc_count, max_units) < 1 or discount_k < 0:
        raise ValueError("counts must be positive and discount_k non-negative")
    if not 0 < max_extension_share <= 1:
        raise ValueError("max_extension_share must be in (0, 1]")
    count_n = max_n + 1
    special = np.zeros(vocab_size, dtype=bool)
    special[list(set(special_ids))] = True
    counters = {n: _NgramCounter(compact_rows) for n in range(1, count_n + 1)}
    windows = dict.fromkeys(range(1, count_n + 1), 0)
    seen = 0
    for document in documents:
        ids = np.asarray(document, dtype=np.int64)
        if ids.size == 0:
            continue
        complete = _complete_starts(ids, classes, special)
        for n in range(1, count_n + 1):
            if ids.size < n:
                break
            valid = _valid_windows(complete, n)
            windows[n] += int(valid.sum())
            counters[n].add_document(_ngram_keys(ids, n, vocab_size)[valid])
        seen += ids.size
        if seen >= max_tokens:
            break
    totals = {n: counters[n].totals() for n in counters}
    log_p = {}
    for n, (keys, counts, _) in totals.items():
        log_p[n] = (keys, np.log(np.maximum(counts, 1)) - np.log(max(windows[n], 1)))

    def lookup(n: int, keys: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        table, values = log_p[n]
        result = np.full(keys.size, np.inf)
        if table.size:
            where = np.searchsorted(table, keys).clip(max=table.size - 1)
            hit = table[where] == keys
            result[hit] = values[where[hit]]
        return result

    kept_keys, kept_n, kept_score = [], [], []
    for n in range(2, max_n + 1):
        keys, counts, docs = totals[n]
        keep = (counts >= min_count) & (docs >= min_doc_count)
        keys, counts = keys[keep], counts[keep]
        if max_extension_share < 1:
            dominant = _dominant_extension(keys, counts, *totals[n + 1][:2], n, vocab_size)
            free = dominant < max_extension_share
            keys, counts = keys[free], counts[free]
        if keys.size == 0:
            continue
        whole = log_p[n][1][np.searchsorted(log_p[n][0], keys)]
        pmi = np.full(keys.size, np.inf)
        for parts in _compositions(n):
            parts_log_p = np.zeros(keys.size)
            end = n
            for length in reversed(parts):
                start = end - length
                part = (keys // vocab_size ** (n - end)) % vocab_size**length
                parts_log_p += lookup(length, part)
                end = start
            pmi = np.minimum(pmi, whole - parts_log_p)
        strong = pmi >= min_pmi
        score = pmi if discount_k == 0 else pmi * counts / (counts + discount_k)
        kept_keys.append(keys[strong])
        kept_n.append(np.full(int(strong.sum()), n))
        kept_score.append(score[strong])
    tables: dict[int, np.ndarray[Any, Any]] = {
        n: np.zeros(0, dtype=np.int64) for n in range(2, max_n + 1)
    }
    if kept_keys:
        keys = np.concatenate(kept_keys)
        lengths = np.concatenate(kept_n)
        score = np.concatenate(kept_score)
        best = np.lexsort((keys, lengths, -score))[:max_units]
        for n in tables:
            tables[n] = np.sort(keys[best][lengths[best] == n])
    return NgramTables(vocab_size=vocab_size, tables=tables)


class WholeWordUnits:
    """Wrap other units so word-continuation pieces join the unit before them."""

    def __init__(
        self, inner: Any, classes: TokenClasses, special_ids: Iterable[int]
    ) -> None:
        self.inner = inner
        self.word = classes.word.copy()
        self.word[list(set(special_ids))] = False
        self.spaced = classes.spaced.copy()

    def __call__(
        self, ids: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        unit = self.inner(ids, rng)
        starts = np.ones_like(ids, dtype=bool)
        starts[:, 1:] = unit[:, 1:] != unit[:, :-1]
        starts &= ~_continuation(ids, self.word, self.spaced)
        return np.cumsum(starts, axis=1) - 1


class NgramUnits:
    """Mask listed n-grams of complete syllables as one unit, longest match first."""

    def __init__(
        self,
        tables: NgramTables,
        classes: TokenClasses,
        special_ids: Iterable[int],
        keep: float = 1.0,
    ) -> None:
        if not 0 < keep <= 1:
            raise ValueError("keep must be in (0, 1]")
        self.base = tables.vocab_size
        self.tables = {
            n: np.sort(keys.astype(np.int64))
            for n, keys in sorted(tables.tables.items(), reverse=True)
            if keys.size
        }
        self.word = classes.word.copy()
        self.word[list(set(special_ids))] = False
        self.spaced = classes.spaced.copy()
        self.keep = keep

    def __call__(
        self, ids: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        ids = np.asarray(ids, dtype=np.int64)
        batch, length = ids.shape
        cont = _continuation(ids, self.word, self.spaced)
        complete = self.word[ids] & ~cont
        complete[:, :-1] &= ~cont[:, 1:]
        joined = cont.copy()
        occupied = np.zeros((batch, length), dtype=bool)
        for n, table in self.tables.items():
            if length < n:
                continue
            width = length - n + 1
            keys = np.zeros((batch, width), dtype=np.int64)
            valid = np.ones((batch, width), dtype=bool)
            free = np.ones((batch, width), dtype=bool)
            for offset in range(n):
                keys = keys * self.base + ids[:, offset : offset + width]
                valid &= complete[:, offset : offset + width]
                free &= ~occupied[:, offset : offset + width]
            cand = valid & free
            match = np.zeros((batch, width), dtype=bool)
            if cand.any():
                match[cand] = _in_sorted(table, keys[cand])
            if self.keep < 1:
                match &= rng.random((batch, width)) < self.keep
            blocked = np.zeros_like(match)
            for back in range(1, n):
                blocked[:, back:] |= match[:, :-back]
            accepted = match & ~blocked
            for offset in range(n):
                occupied[:, offset : offset + width] |= accepted
                if offset:
                    joined[:, offset : offset + width] |= accepted
        starts = ~joined
        return np.cumsum(starts, axis=1) - 1

@dataclass(frozen=True)
class VariantTable:
    """Existing single-token Vietnamese variants, indexed by original token ID."""

    offsets: np.ndarray[Any, Any]
    values: np.ndarray[Any, Any]

    @classmethod
    def from_vocabulary(cls, tokens: list[str]) -> VariantTable:
        """Build case, tone, diacritic and lexicon variants without importing Torch."""
        lexicon = read_yaml(Path(__file__).parents[1] / "hamic/resources/teencode.yaml")
        words = lexicon.get("words", {})
        tone_marks = frozenset("̣̀́̃̉")
        lookup: dict[str, int] = {}
        tones: dict[str, list[int]] = defaultdict(list)
        keys: dict[int, str] = {}
        for index, text in enumerate(tokens):
            if text.removeprefix(" ").isalpha():
                lookup.setdefault(text, index)
                key = unicodedata.normalize("NFC", "".join(
                    c for c in unicodedata.normalize("NFD", text) if c not in tone_marks
                ))
                keys[index] = key
                tones[key].append(index)
        offsets = [0]
        values: list[int] = []
        for index, text in enumerate(tokens):
            if index in keys:
                core = text.removeprefix(" ")
                space = text[:len(text) - len(core)]
                plain = "".join(
                    c for c in unicodedata.normalize("NFD", core)
                    if unicodedata.category(c) != "Mn"
                ).translate(str.maketrans("đĐ", "dD"))
                forms = {plain, core.lower(), core.upper(), core.capitalize()}
                if variants := words.get(core.lower()):
                    variant = str(variants[0])
                    forms.add(variant[:1].upper() + variant[1:] if core[0].isupper() else variant)
                found = set(tones[keys[index]])
                found.update(lookup[space + form] for form in forms if space + form in lookup)
                found.discard(index)
                values.extend(sorted(found))
            offsets.append(len(values))
        return cls(np.asarray(offsets, dtype=np.int64), np.asarray(values, dtype=np.int64))

    @classmethod
    def from_tokenizer_file(cls, path: str | Path) -> VariantTable:
        """Decode the vocabulary and build the table once before training."""
        return cls.from_vocabulary(_decoded_vocabulary(path))

    def sample(
        self, ids: np.ndarray[Any, Any], uniform: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Pick a variant where one exists, leaving other IDs unchanged."""
        start = self.offsets[ids]
        count = self.offsets[ids + 1] - start
        available = count > 0
        result = ids.copy()
        pick = (uniform[available] * count[available]).astype(np.int64)
        result[available] = self.values[start[available] + pick]
        return result, available


@dataclass
class MLMObjective:
    """Dynamic MLM masking with deterministic per-step randomness."""

    vocab_size: int
    special: SpecialTokens = field(default_factory=SpecialTokens)
    seed: int = 0
    mask_rate: float = 0.15
    mask_rate_final: float | None = None
    schedule_start: float = 0.0
    total_steps: int = 1
    mask_share: float = 0.8
    random_share: float = 0.1
    variant_share: float = 0.0
    variants: VariantTable | None = None
    masking: str = "token"
    span_max_length: int = 5
    pmi_pairs: np.ndarray[Any, Any] | None = None
    pmi_keep: float = 1.0
    pmi_max_unit_length: int | None = None
    whole_word_masking: bool = False
    ngram_tables: NgramTables | None = None
    token_classes: TokenClasses | None = None
    _unit_fn: Any = field(init=False, repr=False, compare=False)
    _special_lut: np.ndarray[Any, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must be in (0, 1)")
        if self.masking not in ("token", "span", "pmi", "pmi_ngram"):
            raise ValueError("unsupported masking policy")
        if self.masking == "pmi_ngram" and (
            self.ngram_tables is None or self.token_classes is None
        ):
            raise ValueError("PMI-ngram masking needs mined tables and token classes")
        if self.whole_word_masking and self.token_classes is None:
            raise ValueError("whole-word masking needs token classes")
        if self.mask_share + self.random_share > 1:
            raise ValueError("mask_share + random_share must be <= 1")
        if not 0 <= self.variant_share <= self.mask_share + self.random_share:
            raise ValueError("variant_share must be in [0, mask_share + random_share]")
        if self.variant_share and self.variants is None:
            raise ValueError("variant_share needs a spelling-variant table")
        self._unit_fn = self._units()
        lut = np.zeros(self.vocab_size, dtype=bool)
        lut[list(self.special.special_ids)] = True
        self._special_lut = lut

    def rate_at(self, step: int) -> float:
        """Mask rate at a global micro-step."""
        if self.mask_rate_final is None:
            return self.mask_rate
        start = round(self.total_steps * self.schedule_start)
        span = max(1, self.total_steps - start)
        progress = min(1.0, max(0.0, (step - start) / span))
        return self.mask_rate + (self.mask_rate_final - self.mask_rate) * progress

    def _units(self) -> Any:
        if self.masking == "span":
            units: Any = SpanUnits(max_length=self.span_max_length)
        elif self.masking == "pmi":
            if self.pmi_pairs is None:
                raise ValueError("PMI masking needs a mined pair table")
            units = PmiUnits(
                self.pmi_pairs,
                self.vocab_size,
                self.special.special_ids,
                keep=self.pmi_keep,
                max_length=self.pmi_max_unit_length,
            )
        elif self.masking == "pmi_ngram":
            if self.ngram_tables is None or self.token_classes is None:
                raise ValueError("PMI-ngram masking needs mined tables and token classes")
            units = NgramUnits(
                self.ngram_tables,
                self.token_classes,
                self.special.special_ids,
                keep=self.pmi_keep,
            )
        else:
            units = TokenUnits()
        if self.whole_word_masking:
            if self.token_classes is None:
                raise ValueError("whole-word masking needs token classes")
            units = WholeWordUnits(units, self.token_classes, self.special.special_ids)
        return units

    def __call__(self, input_ids: np.ndarray[Any, Any], step: int) -> MaskedBatch:
        """Return masked inputs and labels for one microbatch."""
        rng = np.random.default_rng(self.seed * 1_000_003 + step)
        ids = np.asarray(input_ids, dtype=np.int64)
        maskable = ~self._special_lut[ids]
        unit = self._unit_fn(ids, rng)
        rate = self.rate_at(step)
        selected = np.take_along_axis(rng.random(ids.shape) < rate, unit, axis=1)
        selected &= maskable
        labels = np.where(selected, ids, _IGNORE_INDEX)
        action = np.take_along_axis(rng.random(ids.shape), unit, axis=1)
        random_ids = rng.integers(
            self.special.first_regular_id, self.vocab_size, size=ids.shape, dtype=np.int64
        )
        masked = ids.copy()
        taken_from_random = min(self.variant_share, self.random_share)
        mask_end = self.mask_share - (self.variant_share - taken_from_random)
        random_end = mask_end + (self.random_share - taken_from_random)
        to_mask = selected & (action < mask_end)
        to_random = selected & (action >= mask_end) & (action < random_end)
        if self.variant_share and self.variants is not None:
            to_variant = selected & (action >= random_end) & (action < random_end + self.variant_share)
            variants, available = self.variants.sample(ids, rng.random(ids.shape))
            masked[to_variant & available] = variants[to_variant & available]
            to_random |= to_variant & ~available
        masked[to_mask] = self.special.mask_id
        masked[to_random] = random_ids[to_random]
        return MaskedBatch(input_ids=masked, labels=labels)


@dataclass(frozen=True)
class CLMObjective:
    """Causal LM: every token is a label; the loss shifts by one."""

    def __call__(self, input_ids: np.ndarray[Any, Any], step: int) -> MaskedBatch:
        del step
        ids = np.asarray(input_ids, dtype=np.int64)
        return MaskedBatch(input_ids=ids, labels=ids.copy())


def _byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level alphabet used by the HAMIC tokenizer."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _decoded_vocabulary(path: str | Any) -> list[str]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    byte_of = _byte_decoder()
    vocab = data["model"]["vocab"]
    tokens = [""] * (max(vocab.values()) + 1)
    for piece, index in vocab.items():
        raw = bytes(byte_of[c] for c in piece if c in byte_of)
        try:
            tokens[index] = raw.decode("utf-8")
        except UnicodeDecodeError:
            tokens[index] = ""
    for added in data.get("added_tokens", []):
        if 0 <= int(added["id"]) < len(tokens):
            tokens[int(added["id"])] = ""
    return tokens


def word_tokens(tokenizer_file: str | Any) -> np.ndarray[Any, Any]:
    """Boolean mask of vocabulary entries made only of letters and digits.

    Matches the GPU predicate: one optional leading space, then an alphabetic core, or a
    stripped alphanumeric core. Tokenizer pieces often start with a space, so rejecting
    every leading space would empty the table.
    """
    tokens = _decoded_vocabulary(tokenizer_file)
    mask = []
    for text in tokens:
        core = text[1:] if text.startswith(" ") else text
        stripped = text.strip()
        mask.append(bool(core) and (core.isalpha() or (stripped.isalnum() and stripped != "")))
    return np.asarray(mask, dtype=bool)


def mine_pmi_pairs(
    documents: Iterable[np.ndarray[Any, Any]],
    vocab_size: int,
    special_ids: Sequence[int],
    max_tokens: int = 100_000_000,
    min_count: int = 50,
    min_pmi: float = 3.0,
    max_pairs: int = 200_000,
    eligible: np.ndarray[Any, Any] | None = None,
    discount_k: float = 0.0,
) -> np.ndarray[Any, Any]:
    """Mine frequent, high-PMI adjacent pairs from the train split.

    This mirrors ``hamic.training.masking.mine_pmi_pairs`` without torch. The in-memory
    implementation is intended for the token budgets a TPU host can hold; a spilled
    implementation can be added without changing the table contract.
    """
    if max_tokens < 1 or min_count < 1 or max_pairs < 1:
        raise ValueError("max_tokens, min_count and max_pairs must be positive")
    special = np.asarray(sorted(set(special_ids)), dtype=np.int64)
    unigram = np.zeros(vocab_size, dtype=np.int64)
    chunks: list[np.ndarray[Any, Any]] = []
    seen = 0
    for document in documents:
        ids = np.asarray(document, dtype=np.int64)
        if ids.size == 0:
            continue
        unigram += np.bincount(ids, minlength=vocab_size)
        seen += ids.size
        if ids.size > 1:
            keep = ~(np.isin(ids[:-1], special) | np.isin(ids[1:], special))
            chunks.append((ids[:-1] * vocab_size + ids[1:])[keep])
        if seen >= max_tokens:
            break
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    keys, counts = np.unique(np.concatenate(chunks), return_counts=True)
    total_pairs = counts.sum()
    total_tokens = unigram.sum()
    keep = counts >= min_count
    if eligible is not None:
        keep &= eligible[keys // vocab_size] & eligible[keys % vocab_size]
    keys, counts = keys[keep], counts[keep]
    left, right = keys // vocab_size, keys % vocab_size
    pmi = np.log(counts / total_pairs) - np.log(
        unigram[left] / total_tokens * unigram[right] / total_tokens
    )
    strong = pmi >= min_pmi
    keys, counts, pmi = keys[strong], counts[strong], pmi[strong]
    score = pmi if discount_k == 0 else pmi * counts / (counts + discount_k)
    best = np.argsort(-score, kind="stable")[:max_pairs]
    return np.sort(keys[best])