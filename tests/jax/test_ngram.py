"""NumPy-only checks for PMI-ngram mining and whole-word units.

These run without JAX/Optax because ``hamic_jax.objectives`` imports only NumPy.
"""

from __future__ import annotations

import numpy as np

from hamic_jax.objectives import (
    MLMObjective,
    NgramTables,
    NgramUnits,
    TokenClasses,
    TokenUnits,
    VariantTable,
    WholeWordUnits,
    mine_pmi_ngrams,
)
from hamic_jax.spec import SPECIAL_TOKENS


def _classes() -> TokenClasses:
    vocab = 128
    return TokenClasses.from_texts([""] * 37 + [" " + chr(97 + i % 26) for i in range(vocab - 37)])


def test_mining_and_units() -> None:
    classes = _classes()
    ids = np.array([[37, 38, 39, 37, 38, 39]])
    tables = mine_pmi_ngrams(
        [ids[0]],
        128,
        list(SPECIAL_TOKENS.special_ids),
        classes=classes,
        max_n=3,
        max_tokens=1000,
        min_count=1,
        min_doc_count=1,
        min_pmi=-10.0,
        max_units=10,
    )
    assert any(len(values) for values in tables.tables.values())
    unit = NgramUnits(tables, classes, list(SPECIAL_TOKENS.special_ids))(
        ids, np.random.default_rng(0)
    )
    assert unit.shape == ids.shape
    assert np.all(unit[:, 1:] >= unit[:, :-1])


def test_whole_word_merges_continuations() -> None:
    classes = _classes()
    classes.spaced[38] = False
    classes.spaced[39] = True
    unit = WholeWordUnits(TokenUnits(), classes, list(SPECIAL_TOKENS.special_ids))(
        np.array([[37, 38, 39]]), np.random.default_rng(0)
    )
    assert unit.tolist() == [[0, 0, 1]]


def test_tables_round_trip(tmp_path) -> None:
    tables = NgramTables(128, {2: np.array([1, 2]), 3: np.array([3])})
    target = tmp_path / "tables.npz"
    tables.save(target)
    loaded = NgramTables.load(target)
    assert loaded.vocab_size == 128
    assert all(np.array_equal(loaded.tables[n], tables.tables[n]) for n in (2, 3))


def test_missing_tables_rejected() -> None:
    try:
        MLMObjective(
            vocab_size=128, special=SPECIAL_TOKENS, masking="pmi_ngram"
        )
    except ValueError:
        return
    raise AssertionError("pmi_ngram without tables must raise")


def test_variant_masking_preserves_the_original_label() -> None:
    variants = VariantTable.from_vocabulary([""] * 37 + ["không", "ko"])
    objective = MLMObjective(vocab_size=39, mask_rate=0.9, variant_share=0.9, variants=variants)
    batch = objective(np.full((2, 128), 37), step=0)
    selected = batch.labels == 37
    assert selected.any()
    assert (batch.input_ids[selected] == 38).any()
    assert set(batch.input_ids[selected]) <= {37, 38}