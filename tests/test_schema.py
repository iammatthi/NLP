"""Unit tests for the unified severity schema (src/schema.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema import (
    LEVELS,
    _validate,
    class_weights,
    load,
    load_split,
    make_splits,
    save_splits,
    to_domain_label,
    to_model_label,
)


def test_label_shift_round_trip_scalar_and_array():
    assert to_model_label(1) == 0
    assert to_model_label(4) == 3
    assert to_domain_label(0) == 1
    assert to_domain_label(3) == 4
    levels = np.array([1, 2, 3, 4])
    assert to_domain_label(to_model_label(levels)).tolist() == levels.tolist()


def test_class_weights_are_inverse_frequency_normalised():
    labels = np.array([1] * 8 + [2] * 4 + [3] * 2 + [4] * 2)
    weights = class_weights(labels)
    assert weights.mean() == pytest.approx(1.0)
    # Rarest classes are upweighted; ordering follows LEVELS.
    assert weights[0] < weights[1] < weights[2]
    assert weights[2] == pytest.approx(2 * weights[1])
    assert weights[3] == pytest.approx(weights[2])


def test_class_weights_missing_class_raises():
    with pytest.raises(ValueError, match="missing class"):
        class_weights(np.array([1, 2, 3, 3]))


def _frame(n_per_level: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                f"post-{level}-{i}" for level in LEVELS for i in range(n_per_level)
            ],
            "label": [level for level in LEVELS for _ in range(n_per_level)],
        }
    )


def test_make_splits_deterministic_stratified_and_disjoint():
    frame = _frame()
    train, val, test = make_splits(frame, seed=42)
    assert (len(train), len(val), len(test)) == (32, 4, 4)
    for part in (train, val, test):
        assert set(part["label"].unique()) == set(LEVELS)
    seen = set(train["text"]) | set(val["text"]) | set(test["text"])
    assert seen == set(frame["text"])
    assert len(seen) == 40
    again = make_splits(frame, seed=42)
    for first, second in zip((train, val, test), again):
        assert first["text"].tolist() == second["text"].tolist()


def test_make_splits_val_zero_and_bad_fractions():
    frame = _frame()
    train, val, test = make_splits(frame, seed=42, train=0.8, val=0.0)
    assert len(val) == 0 and len(train) == 32 and len(test) == 8
    with pytest.raises(ValueError, match="train"):
        make_splits(frame, train=0.9, val=0.2)
    with pytest.raises(ValueError, match="train"):
        make_splits(frame, train=0.0, val=0.1)


def test_validate_rejects_bad_labels_and_empty_text():
    with pytest.raises(ValueError, match="unexpected label"):
        _validate(pd.DataFrame({"text": ["a"], "label": [5]}))
    with pytest.raises(ValueError, match="empty text"):
        _validate(pd.DataFrame({"text": [""], "label": [1]}))


def test_load_unknown_language_raises():
    with pytest.raises(ValueError, match="unknown language"):
        load("fr")


def test_minimum_alias_maps_to_minimal(tmp_path):
    from src.schema import load_english

    path = tmp_path / "en.csv"
    pd.DataFrame({"text": ["a", "b"], "label": ["minimum", "severe"]}).to_csv(
        path, index=False
    )
    frame = load_english(path)
    assert frame["label"].tolist() == [1, 4]


@pytest.mark.parametrize("bad", ["urgent", ""])
def test_unmapped_english_label_raises(tmp_path, bad):
    from src.schema import load_english

    path = tmp_path / "en.csv"
    pd.DataFrame({"text": ["a"], "label": [bad]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unmapped"):
        load_english(path)


def test_save_and_load_split_round_trip(tmp_path):
    frame = _frame()
    train, val, test = make_splits(frame, seed=42)
    save_splits("bn", train, val, test, out_dir=tmp_path)
    for name, expected in (("train", train), ("val", val), ("test", test)):
        actual = load_split("bn", name, out_dir=tmp_path)
        assert actual["label"].tolist() == expected["label"].tolist()
        assert actual["text"].tolist() == expected["text"].tolist()


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_splits"):
        load_split("bn", "test", out_dir=tmp_path)
