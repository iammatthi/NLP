"""Unit tests for ordinal metrics (src/metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    compute_metrics_for_trainer,
    confusion,
    direction_matrix,
    full_report,
    macro_f1,
    mae,
    per_class_f1,
    qwk,
)

GOLD = np.array([1, 2, 3, 4, 1, 2, 3, 4])


def test_perfect_predictions():
    assert macro_f1(GOLD, GOLD.copy()) == 1.0
    assert mae(GOLD, GOLD.copy()) == 0.0
    assert qwk(GOLD, GOLD.copy()) == 1.0
    scores = per_class_f1(GOLD, GOLD.copy())
    assert scores == {"minimal": 1.0, "mild": 1.0, "moderate": 1.0, "severe": 1.0}
    direction = direction_matrix(GOLD, GOLD.copy())
    assert direction["ok"] == 8 and direction["downgrade"] == 0
    assert direction["upgrade"] == 0 and direction["downgrade_share_of_errors"] == 0.0


def test_mae_known_offset():
    pred = GOLD + 1
    assert mae(GOLD, pred) == 1.0


def test_direction_matrix_counts_downgrades_and_upgrades():
    # 1 ok, 1 downgrade (pred<gold), 1 upgrade (pred>gold).
    true = np.array([1, 2, 3])
    pred = np.array([2, 1, 3])
    direction = direction_matrix(true, pred)
    assert direction["ok"] == 1
    assert direction["downgrade"] == 1
    assert direction["upgrade"] == 1
    assert direction["error_rate"] == pytest.approx(2 / 3)
    assert direction["downgrade_rate"] == pytest.approx(1 / 3)
    assert direction["downgrade_share_of_errors"] == 0.5
    assert direction["mean_signed_error"] == pytest.approx(0.0)


def test_direction_matrix_empty():
    direction = direction_matrix(np.array([], dtype=int), np.array([], dtype=int))
    assert direction["n"] == 0
    assert direction["error_rate"] == 0.0
    assert direction["mean_signed_error"] == 0.0


def test_confusion_is_4x4_in_level_order():
    matrix = confusion(GOLD, GOLD.copy())
    assert matrix.shape == (4, 4)
    assert matrix.diagonal().tolist() == [2, 2, 2, 2]


def test_trainer_hook_shifts_model_labels_back_to_domain():
    # One-hot logits at model classes 0..3 must score as domain 1..4.
    logits = np.eye(4, dtype=float)
    labels = np.array([0, 1, 2, 3])
    result = compute_metrics_for_trainer((logits, labels))
    assert result == {"macro_f1": 1.0, "mae": 0.0}
    # Same when logits are wrapped in a tuple (Trainer quirk).
    assert compute_metrics_for_trainer(((logits,), labels)) == result


def test_full_report_keys_and_prefix():
    report = full_report(GOLD, GOLD.copy())
    for key in (
        "macro_f1",
        "mae",
        "qwk",
        "accuracy",
        "minimal",
        "direction",
        "confusion",
    ):
        assert key in report
    assert report["accuracy"] == 1.0
    assert report["direction"]["ok"] == 8
    prefixed = full_report(GOLD, GOLD.copy(), prefix="x_")
    assert "x_macro_f1" in prefixed and "macro_f1" not in prefixed
