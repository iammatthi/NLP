"""Unit tests for bootstrap CIs and Jev graded/calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import bootstrap_cis
from evaluate_jev import compute_ece, compute_graded_metrics

PERFECT = np.array([1, 2, 3, 4] * 10)


def test_bootstrap_perfect_predictions_pin_cis():
    cis = bootstrap_cis(PERFECT, PERFECT.copy(), n_boot=50, seed=42)
    assert cis["macro_f1"] == [1.0, 1.0]
    assert cis["downgrade_rate"] == [0.0, 0.0]
    assert cis["qwk"] == [1.0, 1.0]


def test_bootstrap_is_seed_deterministic_and_ordered():
    pred = PERFECT.copy()
    pred[0] = 1 if pred[0] > 1 else 4
    first = bootstrap_cis(PERFECT, pred, n_boot=50, seed=7)
    second = bootstrap_cis(PERFECT, pred, n_boot=50, seed=7)
    assert first == second
    for bounds in first.values():
        assert bounds[0] <= bounds[1]


def test_graded_metrics_from_direction_counts():
    graded = compute_graded_metrics({"ok": 70, "upgrade": 10, "downgrade": 20})
    assert graded["graded_precision"] == pytest.approx(70 / 80)
    assert graded["graded_recall"] == pytest.approx(70 / 90)
    expected_f = (
        2
        * graded["graded_precision"]
        * graded["graded_recall"]
        / (graded["graded_precision"] + graded["graded_recall"])
    )
    assert graded["graded_fscore"] == pytest.approx(expected_f)


def test_graded_metrics_zero_division_is_zero():
    assert compute_graded_metrics({"ok": 0, "upgrade": 0, "downgrade": 0}) == {
        "graded_precision": 0.0,
        "graded_recall": 0.0,
        "graded_fscore": 0.0,
    }


def test_ece_zero_when_confident_and_correct():
    y_true = np.array([1, 2])
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert compute_ece(y_true, y_true.copy(), probs) == pytest.approx(0.0)


def test_ece_penalises_underconfidence():
    y_true = np.array([1, 2])
    probs = np.full((2, 4), 0.25)
    assert compute_ece(y_true, y_true.copy(), probs) == pytest.approx(0.75)
