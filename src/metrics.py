"""Ordinal-aware evaluation metrics.

Severity is ordered 1 (minimal) -> 4 (severe), so accuracy alone cannot
express *direction* of error. A downgrade (predicted less severe than gold)
is the clinically dangerous failure mode and is reported explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from src.schema import LEVELS, LEVEL_NAMES, to_domain_label


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(LEVELS), average="macro", zero_division=0)
    )


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    scores = f1_score(
        y_true, y_pred, labels=list(LEVELS), average=None, zero_division=0
    )
    return {LEVEL_NAMES[level]: float(s) for level, s in zip(LEVELS, scores)}


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        cohen_kappa_score(y_true, y_pred, labels=list(LEVELS), weights="quadratic")
    )


def direction_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Upgrade / ok / downgrade breakdown of errors.

    downgrade = pred < gold (under-estimates severity)
    upgrade   = pred > gold (over-estimates severity)
    """
    delta = y_pred.astype(int) - y_true.astype(int)
    n = int(delta.size)
    n_ok = int((delta == 0).sum())
    n_down = int((delta < 0).sum())
    n_up = int((delta > 0).sum())
    errors = n - n_ok
    return {
        "n": n,
        "ok": n_ok,
        "downgrade": n_down,
        "upgrade": n_up,
        "error_rate": errors / n if n else 0.0,
        "downgrade_rate": n_down / n if n else 0.0,
        "upgrade_rate": n_up / n if n else 0.0,
        "downgrade_share_of_errors": (n_down / errors) if errors else 0.0,
        "mean_signed_error": float(delta.mean()) if n else 0.0,
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(LEVELS))


def compute_metrics_for_trainer(eval_pred) -> dict[str, float]:
    """HF Trainer hook. Selects checkpoints on macro-F1.

    The dataset yields model-space labels (0..3), so convert both sides
    back to the domain scale (1..4) before scoring.
    """
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = to_domain_label(np.argmax(logits, axis=-1))
    labels = to_domain_label(np.asarray(labels))
    return {"macro_f1": macro_f1(labels, preds), "mae": mae(labels, preds)}


def full_report(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, Any]:
    """Everything we put in the results table for one configuration."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    report: dict[str, Any] = {
        "macro_f1": macro_f1(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "qwk": qwk(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
    }
    report.update(per_class_f1(y_true, y_pred))
    report["direction"] = direction_matrix(y_true, y_pred)
    report["confusion"] = confusion(y_true, y_pred).tolist()
    report["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=list(LEVELS),
        target_names=[LEVEL_NAMES[l] for l in LEVELS],
        zero_division=0,
        output_dict=True,
    )
    if prefix:
        report = {f"{prefix}{k}": v for k, v in report.items()}
    return report
