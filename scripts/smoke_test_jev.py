"""Smoke test for the Experiment 6 Jev ('System One') integration.

By default runs offline to verify:
1. `typesafe_sdk` imports and `Choice` / `Score` question construction.
2. Alignment of clinical rubrics with `src.schema.LEVELS = (1, 2, 3, 4)`.
3. Response parsing (`parse_system_one_response`) into domain labels `1..4`.
4. Full metric report generation via `src.metrics.full_report`.

Pass `--live` to additionally send 1 Bengali validation row and 1 English
validation row to the live TypeSafe API (`TYPESAFE_API_KEY`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.jev_client import (
    NASEEM_DDA_CHOICE_CRITERIA,
    NASEEM_DDA_SCORE_CRITERIA,
    SEVERITY_CHOICE_CRITERIA,
    SEVERITY_SCORE_CRITERIA,
    build_questions,
    load_env_key,
    parse_system_one_response,
    predict_split,
)
from src.metrics import full_report
from src.schema import LEVELS, LEVEL_NAMES, load_split


@dataclass
class _MockChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class _MockScoreAnswer:
    score: float
    probabilities: list[float]
    confidence: float


@dataclass
class _MockSystemOneResponse:
    choices: dict[str, Any]
    scores: dict[str, Any]


def run_offline_checks() -> bool:
    print("=== 1. Offline Schema & SDK Build Check ===")
    questions = build_questions("unified")
    questions_dda = build_questions("naseem_dda")
    assert "severity_choice" in questions and "severity_score" in questions
    assert "severity_choice" in questions_dda and "severity_score" in questions_dda
    assert len(SEVERITY_CHOICE_CRITERIA) == 4
    assert len(SEVERITY_SCORE_CRITERIA) == 4
    assert len(NASEEM_DDA_CHOICE_CRITERIA) == 4
    assert len(NASEEM_DDA_SCORE_CRITERIA) == 4
    for idx, lvl in enumerate(LEVELS):
        name = LEVEL_NAMES[lvl]
        assert name in SEVERITY_CHOICE_CRITERIA
        assert SEVERITY_SCORE_CRITERIA[idx] == SEVERITY_CHOICE_CRITERIA[name]
        assert name in NASEEM_DDA_CHOICE_CRITERIA
        assert NASEEM_DDA_SCORE_CRITERIA[idx] == NASEEM_DDA_CHOICE_CRITERIA[name]
    print("  rubric schema alignment : OK (unified & naseem_dda, 4 levels each)")

    # Test mock parsing across all 4 tiers
    mock_resps = [
        _MockSystemOneResponse(
            choices={
                "severity_choice": _MockChoiceAnswer(
                    choice="minimal",
                    probabilities={"minimal": 0.80, "mild": 0.15, "moderate": 0.04, "severe": 0.01},
                    confidence=0.85,
                )
            },
            scores={
                "severity_score": _MockScoreAnswer(
                    score=1.26,
                    probabilities=[0.78, 0.18, 0.03, 0.01],
                    confidence=0.83,
                )
            },
        ),
        _MockSystemOneResponse(
            choices={
                "severity_choice": _MockChoiceAnswer(
                    choice="mild",
                    probabilities={"minimal": 0.25, "mild": 0.50, "moderate": 0.20, "severe": 0.05},
                    confidence=0.62,
                )
            },
            scores={
                "severity_score": _MockScoreAnswer(
                    score=2.05,
                    probabilities=[0.20, 0.55, 0.20, 0.05],
                    confidence=0.64,
                )
            },
        ),
        _MockSystemOneResponse(
            choices={
                "severity_choice": _MockChoiceAnswer(
                    choice="moderate",
                    probabilities={"minimal": 0.05, "mild": 0.20, "moderate": 0.60, "severe": 0.15},
                    confidence=0.71,
                )
            },
            scores={
                "severity_score": _MockScoreAnswer(
                    score=2.85,
                    probabilities=[0.05, 0.20, 0.60, 0.15],
                    confidence=0.71,
                )
            },
        ),
        _MockSystemOneResponse(
            choices={
                "severity_choice": _MockChoiceAnswer(
                    choice="severe",
                    probabilities={"minimal": 0.01, "mild": 0.04, "moderate": 0.15, "severe": 0.80},
                    confidence=0.88,
                )
            },
            scores={
                "severity_score": _MockScoreAnswer(
                    score=3.74,
                    probabilities=[0.01, 0.04, 0.15, 0.80],
                    confidence=0.88,
                )
            },
        ),
    ]

    parsed = [parse_system_one_response(r, row_idx=i) for i, r in enumerate(mock_resps)]
    y_true = np.array([1, 2, 3, 4], dtype=int)
    y_choice = np.array([p["pred_choice"] for p in parsed], dtype=int)
    y_score = np.array([p["pred_score_round"] for p in parsed], dtype=int)

    assert set(y_choice).issubset(set(LEVELS)), f"out-of-domain choice labels: {y_choice}"
    assert set(y_score).issubset(set(LEVELS)), f"out-of-domain score labels: {y_score}"

    rep = full_report(y_true, y_choice)
    assert np.isclose(rep["macro_f1"], 1.0)
    assert rep["direction"]["downgrade"] == 0
    print("  response parser & metrics: OK (domain labels 1..4 verified)")
    return True


def run_live_checks() -> bool:
    print("\n=== 2. Live API Connectivity Check (1 BN val + 1 EN val row) ===")
    key = load_env_key("TYPESAFE_API_KEY")
    if not key:
        print("  FAIL: TYPESAFE_API_KEY not found in environment or .env")
        return False

    for lang in ("bn", "en"):
        val_df = load_split(lang, "val")
        sample_text = str(val_df.iloc[0]["text"])
        gold_label = int(val_df.iloc[0]["label"])
        res = predict_split([sample_text], lang=lang, cache_path=None, concurrency=1)[0]
        print(
            f"  [{lang.upper()}] gold={gold_label} ({LEVEL_NAMES[gold_label]}) | "
            f"pred_choice={res['pred_choice']} (conf={res['confidence_choice']:.3f}, "
            f"probs={[round(x, 3) for x in res['probs_choice']]}) | "
            f"pred_score={res['pred_score_round']} (E[Y]={res['expected_score']:.3f}, "
            f"raw_score={res['raw_sdk_score']:.3f}, "
            f"probs={[round(x, 3) for x in res['probs_score']]})"
        )
    print("  live API check          : OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send 1 Bengali and 1 English validation example to the live TypeSafe API",
    )
    args = parser.parse_args()

    ok = run_offline_checks()
    if args.live and ok:
        ok = run_live_checks()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
