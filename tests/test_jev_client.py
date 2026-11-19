"""Unit tests for Jev probability parsing (src/jev_client.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.jev_client import (
    _extract_choice_probs,
    _extract_score_probs,
    build_questions,
    parse_system_one_response,
)


def test_choice_probs_normalised_from_dict():
    probs = _extract_choice_probs(
        {"minimal": 1.0, "mild": 3.0, "moderate": 0.0, "severe": 0.0}
    )
    assert probs == pytest.approx([0.25, 0.75, 0.0, 0.0])


def test_choice_probs_accept_list_and_fall_back_to_uniform():
    assert _extract_choice_probs([1, 1, 1, 1]) == pytest.approx([0.25] * 4)
    assert _extract_choice_probs(None) == pytest.approx([0.25] * 4)
    assert _extract_choice_probs([0, 0, 0, 0]) == pytest.approx([0.25] * 4)


def test_score_probs_int_and_str_keys_agree():
    by_int = _extract_score_probs({0: 1.0, 1: 2.0, 2: 1.0, 3: 4.0})
    by_str = _extract_score_probs({"0": 1.0, "1": 2.0, "2": 1.0, "3": 4.0})
    assert by_int == by_str == pytest.approx([0.125, 0.25, 0.125, 0.5])


def test_score_probs_name_and_list_forms():
    by_name = _extract_score_probs(
        {"minimal": 1.0, "mild": 1.0, "moderate": 1.0, "severe": 1.0}
    )
    assert by_name == pytest.approx([0.25] * 4)
    assert _extract_score_probs([2.0, 2.0, 2.0, 2.0]) == pytest.approx([0.25] * 4)


def _response(choice_key="severe", choice_probs=None, score_probs=None, score=2.6):
    return SimpleNamespace(
        choices={
            "severity_choice": SimpleNamespace(
                choice=choice_key,
                probabilities=choice_probs,
                confidence=0.99,
            )
        },
        scores={
            "severity_score": SimpleNamespace(
                probabilities=score_probs,
                score=score,
                confidence=0.9,
            )
        },
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="jev-test",
    )


def test_parse_shifts_sdk_0_to_3_into_domain_1_to_4():
    parsed = parse_system_one_response(
        _response(
            choice_probs={"minimal": 0.1, "mild": 0.2, "moderate": 0.3, "severe": 0.4},
            score_probs={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        )
    )
    assert parsed["pred_choice"] == 4
    # E[Y] = 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 3.0 -> rounds to 3.
    assert parsed["expected_score"] == pytest.approx(3.0)
    assert parsed["pred_score_round"] == 3
    assert parsed["pred_score_argmax"] == 4
    assert parsed["probs_score"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert parsed["confidence_choice"] == pytest.approx(0.99)
    assert parsed["input_tokens"] == 10 and parsed["output_tokens"] == 5


def test_choice_falls_back_to_argmax_when_key_unknown():
    parsed = parse_system_one_response(
        _response(
            choice_key="unsure",
            choice_probs={"minimal": 0.1, "mild": 0.6, "moderate": 0.2, "severe": 0.1},
            score_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
            score=0.0,
        )
    )
    assert parsed["pred_choice"] == 2
    assert parsed["pred_score_argmax"] == 1
    assert parsed["expected_score"] == pytest.approx(1.0)


def test_build_questions_unknown_rubric_raises():
    with pytest.raises(ValueError, match="Unknown rubric"):
        build_questions("nope")


def test_build_questions_returns_choice_and_score():
    questions = build_questions("unified")
    assert set(questions) == {"severity_choice", "severity_score"}
