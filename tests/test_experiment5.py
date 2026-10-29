from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from run_llm import run_experiment
from scripts.analyze_translation_drift import analyze
from src.llm import build_messages, few_shot_examples, parse_level


class FakeScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, messages):
        self.calls += 1
        return {"pred": (self.calls % 4) + 1, "probs": None, "digit_mass": None, "raw": str((self.calls % 4) + 1)}


def test_run_forwards_backend_options(monkeypatch, tmp_path):
    import run_llm

    captured = {}

    class FakeHF:
        def __init__(self, model, load_4bit):
            captured["hf"] = (model, load_4bit)

        def score(self, messages):
            return {"pred": 1, "probs": None, "digit_mass": None, "raw": "1"}

    monkeypatch.setattr(run_llm, "HFScorer", FakeHF)
    monkeypatch.setattr(run_llm, "load_condition", lambda condition, translated_dir: (["x"], np.array([1]), pd.DataFrame({"text": ["x"], "label": [1]})))
    run_experiment("bn", "model", limit=1, bootstrap=1, out_root=tmp_path, scorer=None, load_4bit=False)
    assert captured["hf"] == ("model", False)


def test_parse_level_rejects_hedges_and_handles_names():
    assert parse_level("4") == 4
    assert parse_level("The answer is severe.") == 4
    assert parse_level("1 or 2") is None
    assert parse_level("I cannot answer") is None


def test_few_shot_examples_are_balanced_and_seeded():
    pool = pd.DataFrame({"text": [f"{level}-{i}" for level in range(1, 5) for i in range(5)], "label": [level for level in range(1, 5) for _ in range(5)]})
    first = few_shot_examples(pool, 2, 42)
    second = few_shot_examples(pool, 2, 42)
    assert first == second
    assert {level: [label for _, label in first].count(level) for level in range(1, 5)} == {1: 2, 2: 2, 3: 2, 4: 2}


def test_build_messages_preserves_roles_and_clips():
    messages = build_messages("query", [("example", 2)])
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "Post:\nquery"


def _write_splits(root: Path) -> None:
    (root / "data" / "splits").mkdir(parents=True)
    train = pd.DataFrame({"text": [f"train-{level}-{i}" for level in range(1, 5) for i in range(2)], "label": [level for level in range(1, 5) for _ in range(2)]})
    test = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 2, 3]})
    train.to_csv(root / "data" / "splits" / "bn_train.csv", index=False)
    test.to_csv(root / "data" / "splits" / "bn_test.csv", index=False)


def test_resume_cache_and_output_shapes(tmp_path, monkeypatch):
    _write_splits(tmp_path)
    translated = tmp_path / "translated"
    translated.mkdir()
    pd.DataFrame({"text_bn": ["a", "b", "c"], "text_en": ["A", "B", "C"], "label": [1, 2, 3]}).to_csv(translated / "bn_test_en.csv", index=False)
    pd.DataFrame({"text_bn": ["train-1-0", "train-2-0", "train-3-0", "train-4-0"], "text_en": ["one", "two", "three", "four"], "label": [1, 2, 3, 4]}).to_csv(translated / "bn_pool_en.csv", index=False)
    monkeypatch.chdir(tmp_path)
    scorer = FakeScorer()
    out = run_experiment("bn2en", "fake/model", limit=3, bootstrap=2, out_root=tmp_path / "results", scorer=scorer, translated_dir=translated)
    assert scorer.calls == 3
    first = json.loads((out / "eval_test.json").read_text())
    assert first["n_total"] == 3 and first["n_failed"] == 0
    run_experiment("bn2en", "fake/model", limit=3, bootstrap=2, out_root=tmp_path / "results", scorer=scorer, translated_dir=translated)
    assert scorer.calls == 3
    predictions = pd.read_csv(out / "predictions.csv")
    assert list(predictions.columns) == ["i", "gold", "pred", "probs", "digit_mass", "raw"]


def test_alignment_guard(tmp_path, monkeypatch):
    _write_splits(tmp_path)
    translated = tmp_path / "translated"
    translated.mkdir()
    pd.DataFrame({"text_bn": ["wrong", "b", "c"], "text_en": ["A", "B", "C"], "label": [1, 2, 3]}).to_csv(translated / "bn_test_en.csv", index=False)
    pd.DataFrame({"text_bn": ["x"] * 4, "text_en": ["x"] * 4, "label": [1, 2, 3, 4]}).to_csv(translated / "bn_pool_en.csv", index=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="aligned"):
        run_experiment("bn2en", "fake/model", scorer=FakeScorer(), translated_dir=translated)


def test_translation_drift_report(tmp_path):
    bn = tmp_path / "bn.csv"
    bn2en = tmp_path / "bn2en.csv"
    pd.DataFrame({"i": [0, 1, 2, 3], "gold": [1, 2, 3, 4], "pred": [1, 2, 3, 4]}).to_csv(bn, index=False)
    pd.DataFrame({"i": [0, 1, 2, 3], "gold": [1, 2, 3, 4], "pred": [2, 1, 3, 2]}).to_csv(bn2en, index=False)
    report = analyze(bn, bn2en)
    assert report["translation_upgraded"] == 1
    assert report["translation_downgraded"] == 2
    assert len(report["transition_matrix_rows_bn_columns_bn2en"]) == 4