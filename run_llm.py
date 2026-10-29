"""Run one zero-/few-shot Experiment 5 condition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from evaluate import bootstrap_cis
from src.llm import APIScorer, HFScorer, build_messages, few_shot_examples
from src.metrics import full_report
from src.schema import load_split

TRANSLATED_DIR = Path("data/translated")
RESULTS_ROOT = Path("results/experiment5_llm")


class Scorer(Protocol):
    def score(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("bn", "bn2en", "en"), required=True)
    parser.add_argument("--backend", choices=("hf", "api"), default="hf")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    return parser.parse_args()


def load_condition(condition: str, translated_dir: Path = TRANSLATED_DIR) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Load texts, aligned gold labels, and train-only few-shot pool."""
    if condition == "en":
        test = load_split("en", "test")
        return test["text"].astype(str).tolist(), test["label"].to_numpy(dtype=int), load_split("en", "train")

    bengali_test = load_split("bn", "test")
    gold = bengali_test["label"].to_numpy(dtype=int)
    pool_file = translated_dir / "bn_pool_en.csv"
    pool = pd.read_csv(pool_file)
    if condition == "bn":
        return (
            bengali_test["text"].astype(str).tolist(),
            gold,
            pool[["text_bn", "label"]].rename(columns={"text_bn": "text"}),
        )

    translated = pd.read_csv(translated_dir / "bn_test_en.csv")
    if len(translated) != len(bengali_test):
        raise ValueError("translated test file length differs from bn_test.csv")
    text_aligned = translated["text_bn"].astype(str).str.strip().to_numpy() == bengali_test["text"].astype(str).str.strip().to_numpy()
    labels_aligned = translated["label"].to_numpy(dtype=int) == gold
    if not bool(text_aligned.all() and labels_aligned.all()):
        raise ValueError("translated test file is not aligned with data/splits/bn_test.csv")
    return translated["text_en"].astype(str).tolist(), gold, pool[["text_en", "label"]].rename(columns={"text_en": "text"})


def write_eval(path: Path, y_true: np.ndarray, predictions: list[int | None], meta: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    answered = np.array([prediction is not None for prediction in predictions])
    observed_true = y_true[answered]
    observed_pred = np.asarray([prediction for prediction in predictions if prediction is not None], dtype=int)
    report = full_report(observed_true, observed_pred)
    report.update(meta)
    report["n"] = int(answered.sum())
    report["n_total"] = int(len(y_true))
    report["n_failed"] = int((~answered).sum())
    report["coverage"] = float(answered.mean()) if len(answered) else 0.0
    report["ci95"] = bootstrap_cis(observed_true, observed_pred, n_boot, seed)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_cache(path: Path) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[int(row["i"])] = row
    return cache


def run_experiment(
    condition: str,
    model: str,
    backend: str = "hf",
    shots: int = 0,
    seed: int = 42,
    limit: int | None = None,
    bootstrap: int = 1000,
    out_root: Path = RESULTS_ROOT,
    scorer: Scorer | None = None,
    translated_dir: Path = TRANSLATED_DIR,
    load_4bit: bool = True,
    base_url: str | None = None,
    sleep: float = 0.0,
) -> Path:
    texts, gold, pool = load_condition(condition, translated_dir)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        texts, gold = texts[:limit], gold[:limit]
    shots_data = few_shot_examples(pool, shots, seed) if shots else []
    model_tag = model.rsplit("/", 1)[-1].lower()
    run_name = f"{model_tag}_{condition}_{shots}shot_seed{seed}"
    if limit is not None:
        run_name += "_smoke"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if scorer is None:
        if backend == "hf":
            scorer = HFScorer(model, load_4bit=load_4bit)
        else:
            api_key = os.getenv("API_KEY")
            if not api_key:
                raise SystemExit("set API_KEY for --backend api")
            scorer = APIScorer(model, base_url, api_key, sleep=sleep)

    cache_path = out_dir / "raw.jsonl"
    cache = _load_cache(cache_path)
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            if index in cache:
                continue
            result = dict(scorer.score(build_messages(text, shots_data)))
            result["i"] = index
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            cache[index] = result

    rows = [cache[index] for index in range(len(texts))]
    predictions = [row.get("pred") for row in rows]
    meta = {
        "experiment": 5,
        "condition": condition,
        "backend": backend,
        "model": model,
        "shots_per_class": shots,
        "seed": seed,
        "lang": "en" if condition == "en" else "bn",
        "split": "test",
    }
    report = write_eval(out_dir / "eval_test.json", gold, predictions, {**meta, "mode": "argmax"}, bootstrap, seed)
    if rows and rows[0].get("probs") is not None:
        expected = [int(np.clip(np.floor(np.dot(row["probs"], [1, 2, 3, 4]) + 0.5), 1, 4)) for row in rows]
        write_eval(out_dir / "eval_test_expected.json", gold, expected, {**meta, "mode": "expected"}, bootstrap, seed)
    if report["n_failed"]:
        strict = [prediction if prediction is not None else (4 if label <= 2 else 1) for prediction, label in zip(predictions, gold)]
        write_eval(out_dir / "eval_test_strict.json", gold, strict, {**meta, "mode": "strict_failures_wrong"}, bootstrap, seed)
    pd.DataFrame({
        "i": range(len(rows)),
        "gold": gold,
        "pred": predictions,
        "probs": [json.dumps(row.get("probs")) if row.get("probs") is not None else "" for row in rows],
        "digit_mass": [row.get("digit_mass") for row in rows],
        "raw": [row.get("raw", "") for row in rows],
    }).to_csv(out_dir / "predictions.csv", index=False)
    return out_dir


def main() -> None:
    args = parse_args()
    run_experiment(
        condition=args.condition,
        model=args.model,
        backend=args.backend,
        shots=args.shots,
        seed=args.seed,
        limit=args.limit,
        bootstrap=args.bootstrap,
        out_root=args.out_root,
        load_4bit=not args.no_4bit,
        base_url=args.base_url,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()