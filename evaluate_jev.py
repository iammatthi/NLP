"""Evaluate TypeSafe AI's Jev ('System One') model zero-shot on a held-out split.

Queries both the categorical `Choice` primitive and the ordinal `Score` rubric
primitive in a single forward pass per example, then writes the full ordinal
metric suite matching `evaluate.py`:
- macro-F1, per-class F1, MAE, QWK, accuracy
- upgrade / ok / downgrade error-direction matrix (`downgrade = pred < gold`)
- bootstrap 95% CIs (`macro_f1`, `downgrade_rate`, `qwk`)
- Naseem et al. (2022) directional Graded Precision (`GP`), Graded Recall (`GR`),
  and `FScore`
- Expected Calibration Error (`ECE`) and mean model confidence

Supports two rubrics via `--rubric`:
- `unified` (default): `6a_jev_choice_{lang}` and `6b_jev_score_{lang}`
- `naseem_dda`: `6c_jev_choice_naseem_dda_{lang}` and `6c_jev_score_naseem_dda_{lang}`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from evaluate import bootstrap_cis
from src.jev_client import RUBRIC_CONFIGS, predict_split
from src.metrics import full_report
from src.schema import LEVELS, LEVEL_NAMES, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("bn", "en"), default="bn")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument(
        "--rubric",
        choices=tuple(RUBRIC_CONFIGS.keys()),
        default="unified",
        help="Rubric specification: 'unified' (6a/6b DSM-5) or 'naseem_dda' (6c Reddit DDA ablation)",
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "experiment6",
        help="Directory where tracked eval_{split}.json reports are written",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "experiment6",
        help="Gitignored directory for raw JSONL response cache and .npz arrays",
    )
    return parser.parse_args()


def compute_graded_metrics(direction: dict[str, Any]) -> dict[str, float]:
    """Compute Naseem et al. (2022, Eq. 5) ordinal Graded Precision, Recall, and FScore."""
    n_ok = float(direction["ok"])
    n_up = float(direction["upgrade"])
    n_down = float(direction["downgrade"])
    gp = n_ok / (n_ok + n_up) if (n_ok + n_up) > 0 else 0.0
    gr = n_ok / (n_ok + n_down) if (n_ok + n_down) > 0 else 0.0
    fscore = (2.0 * gp * gr / (gp + gr)) if (gp + gr) > 0 else 0.0
    return {
        "graded_precision": float(gp),
        "graded_recall": float(gr),
        "graded_fscore": float(fscore),
    }


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute 10-bin Expected Calibration Error (ECE) over top-predicted class probability."""
    confidences = np.max(probs, axis=1)
    accuracies = (y_true == y_pred).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b_idx in range(n_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        if np.any(mask):
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def build_variant_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    confidences: np.ndarray,
    variant_name: str,
    lang: str,
    split: str,
    n_boot: int,
    seed: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble full evaluation JSON report matching `evaluate.py` + calibration extras."""
    report = full_report(y_true, y_pred)
    report["n"] = int(len(y_true))
    report["lang"] = lang
    report["split"] = split
    report["model_dir"] = variant_name
    report["ci95"] = bootstrap_cis(y_true, y_pred, n_boot, seed)
    report["graded"] = compute_graded_metrics(report["direction"])
    report["calibration"] = {
        "ece_10bin": compute_ece(y_true, y_pred, probs),
        "mean_max_prob": float(np.mean(np.max(probs, axis=1))),
        "mean_sdk_confidence": float(np.mean(confidences)),
        "predicted_class_counts": {
            LEVEL_NAMES[lvl]: int(np.sum(y_pred == lvl)) for lvl in LEVELS
        },
        "true_class_counts": {
            LEVEL_NAMES[lvl]: int(np.sum(y_true == lvl)) for lvl in LEVELS
        },
    }
    if extra_fields:
        report.update(extra_fields)
    return report


def print_summary(title: str, report: dict[str, Any], out_path: Path) -> None:
    d = report["direction"]
    g = report["graded"]
    c = report["calibration"]
    print(f"\n=== {title} ===")
    print(
        f"macro-F1 : {report['macro_f1']:.4f}  "
        f"(95% CI {report['ci95']['macro_f1'][0]:.4f}-{report['ci95']['macro_f1'][1]:.4f})"
    )
    print(
        f"QWK      : {report['qwk']:.4f}  "
        f"(95% CI {report['ci95']['qwk'][0]:.4f}-{report['ci95']['qwk'][1]:.4f})"
    )
    print(f"MAE      : {report['mae']:.4f}")
    print(f"accuracy : {report['accuracy']:.4f}")
    print(
        f"Graded   : FScore={g['graded_fscore']:.4f}  "
        f"(GP={g['graded_precision']:.4f}, GR={g['graded_recall']:.4f})"
    )
    print(
        f"Calib    : ECE={c['ece_10bin']:.4f}  "
        f"mean_max_prob={c['mean_max_prob']:.4f}  "
        f"mean_sdk_conf={c['mean_sdk_confidence']:.4f}"
    )
    print("per-class F1 (predicted count / true count):")
    for level in LEVELS:
        name = LEVEL_NAMES[level]
        p_cnt = c["predicted_class_counts"][name]
        t_cnt = c["true_class_counts"][name]
        print(f"  {name:9s} {report[name]:.4f}  ({p_cnt:3d} / {t_cnt:3d})")
    print("error direction:")
    print(f"  ok        {d['ok']:5d}  ({1 - d['error_rate']:.4f})")
    print(
        f"  downgrade {d['downgrade']:5d}  ({d['downgrade_rate']:.4f})  "
        f"[share of errors: {d['downgrade_share_of_errors']:.4f}]"
    )
    print(f"  upgrade   {d['upgrade']:5d}  ({d['upgrade_rate']:.4f})")
    print(f"  signed    {d['mean_signed_error']:+.4f}")
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    frame = load_split(args.lang, args.split)
    texts = frame["text"].tolist()
    y_true = frame["label"].to_numpy(dtype=int)

    suffix = "" if args.rubric == "unified" else f"_{args.rubric}"
    cache_path = args.cache_dir / f"cache_{args.lang}_{args.split}{suffix}.jsonl"
    print(
        f"Evaluating Jev ('System One') zero-shot: lang={args.lang}  "
        f"split={args.split}  rubric={args.rubric}  n={len(frame)}"
    )

    records = predict_split(
        texts=texts,
        lang=args.lang,
        cache_path=cache_path,
        concurrency=args.concurrency,
        rubric=args.rubric,
    )

    y_pred_choice = np.array([r["pred_choice"] for r in records], dtype=int)
    probs_choice = np.array([r["probs_choice"] for r in records], dtype=float)
    conf_choice = np.array([r["confidence_choice"] for r in records], dtype=float)

    y_pred_score_round = np.array([r["pred_score_round"] for r in records], dtype=int)
    y_pred_score_argmax = np.array([r["pred_score_argmax"] for r in records], dtype=int)
    expected_score = np.array([r["expected_score"] for r in records], dtype=float)
    probs_score = np.array([r["probs_score"] for r in records], dtype=float)
    conf_score = np.array([r["confidence_score"] for r in records], dtype=float)

    # Save full numeric arrays to gitignored outputs/experiment6/ for downstream analysis
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.cache_dir / f"predictions_{args.lang}_{args.split}{suffix}.npz"
    np.savez(
        npz_path,
        y_true=y_true,
        y_pred_choice=y_pred_choice,
        probs_choice=probs_choice,
        conf_choice=conf_choice,
        y_pred_score_round=y_pred_score_round,
        y_pred_score_argmax=y_pred_score_argmax,
        expected_score=expected_score,
        probs_score=probs_score,
        conf_score=conf_score,
    )

    if args.rubric == "unified":
        choice_tag = f"6a_jev_choice_{args.lang}"
        score_tag = f"6b_jev_score_{args.lang}"
    else:
        choice_tag = f"6c_jev_choice_{args.rubric}_{args.lang}"
        score_tag = f"6c_jev_score_{args.rubric}_{args.lang}"

    # 1. Choice Primitive (Categorical)
    choice_dir = args.results_dir / choice_tag
    choice_dir.mkdir(parents=True, exist_ok=True)
    choice_out = choice_dir / f"eval_{args.split}.json"
    choice_report = build_variant_report(
        y_true=y_true,
        y_pred=y_pred_choice,
        probs=probs_choice,
        confidences=conf_choice,
        variant_name=choice_tag,
        lang=args.lang,
        split=args.split,
        n_boot=args.bootstrap,
        seed=args.seed,
        extra_fields={"primitive": "Choice", "rubric": args.rubric},
    )
    choice_out.write_text(json.dumps(choice_report, indent=2), encoding="utf-8")
    print_summary(f"{choice_tag} ({args.lang}-{args.split})", choice_report, choice_out)

    # 2. Score Primitive (Ordinal Expected Rubric Position)
    score_dir = args.results_dir / score_tag
    score_dir.mkdir(parents=True, exist_ok=True)
    score_out = score_dir / f"eval_{args.split}.json"
    score_argmax_rep = full_report(y_true, y_pred_score_argmax)
    score_report = build_variant_report(
        y_true=y_true,
        y_pred=y_pred_score_round,
        probs=probs_score,
        confidences=conf_score,
        variant_name=score_tag,
        lang=args.lang,
        split=args.split,
        n_boot=args.bootstrap,
        seed=args.seed,
        extra_fields={
            "primitive": "Score (rounded expected value E[Y])",
            "rubric": args.rubric,
            "continuous_mae": float(np.mean(np.abs(expected_score - y_true))),
            "score_argmax_summary": {
                "macro_f1": score_argmax_rep["macro_f1"],
                "qwk": score_argmax_rep["qwk"],
                "mae": score_argmax_rep["mae"],
                "accuracy": score_argmax_rep["accuracy"],
                "downgrade_rate": score_argmax_rep["direction"]["downgrade_rate"],
                "upgrade_rate": score_argmax_rep["direction"]["upgrade_rate"],
                "per_class_f1": {
                    LEVEL_NAMES[lvl]: score_argmax_rep[LEVEL_NAMES[lvl]]
                    for lvl in LEVELS
                },
            },
        },
    )
    score_out.write_text(json.dumps(score_report, indent=2), encoding="utf-8")
    print_summary(f"{score_tag} ({args.lang}-{args.split})", score_report, score_out)


if __name__ == "__main__":
    main()
