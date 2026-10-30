"""Evaluate a trained severity classifier on a held-out split.

Writes the full ordinal metric suite: macro-F1, per-class F1, MAE, QWK, and
the upgrade/ok/downgrade error-direction matrix (downgrade = pred < gold),
plus bootstrap 95% CIs for macro-F1, QWK and the downgrade rate.

`eval_test.json` is written *inside* `--model-dir`, so it travels with the
checkpoint it describes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Allow running without installing the project, including from a different cwd.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.metrics import (
    direction_matrix,
    full_report,
    macro_f1,
    qwk,
)
from src.schema import LEVELS, LEVEL_NAMES, load_split, to_domain_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lang", choices=("bn", "en"), default="bn")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="optional row-aligned test predictions CSV")
    return parser.parse_args()


def predict(
    model_dir: Path, texts: list[str], max_length: int, batch_size: int
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    # argmax yields model-space classes 0..3; reports live in domain space 1..4
    return to_domain_label(np.asarray(preds, dtype=int))


def bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mf1_samples = np.empty(n_boot)
    down_samples = np.empty(n_boot)
    qwk_samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mf1_samples[i] = macro_f1(y_true[idx], y_pred[idx])
        down_samples[i] = (y_pred[idx] < y_true[idx]).mean()
        qwk_samples[i] = qwk(y_true[idx], y_pred[idx])
    return {
        "macro_f1": [float(x) for x in np.percentile(mf1_samples, [2.5, 97.5])],
        "downgrade_rate": [float(x) for x in np.percentile(down_samples, [2.5, 97.5])],
        "qwk": [float(x) for x in np.percentile(qwk_samples, [2.5, 97.5])],
    }


def main() -> None:
    args = parse_args()
    frame = load_split(args.lang, args.split)
    texts = frame["text"].tolist()
    y_true = frame["label"].to_numpy(dtype=int)
    print(f"split={args.split}  n={len(frame)}  model={args.model_dir}")

    y_pred = predict(args.model_dir, texts, args.max_length, args.batch_size)

    if args.predictions_out is not None:
        import pandas as pd

        predictions = pd.DataFrame({
            "split_row": np.arange(len(frame)),
            "source_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
            "label": y_true,
            "prediction": y_pred,
        })
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_out, index=False)

    report = full_report(y_true, y_pred)
    report["n"] = int(len(frame))
    report["lang"] = args.lang
    report["split"] = args.split
    report["model_dir"] = str(args.model_dir)
    report["ci95"] = bootstrap_cis(y_true, y_pred, args.bootstrap, args.seed)

    out_path = args.out or (args.model_dir / f"eval_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    d = report["direction"]
    print(f"\nmacro-F1 : {report['macro_f1']:.4f}  (95% CI {report['ci95']['macro_f1'][0]:.4f}-{report['ci95']['macro_f1'][1]:.4f})")
    print(f"QWK      : {report['qwk']:.4f}")
    print(f"MAE      : {report['mae']:.4f}")
    print(f"accuracy : {report['accuracy']:.4f}")
    print("\nper-class F1:")
    for level in LEVELS:
        name = LEVEL_NAMES[level]
        print(f"  {name:9s} {report[name]:.4f}")
    print("\nerror direction:")
    print(f"  ok        {d['ok']:5d}  ({1 - d['error_rate']:.4f})")
    print(f"  downgrade {d['downgrade']:5d}  ({d['downgrade_rate']:.4f})  <- under-estimates severity")
    print(f"  upgrade   {d['upgrade']:5d}  ({d['upgrade_rate']:.4f})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
