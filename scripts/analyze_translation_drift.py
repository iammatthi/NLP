"""Compare paired bn and bn2en prediction files for one run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.schema import LEVELS, LEVEL_NAMES


def _mcnemar(table: np.ndarray) -> dict[str, float | int]:
    discordant = int(table[0, 1] + table[1, 0])
    if discordant == 0:
        return {"b01": int(table[0, 1]), "b10": int(table[1, 0]), "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(int(table[0, 1] - table[1, 0])) - 1) ** 2 / discordant
    # Exact two-sided binomial test, avoiding a statsmodels dependency.
    smaller = min(int(table[0, 1]), int(table[1, 0]))
    p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2**discordant))
    return {"b01": int(table[0, 1]), "b10": int(table[1, 0]), "chi2": float(chi2), "p_value": float(p_value)}


def analyze(bn_path: Path, bn2en_path: Path) -> dict:
    native = pd.read_csv(bn_path)
    translated = pd.read_csv(bn2en_path)
    if len(native) != len(translated):
        raise ValueError("paired prediction files have different lengths")
    required = {"i", "gold", "pred"}
    if not required.issubset(native) or not required.issubset(translated):
        raise ValueError(f"prediction files must contain {sorted(required)}")
    if not np.array_equal(native["i"].to_numpy(), translated["i"].to_numpy()):
        raise ValueError("paired prediction files have different row IDs")
    if not np.array_equal(native["gold"].to_numpy(), translated["gold"].to_numpy()):
        raise ValueError("paired prediction files have different gold labels")
    native_pred = native["pred"].to_numpy()
    translated_pred = translated["pred"].to_numpy()
    valid = pd.notna(native_pred) & pd.notna(translated_pred)
    gold = native.loc[valid, "gold"].to_numpy(dtype=int)
    before = native.loc[valid, "pred"].to_numpy(dtype=int)
    after = translated.loc[valid, "pred"].to_numpy(dtype=int)
    matrix = [[int(((before == old) & (after == new)).sum()) for new in LEVELS] for old in LEVELS]
    correctness = np.array([[int(((before == gold) & (after == gold)).sum()), int(((before == gold) & (after != gold)).sum())], [int(((before != gold) & (after == gold)).sum()), int(((before != gold) & (after != gold)).sum())]])
    per_class = {}
    for level in LEVELS:
        mask = gold == level
        per_class[LEVEL_NAMES[level]] = {
            "n": int(mask.sum()),
            "downgraded": int(((after < before) & mask).sum()),
            "downgrade_rate": float(((after < before) & mask).sum() / mask.sum()) if mask.any() else 0.0,
        }
    return {
        "n_paired": int(valid.sum()),
        "n_failed_or_missing": int((~valid).sum()),
        "transition_matrix_rows_bn_columns_bn2en": matrix,
        "translation_downgraded": int((after < before).sum()),
        "translation_upgraded": int((after > before).sum()),
        "mcnemar_correctness": _mcnemar(correctness),
        "downgrade_rate_by_gold_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bn", type=Path, required=True)
    parser.add_argument("--bn2en", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.bn, args.bn2en)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()