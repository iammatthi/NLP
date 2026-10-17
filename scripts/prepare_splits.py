"""Build and persist the stratified train/val/test splits.

Writes `data/splits/{lang}_{train,val,test}.csv` so every downstream run
trains and evaluates on identical data.

Example (experiment 4 - Bengali splits, the ones 4a and 4b use):

    uv run python scripts/prepare_splits.py --lang bn --seed 42

Example (English splits, needed later for configs 1-3):

    uv run python scripts/prepare_splits.py --lang en --seed 42

Example (non-default split ratios):

    uv run python scripts/prepare_splits.py --lang bn --seed 42 --train-size 0.7 --val-size 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the project (e.g. `python scripts/x.py` on
# Colab), where the repo root is not otherwise on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.schema import LEVELS, LEVEL_NAMES, load, make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("bn", "en"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    frame = load(args.lang)
    train, val, test = make_splits(
        frame, seed=args.seed, train=args.train_size, val=args.val_size
    )
    paths = save_splits(args.lang, train, val, test)

    print(f"lang={args.lang}  n={len(frame)}  seed={args.seed}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        counts = part["label"].value_counts().reindex(LEVELS, fill_value=0)
        dist = ", ".join(
            f"{LEVEL_NAMES[lvl]}={counts[lvl]}" for lvl in LEVELS
        )
        print(f"  {name:5s} {len(part):5d}  ({dist})")
    print("written:")
    for name, path in paths.items():
        print(f"  {name:5s} {path}")


if __name__ == "__main__":
    main()
