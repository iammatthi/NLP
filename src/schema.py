"""Unified ordinal severity schema and dataset loading.

Levels are ordered 1 (minimal) -> 4 (severe). Both source datasets are
mapped onto this scale:

    Kabir et al. 2022 (Bengali, `labels` int 1-4)      -> used as-is
    Naseem et al. 2022 (English, `label` str)          -> minimum/mild/moderate/severe
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LEVELS = (1, 2, 3, 4)
LEVEL_NAMES = {1: "minimal", 2: "mild", 3: "moderate", 4: "severe"}
NAME_TO_LEVEL = {v: k for k, v in LEVEL_NAMES.items()}
# Naseem et al. 2022 labels the first tier "minimum", not "minimal".
NAME_TO_LEVEL["minimum"] = 1

# The classifier head is 0-indexed (num_labels=4 -> classes 0..3) while the
# domain scale is 1..4. Convert at the model boundary only; metrics and
# reports stay in domain space.
NUM_LABELS = len(LEVELS)


def to_model_label(level: int | np.ndarray) -> int | np.ndarray:
    """Domain level 1..4 -> model class 0..3."""
    return np.asarray(level) - 1 if isinstance(level, np.ndarray) else int(level) - 1


def to_domain_label(index: int | np.ndarray) -> int | np.ndarray:
    """Model class 0..3 -> domain level 1..4."""
    return np.asarray(index) + 1 if isinstance(index, np.ndarray) else int(index) + 1

BENGALI_PATH = Path("data/raw/depression-severity-bengali.xlsx")
ENGLISH_PATH = Path("data/raw/depression-severity-english.csv")
SPLITS_DIR = Path("data/splits")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def load_bengali(path: Path | str = BENGALI_PATH) -> pd.DataFrame:
    """Load Kabir et al. 2022 and normalise to `text` / `label` (int 1-4)."""
    df = pd.read_excel(path)
    df = df.rename(columns={"posts": TEXT_COLUMN, "labels": LABEL_COLUMN})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    _validate(df)
    return df[[TEXT_COLUMN, LABEL_COLUMN]].reset_index(drop=True)


def load_english(path: Path | str = ENGLISH_PATH) -> pd.DataFrame:
    """Load Naseem et al. 2022 and map string labels onto the 1-4 scale."""
    df = pd.read_csv(path)
    df = df.rename(columns={"text": TEXT_COLUMN, "label": LABEL_COLUMN})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    raw = df[LABEL_COLUMN].astype(str).str.strip()
    unmapped = sorted(set(raw) - set(NAME_TO_LEVEL))
    if unmapped:
        raise ValueError(
            f"unmapped English label values {unmapped}; "
            f"known: {sorted(NAME_TO_LEVEL)}"
        )
    df[LABEL_COLUMN] = raw.map(NAME_TO_LEVEL).astype(int)
    _validate(df)
    return df[[TEXT_COLUMN, LABEL_COLUMN]].reset_index(drop=True)


def load(lang: str) -> pd.DataFrame:
    if lang == "bn":
        return load_bengali()
    if lang == "en":
        return load_english()
    raise ValueError(f"unknown language {lang!r}, expected 'bn' or 'en'")


def _validate(df: pd.DataFrame) -> None:
    bad = set(df[LABEL_COLUMN].unique()) - set(LEVELS)
    if bad:
        raise ValueError(f"unexpected label values: {sorted(bad)}")
    if df[TEXT_COLUMN].isna().any() or (df[TEXT_COLUMN] == "").any():
        raise ValueError("empty text rows present")


def make_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split. Remainder goes to test."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("need 0 < train < 1, 0 <= val < 1, train + val < 1")
    test = 1.0 - train - val
    train_df, temp_df = train_test_split(
        df, train_size=train, random_state=seed, stratify=df[LABEL_COLUMN]
    )
    if val == 0:
        return (
            train_df.reset_index(drop=True),
            train_df.iloc[0:0].copy(),
            temp_df.reset_index(drop=True),
        )
    relative_val = val / (val + test)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=seed,
        stratify=temp_df[LABEL_COLUMN],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def class_weights(labels: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1. Order matches LEVELS."""
    counts = np.array(
        [np.sum(np.asarray(labels) == level) for level in LEVELS], dtype=np.float64
    )
    if (counts == 0).any():
        raise ValueError(f"missing class in labels: counts={counts}")
    weights = counts.sum() / (len(LEVELS) * counts)
    return weights / weights.mean()


def save_splits(
    lang: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path | str = SPLITS_DIR,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        p = out / f"{lang}_{name}.csv"
        frame.to_csv(p, index=False)
        paths[name] = p
    return paths


def load_split(lang: str, name: str, out_dir: Path | str = SPLITS_DIR) -> pd.DataFrame:
    path = Path(out_dir) / f"{lang}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/prepare_splits.py` first"
        )
    return pd.read_csv(path)
