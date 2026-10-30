"""Build translated English splits matched by source row to the original splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.schema import ENGLISH_PATH, load_english, make_splits, save_splits


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(
    translations_path: Path,
    source_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    source = load_english(source_path)
    source.insert(0, "source_row", range(len(source)))
    translations = pd.read_csv(translations_path, keep_default_na=False)
    required = {"source_row", "source_text", "source_sha256", "text", "label"}
    missing = required - set(translations.columns)
    if missing:
        raise ValueError(f"Missing translation columns: {sorted(missing)}")
    ids = pd.to_numeric(translations["source_row"], errors="raise")
    if ids.isna().any() or (ids % 1 != 0).any():
        raise ValueError("Translation source_row values must be integers")
    translations["source_row"] = ids.astype(int)
    if translations["source_row"].duplicated().any():
        raise ValueError("Duplicate translation source_row values")
    if set(translations["source_row"]) != set(source["source_row"]):
        raise ValueError("Translation rows do not match the complete English source")
    translations = translations.set_index("source_row").loc[source["source_row"]]
    if translations["source_text"].tolist() != source["text"].tolist():
        raise ValueError("Translation source texts do not match the English source")
    hashes = [sha256(text.encode("utf-8")) for text in source["text"]]
    if translations["source_sha256"].tolist() != hashes:
        raise ValueError("Translation source hashes do not match the English source")
    labels = pd.to_numeric(translations["label"], errors="raise")
    if labels.tolist() != source["label"].tolist():
        raise ValueError("Translation labels do not match the English source")
    if any(not isinstance(text, str) or not text.strip() for text in translations["text"]):
        raise ValueError("Missing or empty translated text")

    partitions = make_splits(source, seed=seed)
    translated_parts = []
    for part in partitions:
        selected = translations.loc[part["source_row"]]
        translated_parts.append(pd.DataFrame({
            "text": selected["text"].tolist(),
            "label": part["label"].tolist(),
            "source_row": part["source_row"].tolist(),
        }))
    paths = save_splits("en", *translated_parts, out_dir=out_dir)
    manifest = {
        "source_file_sha256": sha256(source_path.read_bytes()),
        "translation_file_sha256": sha256(translations_path.read_bytes()),
        "split_seed": seed,
        "counts": {name: len(part) for name, part in zip(("train", "val", "test"), translated_parts)},
        "split_file_sha256": {name: sha256(path.read_bytes()) for name, path in paths.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations", type=Path, default=Path("data/translated/en_to_bn.csv"))
    parser.add_argument("--source", type=Path, default=ENGLISH_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits/en_to_bn"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.translations, args.source, args.out_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
