import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_translated_splits import prepare, sha256
from src.schema import load_bengali, load_english, make_splits


class PrepareTranslatedSplitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_path = self.root / "source.csv"
        self.translations_path = self.root / "translated.csv"
        self.out = self.root / "splits"
        labels = ["minimum", "mild", "moderate", "severe"] * 10
        texts = ["duplicate" if idx < 2 else f"post {idx}" for idx in range(40)]
        pd.DataFrame({"text": texts, "label": labels}).to_csv(self.source_path, index=False)
        source = load_english(self.source_path)
        self.translations = pd.DataFrame({
            "source_row": range(len(source)),
            "source_text": source["text"],
            "source_sha256": [sha256(t.encode("utf-8")) for t in source["text"]],
            "text": [f"translation {idx}" for idx in range(len(source))],
            "label": source["label"],
        })

    def test_translated_splits_match_original_rows_and_labels(self):
        self.translations.sample(frac=1, random_state=7).to_csv(self.translations_path, index=False)
        manifest = prepare(self.translations_path, self.source_path, self.out, seed=42)
        source = load_english(self.source_path)
        source.insert(0, "source_row", range(len(source)))
        for name, part in zip(("train", "val", "test"), make_splits(source, seed=42)):
            actual = pd.read_csv(self.out / f"en_{name}.csv")
            self.assertEqual(actual["source_row"].tolist(), part["source_row"].tolist())
            self.assertEqual(actual["label"].tolist(), part["label"].tolist())
            self.assertEqual(actual["text"].tolist(), [f"translation {i}" for i in part["source_row"]])
            self.assertEqual(manifest["counts"][name], len(part))

    def test_invalid_source_or_missing_translation_is_rejected(self):
        for column, value, message in (
            ("source_text", "tampered", "source texts"),
            ("label", 4, "labels"),
            ("text", "", "empty translated"),
        ):
            with self.subTest(column=column):
                invalid = self.translations.copy()
                invalid.loc[0, column] = value
                invalid.to_csv(self.translations_path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.translations_path, self.source_path, self.out)

    def test_duplicate_source_rows_are_rejected(self):
        invalid = self.translations.copy()
        invalid.loc[1, "source_row"] = 0
        invalid.to_csv(self.translations_path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare(self.translations_path, self.source_path, self.out)

    def test_bengali_source_builds_translated_english_splits(self):
        source_path = self.root / "source.xlsx"
        pd.DataFrame({
            "posts": [f"post {idx}" for idx in range(40)],
            "labels": [1, 2, 3, 4] * 10,
        }).to_excel(source_path, index=False)
        source = load_bengali(source_path)
        translations = pd.DataFrame({
            "source_row": range(len(source)),
            "source_text": source["text"],
            "source_sha256": [sha256(t.encode("utf-8")) for t in source["text"]],
            "text": [f"translation {idx}" for idx in range(len(source))],
            "label": source["label"],
        })
        translations.sample(frac=1, random_state=7).to_csv(self.translations_path, index=False)
        manifest = prepare(self.translations_path, source_path, self.out, seed=42, lang="bn")
        self.assertEqual(manifest["source_language"], "bn")
        self.assertEqual(manifest["target_language"], "en")
        source.insert(0, "source_row", range(len(source)))
        for name, part in zip(("train", "val", "test"), make_splits(source, seed=42)):
            actual = pd.read_csv(self.out / f"bn_{name}.csv")
            self.assertEqual(actual["source_row"].tolist(), part["source_row"].tolist())
            self.assertEqual(actual["label"].tolist(), part["label"].tolist())
            self.assertEqual(actual["text"].tolist(), [f"translation {i}" for i in part["source_row"]])
            self.assertEqual(manifest["counts"][name], len(part))


if __name__ == "__main__":
    unittest.main()
