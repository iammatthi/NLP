import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.translate_english import load_checkpoint, translate


class FakeTranslate:
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def translate_text(self, **request):
        if len(self.calls) == self.fail_after:
            raise RuntimeError("interrupted")
        self.calls.append(request)
        return {"TranslatedText": f"translation of {request['Text']}"}


class TranslationCheckpointTests(unittest.TestCase):
    def test_resume_preserves_rows_and_sends_only_source_text(self):
        frame = pd.DataFrame({"text": ["same", "same", "other"], "label": [1, 2, 4]})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.jsonl"
            first = FakeTranslate(fail_after=1)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                translate(frame, first, checkpoint, "us-east-1")
            second = FakeTranslate()
            result = translate(frame, second, checkpoint, "us-east-1")
            self.assertEqual(list(result), [0, 1, 2])
            self.assertEqual([r["Text"] for r in second.calls], ["same", "other"])
            self.assertTrue(all(set(r) == {"Text", "SourceLanguageCode", "TargetLanguageCode"} for r in second.calls))
            self.assertEqual(result[1]["label"], 2)

    def test_changed_source_or_region_cannot_resume(self):
        frame = pd.DataFrame({"text": ["source"], "label": [1]})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.jsonl"
            translate(frame, FakeTranslate(), checkpoint, "us-east-1")
            changed = pd.DataFrame({"text": ["changed"], "label": [1]})
            with self.assertRaisesRegex(ValueError, "source text has changed"):
                load_checkpoint(checkpoint, changed)
            with self.assertRaisesRegex(ValueError, "region differs"):
                translate(frame, FakeTranslate(), checkpoint, "us-west-2")


if __name__ == "__main__":
    unittest.main()
