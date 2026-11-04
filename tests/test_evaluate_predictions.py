import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate


class EvaluatePredictionsTests(unittest.TestCase):
    def test_optional_prediction_file_is_aligned_with_test_rows(self):
        frame = pd.DataFrame({"text": ["one", "two", "three", "four"], "label": [1, 2, 3, 4]})
        predictions = np.array([2, 2, 1, 4])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report.json"
            pred_path = root / "predictions.csv"
            argv = ["evaluate.py", "--model-dir", str(root), "--lang", "bn", "--split", "test",
                    "--bootstrap", "10", "--out", str(out), "--predictions-out", str(pred_path)]
            with patch("sys.argv", argv), patch.object(evaluate, "load_split", return_value=frame), \
                 patch.object(evaluate, "predict", return_value=predictions):
                evaluate.main()
            actual = pd.read_csv(pred_path)
            self.assertEqual(actual["split_row"].tolist(), [0, 1, 2, 3])
            self.assertEqual(actual["label"].tolist(), frame["label"].tolist())
            self.assertEqual(actual["prediction"].tolist(), predictions.tolist())
            self.assertEqual(actual["source_sha256"].tolist(), [
                hashlib.sha256(text.encode("utf-8")).hexdigest() for text in frame["text"]
            ])
            self.assertTrue(out.is_file())

    def test_splits_dir_is_forwarded_to_the_loader_and_recorded(self):
        frame = pd.DataFrame({"text": ["one", "two"], "label": [1, 2]})
        predictions = np.array([1, 2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits_dir = root / "bn_to_en"
            out = root / "report.json"
            argv = ["evaluate.py", "--model-dir", str(root), "--lang", "bn", "--split", "test",
                    "--splits-dir", str(splits_dir), "--bootstrap", "10", "--out", str(out)]
            with patch("sys.argv", argv), patch.object(evaluate, "load_split", return_value=frame) as load, \
                 patch.object(evaluate, "predict", return_value=predictions):
                evaluate.main()
            load.assert_called_with("bn", "test", out_dir=splits_dir)
            report = json.loads(out.read_text())
            self.assertEqual(report["splits_dir"], str(splits_dir))


if __name__ == "__main__":
    unittest.main()
