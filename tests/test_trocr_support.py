import tempfile
import unittest
from pathlib import Path

from road_ocr.metrics import score_pairs
from road_ocr.trocr_support import exact_metrics, estimate_hours, require_budget, verify_folds


class TrOCRGuardsTest(unittest.TestCase):
    def test_original_spacing_is_preserved_in_reference(self):
        refs, preds = ["a  b", "c"], ["a b", "c"]
        expected = score_pairs(zip(refs, preds))
        actual = exact_metrics(refs, preds)
        self.assertEqual(actual["combined"], expected.combined)
        self.assertGreater(actual["cer"], 0)
        with self.assertRaises(ValueError):
            exact_metrics(refs, preds[:1])

    def test_changed_fold_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            path.write_text("ID,fold\nsynthetic,0\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_folds(path)

    def test_estimate_includes_epoch_evaluation_and_test_decoding(self):
        estimate = estimate_hours(train_seconds=20, smoke_steps=2, train_rows=160,
            effective_batch=16, epochs=3, decode_seconds=8, smoke_rows=8,
            validation_rows=40, test_rows=100, safety_factor=1)
        self.assertAlmostEqual(estimate * 3600, 300 + 160 + 100)

    def test_budget_includes_elapsed_work_and_rejects_nan(self):
        require_budget(1, 1800, 2)
        with self.assertRaisesRegex(RuntimeError, "Budget gate"):
            require_budget(1.75, 1800, 2)
        with self.assertRaises(ValueError):
            require_budget(float("nan"), 0, 2)
