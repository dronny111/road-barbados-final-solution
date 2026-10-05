import csv
import tempfile
import unittest
from pathlib import Path

from road_ocr.lines import select_training_rows


class TrainingSelectionTest(unittest.TestCase):
    def test_full_data_has_no_validation_and_needs_no_manifest(self):
        rows = [{"ID": "a", "Target": "one"}, {"ID": "b", "Target": "two"}]
        training, validation = select_training_rows(rows, "absent.csv", None, full_data=True)
        self.assertEqual(training, rows)
        self.assertEqual(validation, [])
        with self.assertRaises(ValueError):
            select_training_rows(rows, "absent.csv", 0, full_data=True)
        with self.assertRaises(ValueError):
            select_training_rows(rows, "absent.csv", None)

    def test_fold_training_excludes_every_held_out_row(self):
        rows = [{"ID": "a", "Target": "one"}, {"ID": "b", "Target": "two"}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "folds.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["ID", "fold"], ["a", 0], ["b", 1]])
            training, validation = select_training_rows(rows, manifest, 0)
        self.assertEqual(training, rows[1:])
        self.assertEqual(validation, rows[:1])


class ExcludeFoldTest(unittest.TestCase):
    def test_a_sibling_drops_a_second_fold_and_keeps_validation(self):
        import csv
        import tempfile
        from pathlib import Path
        from road_ocr.lines import select_training_rows

        rows = [{'ID': f'r{i}', 'Target': 't'} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'folds.csv'
            with manifest.open('w', newline='') as handle:
                csv.writer(handle).writerows([['ID', 'fold'], *[[r['ID'], i % 5] for i, r in enumerate(rows)]])
            base_fit, base_val = select_training_rows(rows, manifest, 0)
            fit, val = select_training_rows(rows, manifest, 0, exclude_fold=1)
            self.assertEqual(val, base_val)                                   # still held out on fold 0
            self.assertEqual({r['ID'] for r in fit}, {r['ID'] for r in base_fit} - {'r1', 'r6'})
            with self.assertRaisesRegex(ValueError, 'must differ'):
                select_training_rows(rows, manifest, 0, exclude_fold=0)
            with self.assertRaisesRegex(ValueError, 'full-data'):
                select_training_rows(rows, None, None, full_data=True, exclude_fold=1)
