import csv
import tempfile
import unittest
from pathlib import Path

from road_ocr.records import CsvValidationError
from scripts.validate_submission import validate


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class SubmissionTest(unittest.TestCase):
    def test_accepts_complete_submission_in_any_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "test.csv", ["ID"], [["a"], ["b"]])
            write_csv(root / "sub.csv", ["ID", "Target"], [["b", "two"], ["a", "one"]])
            ids, _ = validate(root / "sub.csv", root / "test.csv")
            self.assertEqual(ids, ["a", "b"])

    def test_rejects_empty_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "test.csv", ["ID"], [["a"]])
            write_csv(root / "sub.csv", ["ID", "Target"], [["a", ""]])
            with self.assertRaises(CsvValidationError):
                validate(root / "sub.csv", root / "test.csv")


if __name__ == "__main__":
    unittest.main()
