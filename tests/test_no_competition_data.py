"""No competition ID or label line may appear in any tracked text file except the submitted CSV.

Runs only when Train.csv / Test.csv are present locally (they are never shipped); a clone without
the data skips it. Run it before every commit.
"""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "submission", "__pycache__", "data", "images", "predictions", "adapters", "output"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".safetensors", ".pt", ".pth", ".ckpt", ".zip", ".pyc"}


def competition_strings():
    found = set()
    for name in ("Train.csv", "Test.csv"):
        path = ROOT / name
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    found.add(row["ID"])
                    if len(row.get("Target", "")) >= 25:
                        found.add(row["Target"])
    return found


class NoCompetitionDataTest(unittest.TestCase):
    def test_no_id_or_label_in_tracked_text(self):
        strings = competition_strings()
        if not strings:
            self.skipTest("no local competition files")
        for path in ROOT.rglob("*"):
            if (not path.is_file() or path.suffix in SKIP_SUFFIXES or SKIP_DIRS & set(path.relative_to(ROOT).parts)
                    or path.name in {"Train.csv", "Test.csv", "SampleSubmission.csv"}):
                continue
            text = path.read_text(errors="ignore")
            leaked = [s for s in strings if s in text]
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertEqual(leaked[:3], [], f"competition strings found in {path.name}")


if __name__ == "__main__":
    unittest.main()
