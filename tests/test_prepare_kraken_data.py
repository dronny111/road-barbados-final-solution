import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_colab_ppocr_notebook import build  # noqa: E402
from prepare_kraken_data import materialize, read_fold_map, select_rows  # noqa: E402
from road_ocr.records import CsvValidationError  # noqa: E402


class PrepareKrakenDataTest(unittest.TestCase):
    def test_selection_is_stable_and_rejects_an_incomplete_manifest(self):
        rows = [{"ID": name, "Target": name.upper()} for name in ("a", "b", "c", "d")]
        folds = {"a": 0, "b": 1, "c": 0, "d": 1}

        first = select_rows(rows, folds, 0, 123, None, None)
        second = select_rows(list(reversed(rows)), folds, 0, 123, None, None)

        self.assertEqual(first, second)
        self.assertEqual({row["ID"] for row in first[0]}, {"b", "d"})
        self.assertEqual({row["ID"] for row in first[1]}, {"a", "c"})
        with self.assertRaises(CsvValidationError):
            select_rows(rows, {"a": 0}, 0, 123, None, None)

    def test_fold_reader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            path.write_text("ID,fold\na,0\na,1\n", encoding="utf-8")
            with self.assertRaises(CsvValidationError):
                read_fold_map(path)

    def test_materializes_images_labels_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (20, 10), "white").save(images / "a.jpg")
            output = root / "data" / "train"
            manifest = root / "data" / "train.txt"

            materialize([{"ID": "a", "Target": "A line"}], images, output, manifest)

            self.assertTrue((output / "a.jpg").is_file())
            self.assertEqual((output / "a.gt.txt").read_text(), "A line")
            self.assertEqual(manifest.read_text().strip(), str((output / "a.jpg").absolute()))

    def test_colab_builder_embeds_every_required_source(self):
        # Generated Kaggle notebooks have no .git directory, so keep this test
        # valid both in a clone and in the self-contained notebook runtime.
        with patch("build_colab_ppocr_notebook.git", return_value=""):
            notebook = build(fold=0, epochs=1, lr=2e-4, batch=2, seed=20260906)
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn('"scripts/prepare_kraken_data.py"', source)
        self.assertIn("def select_rows", source)


if __name__ == "__main__":
    unittest.main()
