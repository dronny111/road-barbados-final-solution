import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from road_ocr.lines import (
    model_patch,
    PATCH,
    clean_text,
    line_size,
    load_vlm,
    resolve_precision,
    split_by_fold,
)
from road_ocr.records import CsvValidationError


class CleanTextTest(unittest.TestCase):
    def test_collapses_every_whitespace_form_to_single_spaces(self):
        self.assertEqual(clean_text("  a\tb\n c  "), "a b c")


class ResolvePrecisionTest(unittest.TestCase):
    """Turing reports bf16 support that is software emulation, not hardware."""

    def resolved(self, major):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda *args, **kwargs: True,
            get_device_capability=lambda index=None: (major, 0),
        )
        torch = types.ModuleType("torch")
        torch.cuda, torch.bfloat16, torch.float16 = cuda, "bf16-dtype", "fp16-dtype"
        with patch.dict(sys.modules, {"torch": torch}):
            return resolve_precision("auto")[1]

    def test_auto_takes_fp16_on_pre_ampere_and_bf16_from_ampere_on(self):
        self.assertEqual(self.resolved(7), "fp16")  # T4, sm_75
        self.assertEqual(self.resolved(6), "fp16")  # P100, sm_60
        self.assertEqual(self.resolved(8), "bf16")  # A100, sm_80


class LineSizeTest(unittest.TestCase):
    def test_preserves_aspect_and_height_for_a_typical_crop(self):
        # Median crop in this dataset is about 1119x65, an aspect near 17:1.
        width, height = line_size(1119, 65, target_height=112, max_pixels=451_584)
        self.assertEqual(height, 112)
        self.assertAlmostEqual(width / height, 1119 / 65, delta=0.5)
        self.assertEqual(width % PATCH, 0)

    def test_caps_the_widest_crop_without_squashing_it(self):
        # Widest observed crop is about 40.5:1; it must scale down, not distort.
        width, height = line_size(6051, 149, target_height=112, max_pixels=451_584)
        self.assertLessEqual(width * height, 451_584)
        self.assertAlmostEqual(width / height, 6051 / 149, delta=0.5)
        self.assertGreater(height, PATCH)

    def test_never_exceeds_the_pixel_cap_across_the_observed_geometry(self):
        for width in (267, 1119, 3000, 6051):
            for height in (28, 65, 149, 1131):
                if not 3.0 <= width / height <= 41.0:
                    continue  # outside the geometry the preflight audit observed
                new_width, new_height = line_size(width, height, 112, 451_584)
                self.assertLessEqual(new_width * new_height, 451_584)
                self.assertAlmostEqual(
                    new_width / new_height, width / height, delta=0.5
                )

    def test_rejects_a_degenerate_image(self):
        with self.assertRaises(ValueError):
            line_size(0, 10, target_height=112, max_pixels=451_584)


class SplitByFoldTest(unittest.TestCase):
    def test_holds_out_exactly_the_requested_fold(self):
        rows = [{"ID": name, "Target": "x"} for name in ("a", "b", "c", "d")]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "folds.csv"
            manifest.write_text("ID,fold\na,0\nb,1\nc,0\nd,1\n", encoding="utf-8")
            train, validation = split_by_fold(rows, manifest, 1)
            self.assertEqual([row["ID"] for row in train], ["a", "c"])
            self.assertEqual([row["ID"] for row in validation], ["b", "d"])
            with self.assertRaises(CsvValidationError):
                split_by_fold(rows, manifest, 4)


class LoadVlmPlacementTest(unittest.TestCase):
    """MPS must be placed by hand; accelerate maps its overflow to disk.

    A disk-offloaded base model sends PeftModel.from_pretrained down an offload
    path that raises KeyError, and because the budget is free RAM at load time
    it only happened when the machine was busy.
    """

    def loaded(self, device):
        seen = {}

        class FakeModel:
            def to(self, target):
                seen["to"] = str(target)
                return self

        def from_pretrained(model_id, **kwargs):
            seen["kwargs"] = kwargs
            return FakeModel()

        transformers = types.ModuleType("transformers")
        transformers.AutoModelForImageTextToText = types.SimpleNamespace(
            from_pretrained=from_pretrained
        )
        with patch.dict(sys.modules, {"transformers": transformers}):
            load_vlm("some/checkpoint", "fp16-dtype", device=device, device_map="auto")
        return seen

    def test_places_an_mps_model_itself_instead_of_mapping_it(self):
        seen = self.loaded("mps")
        self.assertNotIn("device_map", seen["kwargs"])
        self.assertEqual(seen["to"], "mps")

    def test_leaves_the_device_map_alone_on_cuda(self):
        seen = self.loaded("cuda")
        self.assertEqual(seen["kwargs"]["device_map"], "auto")
        self.assertNotIn("to", seen)


if __name__ == "__main__":
    unittest.main()
