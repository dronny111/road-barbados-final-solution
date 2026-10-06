import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from road_ocr.buckets import pad_to_bucket
from road_ocr.line_crop import crop_to_line
from road_ocr.lines import load_line_image


class LineCropTest(unittest.TestCase):
    def test_trims_vertical_paper_but_keeps_the_full_width_and_ink(self):
        image = Image.new("RGB", (280, 100), (238, 235, 225))
        ImageDraw.Draw(image).rectangle((20, 40, 260, 60), fill=(25, 20, 15))

        cropped = crop_to_line(image)

        self.assertEqual(cropped.width, image.width)
        self.assertLess(cropped.height, image.height)
        self.assertLess(cropped.convert("L").getextrema()[0], 30)

    def test_blank_crop_is_left_at_its_original_size(self):
        image = Image.new("L", (100, 40), 230)
        cropped = crop_to_line(image)
        self.assertEqual(cropped.size, image.size)
        self.assertEqual(cropped.mode, "RGB")

    def test_rejects_invalid_controls(self):
        image = Image.new("RGB", (20, 20), "white")
        with self.assertRaises(ValueError):
            crop_to_line(image, threshold=256)
        with self.assertRaises(ValueError):
            crop_to_line(image, padding_fraction=-0.1)


class BucketTest(unittest.TestCase):
    def test_pads_to_the_next_patch_bucket_without_resizing_content(self):
        image = Image.new("RGB", (280, 56), (240, 235, 225))
        ImageDraw.Draw(image).rectangle((0, 0, 27, 55), fill=(10, 20, 30))

        padded = pad_to_bucket(image, patch=28)

        self.assertEqual(padded.size, (448, 56))
        offset = (padded.width - image.width) // 2
        self.assertEqual(padded.crop((offset, 0, offset + image.width, image.height)).tobytes(),
                         image.tobytes())

    def test_pixel_cap_keeps_the_existing_patch_aligned_width(self):
        image = Image.new("RGB", (280, 56), "white")
        self.assertIs(pad_to_bucket(image, patch=28, max_pixels=image.width * image.height), image)

    def test_rejects_unaligned_input(self):
        with self.assertRaises(ValueError):
            pad_to_bucket(Image.new("RGB", (281, 56)), patch=28)


class OptionalLineLoadingTest(unittest.TestCase):
    def test_autocrop_and_shape_buckets_work_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.jpg"
            image = Image.new("RGB", (560, 100), "white")
            ImageDraw.Draw(image).rectangle((20, 42, 540, 58), fill="black")
            image.save(path)

            loaded = load_line_image(
                path,
                target_height=112,
                max_pixels=451_584,
                autocrop=True,
                shape_buckets=True,
            )

        self.assertEqual(loaded.width % 28, 0)
        self.assertEqual(loaded.height % 28, 0)
        self.assertLessEqual(loaded.width * loaded.height, 451_584)


if __name__ == "__main__":
    unittest.main()
