"""Training-only jitter: reproducible, size-preserving, and never on by default."""
import importlib.util
import random
import unittest

from road_ocr.augment import DEFAULTS, augment_line


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow optional")
class AugmentTest(unittest.TestCase):
    @staticmethod
    def page():
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (448, 112), (232, 222, 198))
        draw = ImageDraw.Draw(image)
        for x in range(10, 430, 17):
            draw.line((x, 80, x + 9, 30), fill=(40, 30, 25), width=2)
        return image

    def test_canvas_and_mode_survive_every_draw(self):
        image = self.page()
        for seed in range(25):
            result = augment_line(image, random.Random(seed))
            self.assertEqual(result.size, image.size)
            self.assertEqual(result.mode, "RGB")

    def test_same_seed_repeats_and_different_seeds_diverge(self):
        image = self.page()
        first = augment_line(image, random.Random(7)).tobytes()
        self.assertEqual(first, augment_line(image, random.Random(7)).tobytes())
        others = {augment_line(image, random.Random(s)).tobytes() for s in range(1, 9)}
        self.assertGreater(len(others), 1)
        self.assertNotIn(image.tobytes(), others)

    def test_a_shared_generator_keeps_drawing_fresh_jitter(self):
        image, rng = self.page(), random.Random(11)
        draws = {augment_line(image, rng).tobytes() for _ in range(6)}
        self.assertGreater(len(draws), 1)

    def test_zeroed_settings_are_an_exact_no_op_and_negatives_are_refused(self):
        image = self.page()
        off = dict.fromkeys(DEFAULTS, 0)
        self.assertEqual(augment_line(image, random.Random(3), **off).tobytes(), image.tobytes())
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            augment_line(image, random.Random(3), blur=-0.1)

    def test_the_collator_leaves_images_alone_unless_asked(self):
        spec = importlib.util.spec_from_file_location("vlm_finetune", "scripts/vlm_finetune.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seen = []

        class Processor:
            tokenizer = staticmethod(lambda text, **kw: {"input_ids": [1]})
            apply_chat_template = staticmethod(lambda messages, **kw: "prompt")

            def __call__(self, text, images, **kwargs):
                seen.extend(group[0] for group in images)
                raise RuntimeError("stop after the images are collected")

        image = self.page()
        features = [{"image": image, "target": "x"}]
        for collator, augmented in ((module.Collator(Processor()), False),
                                    (module.Collator(Processor(), augment=random.Random(5)), True)):
            seen.clear()
            with self.assertRaises(RuntimeError):
                collator(features)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].tobytes() != image.tobytes(), augmented)
            self.assertEqual(collator.augmented_examples, int(augmented))


if __name__ == "__main__":
    unittest.main()
