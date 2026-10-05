import unittest

from road_ocr.metrics import edit_distance, score_pairs


class MetricsTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance([], []), 0)

    def test_exact_score_is_zero(self):
        result = score_pairs([("old hand", "old hand")])
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.combined, 0.0)

    def test_micro_weighting_favors_longer_references(self):
        result = score_pairs([("a", "x"), ("abcdefghi", "abcdefghi")])
        self.assertAlmostEqual(result.cer, 0.1)
        self.assertAlmostEqual(result.wer, 0.5)
        self.assertAlmostEqual(result.combined, 0.3)


if __name__ == "__main__":
    unittest.main()
