import importlib.util
import random
import sys
import unittest
from collections import Counter

spec = importlib.util.spec_from_file_location("vlm_finetune", "scripts/vlm_finetune.py")
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)
sys.path.insert(0, "scripts")
import make_candidate_targets as mct  # noqa: E402


class SoftTargetTest(unittest.TestCase):
    def test_candidates_weight_the_vote_and_normalise(self):
        got = dict(map(tuple, mct.candidates("a", ["a", "b", "a", ""])))
        self.assertAlmostEqual(sum(got.values()), 1.0)
        self.assertEqual(got, {"a": 3 / 4, "b": 1 / 4})

    def test_real_labels_and_no_sampler_use_the_single_target(self):
        feature = finetune.with_candidates({"target": "x"}, None)
        self.assertEqual((feature["candidates"], feature["weights"]), (["x"], [1.0]))
        self.assertEqual(finetune.pick_target(feature, random.Random(1)), "x")
        soft = finetune.with_candidates({"target": "v"}, [["v", .9], ["w", .1]])
        self.assertEqual(finetune.pick_target(soft, None), "v")

    def test_sampling_is_seeded_and_follows_the_weights(self):
        soft = finetune.with_candidates({"target": "v"}, [["v", .9], ["w", .1]])
        draw = lambda seed: [finetune.pick_target(soft, r) for r in [random.Random(seed)] for _ in range(2000)]
        self.assertEqual(draw(3), draw(3))
        share = Counter(draw(3))["v"] / 2000
        self.assertTrue(0.86 < share < 0.94, share)


if __name__ == "__main__":
    unittest.main()
