import sys
import unittest

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import stack_candidates as sc  # noqa: E402
from stratum_weighted_vote import build, pick  # noqa: E402

REFS = ["a b c", "x y z", "p q r", "a b c"]
MEMBERS = [dict(zip("1234", t)) for t in (["a b c", "x y z", "p q", "a b d"], ["a b d", "x y", "p q r", "a b d"],
                                          ["a b c", "x y z", "p q r", "a b c"])]


class StackTest(unittest.TestCase):
    def setUp(self):
        self.ids = list("1234")
        self.cost, self.ce, self.we, self.pools = build(MEMBERS, REFS, self.ids)
        nll = np.zeros(self.cost.shape[:2])
        X, mask = sc.features(self.cost, nll, nll, self.pools, [100] * 4)
        self.D = sc.Data(X, mask, self.ce, self.we, np.array([5.0] * 4), np.array([3.0] * 4), REFS)

    def test_uniform_choice_equals_vote_pick(self):
        got = sc.choose(self.D.X[:, :, :3].sum(2), self.D.mask)
        self.assertTrue((got == pick(self.cost, np.ones(3))).all())

    def test_grouped_folds_keep_equal_labels_together(self):
        f = sc.grouped_folds(REFS, 1, 2)
        self.assertEqual(f[0], f[3])

    def test_oracle_never_loses_to_a_stacker(self):
        rows = np.arange(4)
        oracle = self.D.combined(rows, sc.choose(self.D.y, self.D.mask))
        for n in ("hill",):
            self.assertLessEqual(oracle, self.D.combined(rows, sc.hill(self.D, rows, rows, 0, 3)) + 1e-12)


if __name__ == "__main__":
    unittest.main()
