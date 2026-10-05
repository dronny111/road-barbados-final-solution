import sys
import unittest

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from make_confident_pseudo import confident  # noqa: E402


class ConfidentTest(unittest.TestCase):
    def test_threshold_is_ceil_two_thirds_and_blank_votes_dropped(self):
        vote = {"a": "x y", "b": "x y", "c": "p", "d": " "}
        members = [{"a": "x y", "b": "x y", "c": "p", "d": ""}, {"a": "x y", "b": "x z", "c": "q", "d": ""},
                   {"a": "x y", "b": "x z", "c": "q", "d": ""}]
        # N=3 -> need 2: a has 3, b has 1, c has 1; the blank vote is never kept.
        self.assertEqual(confident(vote, members), {"a": 3})
        self.assertEqual(set(confident(vote, members, frac=1 / 3)), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
