import unittest

from scripts.make_folds import assign_folds


class FoldTest(unittest.TestCase):
    def test_duplicate_labels_stay_together_and_assignment_is_stable(self):
        rows = [
            {"ID": "a", "Target": "same label"},
            {"ID": "b", "Target": "same label"},
            {"ID": "c", "Target": "another line"},
            {"ID": "d", "Target": "third line"},
        ]
        first = assign_folds(rows, folds=2, seed=7)
        second = assign_folds(rows, folds=2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["a"], first["b"])


if __name__ == "__main__":
    unittest.main()
