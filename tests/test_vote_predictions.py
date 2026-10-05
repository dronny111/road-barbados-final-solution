import unittest

from scripts.vote_predictions import line_cost, mbr_vote, vote, word_vote
from scripts.make_pseudo_labels import agreed


class VoteTest(unittest.TestCase):
    """Two members must agree exactly; otherwise the first member's text stands."""

    def vote(self, *members):
        return vote([dict(zip('abcd', texts, strict=False)) for texts in members])

    def test_agreement_and_fallback(self):
        voted, counts = self.vote(
            ['same', 'lone', 'pair', 'first'],   # fallback member
            ['same', 'other', 'pair', 'second'],
            ['same', 'third', 'nope', 'third'],
        )
        self.assertEqual(voted, {'a': 'same', 'b': 'lone', 'c': 'pair', 'd': 'first'})
        self.assertEqual(dict(counts), {'unanimous': 1, 'fallback': 2, 'majority': 1})

    def test_majority_outvotes_the_fallback_member(self):
        voted, _ = self.vote(['alone'], ['agreed'], ['agreed'])
        self.assertEqual(voted, {'a': 'agreed'})

    def test_members_must_cover_the_same_ids(self):
        with self.assertRaisesRegex(ValueError, 'different IDs'):
            vote([{'a': 'x'}, {'b': 'x'}, {'a': 'x'}])

    def test_an_even_member_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'odd number'):
            vote([{'a': 'x'}, {'a': 'x'}])



class WordVoteTest(unittest.TestCase):
    """A strict majority overrides the first member one word or gap at a time."""

    def vote(self, *texts):
        return word_vote([{'a': t} for t in texts])[0]['a']

    def test_each_member_wrong_on_a_different_word(self):
        # The exact vote falls back for this line; the word vote repairs both errors.
        self.assertEqual(self.vote('the sayd land of', 'the said lond of', 'the said land of'),
                         'the said land of')

    def test_a_lone_member_cannot_override(self):
        self.assertEqual(self.vote('the said land', 'the sayd land', 'the said lond'), 'the said land')

    def test_majority_deletion_and_insertion(self):
        self.assertEqual(self.vote('the the said land', 'the said land', 'the said land'), 'the said land')
        self.assertEqual(self.vote('said land', 'the said land', 'the said land'), 'the said land')
        self.assertEqual(self.vote('the said', 'the said land', 'the said land'), 'the said land')

    def test_unchanged_rows_keep_their_exact_spacing(self):
        self.assertEqual(self.vote('the  said', 'the said', 'the sayd'), 'the  said')

    def test_ids_and_member_count_are_checked(self):
        with self.assertRaisesRegex(ValueError, 'different IDs'):
            word_vote([{'a': 'x'}, {'b': 'x'}, {'a': 'x'}])
        with self.assertRaisesRegex(ValueError, 'odd number'):
            word_vote([{'a': 'x'}, {'a': 'x'}])



class MbrVoteTest(unittest.TestCase):
    def pick(self, *texts, with_word_vote=False):
        return mbr_vote([{'a': t} for t in texts], with_word_vote)[0]['a']

    def test_cost_is_the_leaderboard_line_cost(self):
        self.assertAlmostEqual(line_cost('the sayd land', 'the said land'), 1 / 12 + 1 / 55)
        self.assertEqual(line_cost('same', 'same'), 0)

    def test_picks_the_consensus_reading_and_keeps_the_pivot_on_ties(self):
        self.assertEqual(self.pick('the sayd land', 'the said land', 'the said land'), 'the said land')
        self.assertEqual(self.pick('alpha', 'beta', 'gamma'), 'alpha')  # all equally far: fallback

    def test_word_vote_candidate_can_beat_every_member(self):
        # Each member is wrong on a different word; only the word vote is right everywhere.
        texts = ('the sayd land of', 'the said lond of', 'the said land off')
        self.assertIn(self.pick(*texts), texts)
        self.assertEqual(self.pick(*texts, with_word_vote=True), 'the said land of')


class PseudoLabelTest(unittest.TestCase):
    def test_keeps_only_lines_every_member_decodes_identically(self):
        a = {'x': 'the said', 'y': 'land', 'z': ' '}
        b = {'x': 'the said', 'y': 'lond', 'z': ' '}
        self.assertEqual(agreed([a, b]), {'x': 'the said'})  # blank agreement is dropped
        with self.assertRaisesRegex(ValueError, 'different IDs'):
            agreed([a, {'x': 'the said'}])


if __name__ == '__main__':
    unittest.main()
