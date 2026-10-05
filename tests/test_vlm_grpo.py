import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vlm_grpo import combined_error, completion_text, corpus_edits_reward, transcription_reward


class CombinedErrorTest(unittest.TestCase):
    def test_exact_match_scores_zero(self):
        self.assertEqual(combined_error("the said John", "the said John"), 0.0)

    def test_matches_the_scorer_definition(self):
        # One character substitution in 13, which makes one word of 3 wrong.
        error = combined_error("the said John", "the sald John")
        self.assertAlmostEqual(error, 0.5 * (1 / 13) + 0.5 * (1 / 3))

    def test_rejects_an_empty_reference(self):
        with self.assertRaises(ValueError):
            combined_error("", "anything")


class CompletionTextTest(unittest.TestCase):
    def test_reads_a_plain_string(self):
        self.assertEqual(completion_text("hello"), "hello")

    def test_reads_a_chat_completion(self):
        self.assertEqual(completion_text([{"role": "assistant", "content": "hello"}]), "hello")

    def test_reads_structured_content(self):
        completion = [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
        self.assertEqual(completion_text(completion), "hello")

    def test_empty_completion_is_empty_text(self):
        self.assertEqual(completion_text([]), "")


class RewardTest(unittest.TestCase):
    def test_exact_transcription_earns_one(self):
        rewards = transcription_reward(["the said John"], ["the said John"])
        self.assertEqual(rewards, [1.0])

    def test_reward_is_never_negative(self):
        # A repetition loop is far longer than its reference, so raw 1 - error
        # would go negative and swamp the group's advantage.
        loop = "p: " * 60
        rewards = transcription_reward([loop], ["p:sone or p:sones whatsoever"])
        self.assertEqual(rewards, [0.0])

    def test_better_transcription_earns_more(self):
        close, far = transcription_reward(
            ["the said John", "entirely different words here"],
            ["the said John", "the said John"],
        )
        self.assertGreater(close, far)

    def test_ignores_whitespace_the_model_cannot_control(self):
        rewards = transcription_reward(["the said John"], ["the  said   John "])
        self.assertEqual(rewards, [1.0])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            transcription_reward(["a", "b"], ["a"])


class CorpusEditsRewardTest(unittest.TestCase):
    def reward(self, completions, targets):
        return corpus_edits_reward(completions, targets, mean_chars=13.0, mean_words=3.0)

    def test_exact_transcription_earns_zero(self):
        self.assertEqual(self.reward(["the said John"], ["the said  John"]), [0.0])

    def test_counts_edits_in_average_line_units(self):
        # One character edit in the average 13, one word edit in the average 3.
        (reward,) = self.reward(["the sald John"], ["the said John"])
        self.assertAlmostEqual(reward, -(0.5 / 13 + 0.5 / 3))

    def test_long_line_errors_outweigh_short_line_errors(self):
        # Same per-line error rate, but the long line costs the corpus more edits.
        # The clipped per-line reward cannot tell these apart.
        short, long = self.reward(
            ["the sald John", "the sald John the sald John"],
            ["the said John", "the said John the said John"],
        )
        self.assertLess(long, short)
        a, b = transcription_reward(
            ["the sald John", "the sald John the sald John"],
            ["the said John", "the said John the said John"],
        )
        self.assertAlmostEqual(a, b, places=2)  # the joining space makes it 27 chars, not 26

    def test_is_not_clipped(self):
        (reward,) = self.reward(["p: " * 60], ["p:sone or p:sones whatsoever"])
        self.assertLess(reward, -1.0)


if __name__ == "__main__":
    unittest.main()
