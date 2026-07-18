import unittest

from summarizer import generate_summary
from discriminator import score_summary


class RuntimeFallbackTests(unittest.TestCase):
    def test_generate_summary_returns_non_empty_text(self):
        text = "The quick brown fox jumps over the lazy dog. This sentence is included to test fallback summarization logic."
        summary = generate_summary(text)
        self.assertIsInstance(summary, str)
        self.assertTrue(summary.strip())

    def test_score_summary_returns_float_in_range(self):
        score = score_summary("A short article about cats.", "Cats are pets.")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
