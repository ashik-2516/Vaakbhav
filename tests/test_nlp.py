"""
VaakBhav — Unit Tests for NLP Preprocessing Pipeline
"""

import unittest
from train_model import clean_text, NEGATION_WORDS, HINGLISH_MAP

class TestNLPPreprocessing(unittest.TestCase):

    def test_negation_preservation(self):
        text = "This movie is not good at all, bilkul acha nahi hai"
        cleaned = clean_text(text)
        self.assertIn("not", cleaned.split())
        self.assertIn("nahi", cleaned.split())

    def test_devanagari_script_retention(self):
        text = "यह फिल्म बहुत अच्छी थी story awesome hai"
        cleaned = clean_text(text)
        self.assertIn("यह", cleaned)
        self.assertIn("फिल्म", cleaned)
        self.assertIn("awesome", cleaned)

    def test_character_elongation_normalization(self):
        text = "superrrrr movie achaaaa"
        cleaned = clean_text(text)
        self.assertIn("superr", cleaned)
        self.assertIn("achaa", cleaned)

    def test_hinglish_term_mapping(self):
        text = "bahut bakwas quality hai"
        cleaned = clean_text(text)
        self.assertIn("terrible", cleaned)
        self.assertIn("very", cleaned)

if __name__ == "__main__":
    unittest.main()
