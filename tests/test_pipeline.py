"""
VaakBhav — ML Pipeline Integrity & Model Serialization Tests
"""

import unittest
import os
import joblib
from app import predict, model, char_vectorizer, label_encoder

class TestMLPipeline(unittest.TestCase):

    def test_model_artifacts_loaded(self):
        self.assertIsNotNone(model, "Model artifact sentiment_model.pkl is missing!")
        self.assertIsNotNone(char_vectorizer, "Vectorizer artifact char_vectorizer.pkl is missing!")
        self.assertIsNotNone(label_encoder, "Label encoder artifact label_encoder.pkl is missing!")

    def test_predict_schema(self):
        res = predict("Yeh mobile phone bahut mast hai aur sasta bhi hai.")
        self.assertIn("sentiment", res)
        self.assertIn(res["sentiment"], ["Positive", "Negative", "Neutral"])
        self.assertIn("scores", res)
        self.assertIn("confidence", res)
        self.assertIn("vader", res)
        self.assertIn("hinglish", res)
        self.assertIn("markers", res)

    def test_scores_sum_to_one(self):
        res = predict("The service was horrible and slow.")
        scores = res.get("scores", {})
        total_prob = sum(scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=1)

if __name__ == "__main__":
    unittest.main()
