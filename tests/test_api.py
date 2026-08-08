"""
VaakBhav — Integration & API Endpoint Unit Tests
"""

import unittest
import json
from app import app

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get("status"), "ok")

    def test_predict_single_text_positive(self):
        payload = {"text": "Yeh film toh kamaal ki thi! Ek dum mast story!"}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get("sentiment"), "Positive")
        self.assertIn("confidence", data)
        self.assertIn("scores", data)

    def test_predict_single_text_negative(self):
        payload = {"text": "Yeh product bilkul bekar hai, waste of money, ghatiya quality."}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get("sentiment"), "Negative")

    def test_predict_empty_input(self):
        payload = {"text": "   "}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)

    def test_api_compare(self):
        payload = {"texts": ["Awesome movie!", "Worst experience ever."]}
        res = self.client.post("/api/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data.get("results", [])), 2)

    def test_api_word_sentiment(self):
        payload = {"text": "good product but bad service"}
        res = self.client.post("/api/word-sentiment", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(len(data.get("words", [])) > 0)

if __name__ == "__main__":
    unittest.main()
