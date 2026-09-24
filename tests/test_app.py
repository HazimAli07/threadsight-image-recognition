"""Focused tests for image preprocessing and the running prediction API."""

from __future__ import annotations

import io
import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from app import Handler, load_models, preprocess_upload
from src.dataset import CLASSES


class PreprocessingTests(unittest.TestCase):
    def test_light_background_is_inverted_and_centered(self):
        image = Image.new("L", (80, 80), 255)
        for x in range(25, 55):
            for y in range(15, 65):
                image.putpixel((x, y), 20)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        output = preprocess_upload(buffer.getvalue())
        self.assertEqual(output.shape, (28, 28))
        self.assertLess(int(output[0, 0]), 20)
        self.assertGreater(int(output[14, 14]), 200)

    def test_blank_image_is_rejected(self):
        buffer = io.BytesIO()
        Image.new("L", (40, 40), 0).save(buffer, format="PNG")
        with self.assertRaisesRegex(ValueError, "No visible object"):
            preprocess_upload(buffer.getvalue())


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.ml, Handler.cnn, Handler.metrics = load_models()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_real_sample_returns_two_model_predictions(self):
        sample = Handler.metrics["samples"][0]
        body = json.dumps({"sample": sample["file"]}).encode()
        request = urllib.request.Request(self.base + "/api/predict", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["expected_label"], sample["label"])
        self.assertIn(result["ml"]["label"], CLASSES)
        self.assertEqual(len(result["dl"]["top_three"]), 3)
        self.assertTrue(result["processed_image"].startswith("data:image/png;base64,"))

    def test_unknown_sample_is_rejected(self):
        body = json.dumps({"sample": "not-a-sample.png"}).encode()
        request = urllib.request.Request(self.base + "/api/predict", body, {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_uploaded_image_uses_both_models(self):
        image_file = Path(__file__).resolve().parents[1] / "static" / "samples" / "8-0.png"
        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
        body = json.dumps({"image": f"data:image/png;base64,{encoded}"}).encode()
        request = urllib.request.Request(self.base + "/api/predict", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertIsNone(result["expected_label"])
        self.assertIn(result["ml"]["label"], CLASSES)
        self.assertIn(result["dl"]["label"], CLASSES)


if __name__ == "__main__":
    unittest.main()
