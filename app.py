#!/usr/bin/env python3
"""Local ThreadSight prediction API and web interface."""

from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.dataset import CLASSES
from src.model import ClothingCNN


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
Image.MAX_IMAGE_PIXELS = 12_000_000


def load_models() -> tuple[object, ClothingCNN, dict]:
    metrics_path = ROOT / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Trained models are missing. Run `python train.py` first.")
    ml = joblib.load(ROOT / "models" / "classical_ml.joblib")
    cnn = ClothingCNN()
    cnn.load_state_dict(torch.load(ROOT / "models" / "cnn.pt", map_location="cpu", weights_only=True))
    cnn.eval()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return ml, cnn, metrics


def preprocess_upload(raw: bytes) -> np.ndarray:
    if len(raw) > 3_000_000:
        raise ValueError("Use an image smaller than 3 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Use a PNG, JPEG, or WebP image.")
        image = ImageOps.exif_transpose(image).convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this image.") from exc
    if image.width < 8 or image.height < 8:
        raise ValueError("Image is too small; use at least 8 × 8 pixels.")
    border = np.concatenate((
        np.asarray(image.crop((0, 0, image.width, 1))).ravel(),
        np.asarray(image.crop((0, image.height - 1, image.width, image.height))).ravel(),
        np.asarray(image.crop((0, 0, 1, image.height))).ravel(),
        np.asarray(image.crop((image.width - 1, 0, image.width, image.height))).ravel(),
    ))
    if float(np.median(border)) > 127:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    foreground = image.point(lambda pixel: 255 if pixel > 28 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise ValueError("No visible object was found. Try a clearer image.")
    image = image.crop(bounds)
    image.thumbnail((24, 24), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def image_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).resize((224, 224), Image.Resampling.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def rank_scores(scores: np.ndarray) -> dict:
    indices = np.argsort(scores)[::-1][:3]
    return {
        "label": CLASSES[int(indices[0])],
        "score": round(float(scores[indices[0]]), 4),
        "top_three": [{"label": CLASSES[int(i)], "score": round(float(scores[i]), 4)} for i in indices],
    }


class Handler(BaseHTTPRequestHandler):
    ml: object
    cnn: ClothingCNN
    metrics: dict

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/overview":
            report = self.metrics
            self.send_json(200, {
                "dataset": report["dataset"], "split": report["split"],
                "models": {
                    key: {"method": value["method"], "test": {"accuracy": value["test"]["accuracy"],
                           "macro_f1": value["test"]["macro_f1"],
                           "per_class": value["test"]["per_class"]}}
                    for key, value in report["models"].items()
                },
                "samples": report["samples"], "classes": CLASSES,
            })
            return
        filename = "index.html" if path == "/" else path.lstrip("/")
        file = (STATIC / filename).resolve()
        if not file.is_relative_to(STATIC) or not file.is_file():
            self.send_error(404, "File not found")
            return
        content_types = {".html": "text/html", ".css": "text/css", ".js": "application/javascript", ".png": "image/png", ".svg": "image/svg+xml"}
        self.send_bytes(200, file.read_bytes(), content_types.get(file.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Route not found."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 4_200_000:
                raise ValueError("Image request is too large.")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Invalid request.")
            expected = None
            if "sample" in data:
                sample = next((s for s in self.metrics["samples"] if s["file"] == data["sample"]), None)
                if sample is None:
                    raise ValueError("Sample was not found.")
                image = np.asarray(Image.open(STATIC / "samples" / sample["file"]).convert("L"), dtype=np.uint8)
                expected = sample["label"]
            elif "image" in data:
                encoded = data["image"]
                if not isinstance(encoded, str):
                    raise ValueError("Invalid image data.")
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                image = preprocess_upload(base64.b64decode(encoded, validate=True))
            else:
                raise ValueError("Choose a sample or upload an image.")
            features = image.reshape(1, -1).astype(np.float32) / 255.0
            ml_scores = self.ml.predict_proba(features)[0]
            with torch.inference_mode():
                logits = self.cnn(torch.from_numpy(features.reshape(1, 1, 28, 28)))
                dl_scores = torch.softmax(logits, dim=1)[0].numpy()
            self.send_json(200, {"processed_image": image_data_url(image), "expected_label": expected,
                                 "ml": rank_scores(ml_scores), "dl": rank_scores(dl_scores)})
        except (ValueError, json.JSONDecodeError, base64.binascii.Error) as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    Handler.ml, Handler.cnn, Handler.metrics = load_models()
    host = os.environ.get("THREADSIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("THREADSIGHT_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ThreadSight is running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
