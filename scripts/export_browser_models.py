#!/usr/bin/env python3
"""Export the trained ThreadSight models for browser-only inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import CLASSES  # noqa: E402

OUT = ROOT / "docs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ml = joblib.load(ROOT / "models/classical_ml.joblib")
    scaler = ml.named_steps["standardscaler"]
    classifier = ml.named_steps["sgdclassifier"]
    state = torch.load(ROOT / "models/cnn.pt", map_location="cpu", weights_only=True)
    assert classifier.classes_.tolist() == list(range(len(CLASSES)))

    arrays = {
        "ml_mean": scaler.mean_,
        "ml_scale": scaler.scale_,
        "ml_coef": classifier.coef_,
        "ml_intercept": classifier.intercept_,
        "conv1_weight": state["features.0.weight"].numpy(),
        "conv1_bias": state["features.0.bias"].numpy(),
        "conv2_weight": state["features.3.weight"].numpy(),
        "conv2_bias": state["features.3.bias"].numpy(),
        "dense1_weight": state["classifier.1.weight"].numpy(),
        "dense1_bias": state["classifier.1.bias"].numpy(),
        "dense2_weight": state["classifier.4.weight"].numpy(),
        "dense2_bias": state["classifier.4.bias"].numpy(),
    }
    manifest = {"classes": CLASSES, "arrays": {}}
    binary = bytearray()
    for name, value in arrays.items():
        array = np.asarray(value, dtype="<f4", order="C")
        offset = len(binary) // 4
        binary.extend(array.tobytes())
        manifest["arrays"][name] = {"offset": offset, "shape": list(array.shape)}

    (OUT / "weights.bin").write_bytes(binary)
    (OUT / "model-manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")

    report = json.loads((ROOT / "reports/metrics.json").read_text())
    overview = {
        "split": report["split"],
        "models": {name: {"test": {"accuracy": data["test"]["accuracy"],
                                   "macro_f1": data["test"]["macro_f1"]}}
                   for name, data in report["models"].items()},
        "samples": report["samples"],
    }
    (OUT / "overview.json").write_text(json.dumps(overview, separators=(",", ":")) + "\n")
    print(f"Exported {len(binary):,} model bytes and {len(overview['samples'])} samples")


if __name__ == "__main__":
    main()
