#!/usr/bin/env python3
"""Train a classical ML baseline and a CNN on identical Fashion-MNIST splits."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import CLASSES, load_dataset
from src.model import ClothingCNN


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SAMPLES = ROOT / "static" / "samples"
SEED = 42


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(10), zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": [
            {"class": CLASSES[i], "precision": round(float(precision[i]), 4),
             "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4),
             "support": int(support[i])}
            for i in range(10)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(10)).tolist(),
    }


def cnn_predict(model: ClothingCNN, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    loader = DataLoader(TensorDataset(x), batch_size=512)
    predictions = []
    with torch.inference_mode():
        for (batch,) in loader:
            predictions.append(model(batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train_cnn(x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> tuple[ClothingCNN, list[dict]]:
    model = ClothingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_images = torch.from_numpy(x_train.astype(np.float32) / 255.0).unsqueeze(1)
    train_labels = torch.from_numpy(y_train.astype(np.int64))
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=256, shuffle=True)
    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, 11):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_x)
        val_predictions = cnn_predict(model, x_val, device)
        val_accuracy = float(accuracy_score(y_val, val_predictions))
        row = {"epoch": epoch, "train_loss": round(loss_sum / len(x_train), 4),
               "validation_accuracy": round(val_accuracy, 4)}
        history.append(row)
        print(f"Epoch {epoch}: loss={row['train_loss']:.4f}, val_acc={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def make_samples(test_x: np.ndarray, test_y: np.ndarray) -> list[dict]:
    samples = []
    for class_id in range(10):
        for number, index in enumerate(np.flatnonzero(test_y == class_id)[:2]):
            filename = f"{class_id}-{number}.png"
            Image.fromarray(test_x[index]).save(SAMPLES / filename)
            samples.append({"file": filename, "label": CLASSES[class_id], "class_id": class_id})
    return samples


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    train_x_all, train_y_all, test_x, test_y = load_dataset(DATA)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_y_all)), test_size=5000, stratify=train_y_all, random_state=SEED
    )
    x_train, y_train = train_x_all[train_idx], train_y_all[train_idx]
    x_val, y_val = train_x_all[val_idx], train_y_all[val_idx]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}: {len(x_train)} train / {len(x_val)} validation / {len(test_x)} test", flush=True)

    ml_start = time.perf_counter()
    ml = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=250, tol=0.001, random_state=SEED),
    )
    ml.fit(x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0, y_train)
    ml_seconds = time.perf_counter() - ml_start
    joblib.dump(ml, MODELS / "classical_ml.joblib", compress=3)
    ml_val_pred = ml.predict(x_val.reshape(len(x_val), -1).astype(np.float32) / 255.0)
    ml_test_pred = ml.predict(test_x.reshape(len(test_x), -1).astype(np.float32) / 255.0)
    print(f"ML: val={accuracy_score(y_val, ml_val_pred):.4f}, test={accuracy_score(test_y, ml_test_pred):.4f}", flush=True)

    dl_start = time.perf_counter()
    cnn, history = train_cnn(x_train, y_train, x_val, y_val, device)
    dl_seconds = time.perf_counter() - dl_start
    torch.save(cnn.cpu().state_dict(), MODELS / "cnn.pt")
    dl_val_pred = cnn_predict(cnn, x_val, torch.device("cpu"))
    dl_test_pred = cnn_predict(cnn, test_x, torch.device("cpu"))
    print(f"DL: val={accuracy_score(y_val, dl_val_pred):.4f}, test={accuracy_score(test_y, dl_test_pred):.4f}", flush=True)

    report = {
        "dataset": "Fashion-MNIST",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "seed": SEED,
        "split": {"training": len(train_idx), "validation": len(val_idx), "official_test": len(test_x)},
        "models": {
            "classical_ml": {"method": "Standardized pixels + logistic SGD classifier",
                             "training_seconds": round(ml_seconds, 1),
                             "validation": evaluate(y_val, ml_val_pred), "test": evaluate(test_y, ml_test_pred)},
            "deep_learning": {"method": "Two-block convolutional neural network",
                              "training_seconds": round(dl_seconds, 1), "history": history,
                              "validation": evaluate(y_val, dl_val_pred), "test": evaluate(test_y, dl_test_pred)},
        },
        "samples": make_samples(test_x, test_y),
        "software": {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
                     "scikit_learn": __import__("sklearn").__version__},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved trained models and measured results.", flush=True)


if __name__ == "__main__":
    main()
