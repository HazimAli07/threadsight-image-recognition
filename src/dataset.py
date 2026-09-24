"""Download and read the official Fashion-MNIST IDX files."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np


BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
    "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def download_dataset(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in FILES.items():
        target = folder / filename
        if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == expected_md5:
            continue
        print(f"Downloading {filename}...", flush=True)
        request = urllib.request.Request(f"{BASE}/{filename}", headers={"User-Agent": "ThreadSight/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        actual_md5 = hashlib.md5(data).hexdigest()
        if actual_md5 != expected_md5:
            raise ValueError(f"Checksum mismatch for {filename}: {actual_md5}")
        target.write_bytes(data)


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(f"Unexpected image format: {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    if data.size != count * rows * columns:
        raise ValueError(f"Unexpected image count: {path}")
    return data.reshape(count, rows, columns)


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label format: {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"Unexpected label count: {path}")
    return data


def load_dataset(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_dataset(folder)
    train_x = read_images(folder / "train-images-idx3-ubyte.gz")
    train_y = read_labels(folder / "train-labels-idx1-ubyte.gz")
    test_x = read_images(folder / "t10k-images-idx3-ubyte.gz")
    test_y = read_labels(folder / "t10k-labels-idx1-ubyte.gz")
    if len(train_x) != 60000 or len(test_x) != 10000:
        raise ValueError("Official dataset sizes did not match.")
    return train_x, train_y, test_x, test_y
