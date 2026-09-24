# ThreadSight

ThreadSight is a local image-recognition project that compares **classical machine learning** with **deep learning** on the same clothing images. It includes trained model files, measured results, a browser demo, and the full reproducible training pipeline.

The browser demo accepts a PNG, JPEG, or WebP image and shows the predictions from both models, their top three output scores, and the 28 × 28 grayscale image they actually received. It also includes real sample images from the held-out test set.

## Run the demo

Python 3.12 is recommended for the included demo. On macOS or Linux, open Terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000** in your browser. Press `Ctrl+C` in Terminal to stop it. On Windows, use `py -3.12 -m venv .venv`, `.venv\Scripts\activate`, then the same `pip` and `python app.py` commands.

The model files are already included, so training is not needed to try the demo. Images are processed locally by this app and are not saved. The included weights were trained with PyTorch 2.14 and checked in the demo with PyTorch 2.13; the requirements file pins the tested demo environment.

## What the models do

| Approach | Implementation | Input |
| --- | --- | --- |
| Classical ML | StandardScaler + logistic SGD classifier (scikit-learn) | 784 normalized pixel values |
| Deep learning | Two convolution blocks, pooling, dense layer (PyTorch) | One 28 × 28 grayscale image |

Both models use the same stratified split of the official 60,000 training images: 55,000 for fitting and 5,000 for validation. The official 10,000-image test set is separate from fitting and checkpoint selection. The CNN checkpoint is chosen by validation accuracy. The exact random seed is 42.

Measured accuracy, macro F1, per-class results, confusion matrices, software versions, and CNN training history are in [`reports/metrics.json`](reports/metrics.json). See the readable [`results summary`](reports/RESULTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md) for interpretation and limitations.

## Reproduce training

With the environment active, run:

```bash
python train.py
```

The script downloads the four official Fashion-MNIST files to `data/`, verifies their MD5 checksums against the dataset maintainers' published checksums, trains both models, evaluates them, and replaces the included model and report files. Training works on Apple Silicon MPS when available and falls back to CPU. It may take longer on other machines.

## Project files

```text
app.py                 Local HTTP server and prediction API
train.py               Download, split, train, evaluate, and save models
src/dataset.py         Verified Fashion-MNIST loader
src/model.py           CNN architecture
static/                Browser interface and 20 held-out sample images
models/                Trained classical and CNN weights
reports/metrics.json   Measured results and confusion matrices
tests/                 Smoke and preprocessing tests
```

## Dataset and attribution

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) was created by Han Xiao, Kashif Rasul, and Roland Vollgraf at Zalando Research. It contains 28 × 28 grayscale product images in ten classes and is provided under the MIT License. The included sample images come from its test split. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the dataset license notice.

This is a benchmark and learning project for Fashion-MNIST's ten classes, not a general clothing recognition service. A photograph with a complex background may perform poorly even when the test-set accuracy is high. The displayed model scores have not been calibrated as probabilities of correctness.

## Project attribution

Project owner: Hazim Ali. Implementation was completed with Codex assistance. Dataset attribution and license are documented above.
