# Evaluation results

Both models were fitted on the same 55,000 Fashion-MNIST training examples and assessed on the same 5,000-image validation split. Final figures below use the official 10,000-image test set. The CNN checkpoint was chosen by validation accuracy.

| Model | Validation accuracy | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| Classical ML: standardized pixels + logistic SGD | 85.02% | 82.59% | 0.8248 |
| Deep learning: two-block CNN | 92.38% | **90.88%** | **0.9074** |

The CNN's test accuracy is **8.29 percentage points** higher in this run. That is a comparison on this benchmark and split, not a claim that a CNN will always outperform a classical model on other image tasks.

## What the errors show

Shirt is the hardest class for both models. Its test recall is 52.9% for classical ML and 65.6% for the CNN. The CNN labels 171 of the 1,000 test shirts as T-shirt/top. Similar shapes also cause confusion between coats and pullovers.

| Class | ML recall | CNN recall |
| --- | ---: | ---: |
| T-shirt/top | 76.9% | 91.7% |
| Trouser | 94.6% | 98.1% |
| Pullover | 72.5% | 86.9% |
| Dress | 86.8% | 92.8% |
| Coat | 71.0% | 84.3% |
| Sandal | 91.5% | 98.6% |
| Shirt | 52.9% | 65.6% |
| Sneaker | 91.2% | 95.6% |
| Bag | 93.8% | 98.5% |
| Ankle boot | 94.7% | 96.7% |

The complete per-class precision, recall, F1, support, confusion matrices, and CNN epoch history are in [`metrics.json`](metrics.json). The table above is a human-readable summary of that machine-readable report.

## Interpretation limits

These are 28 × 28 grayscale catalogue images with simple backgrounds. Uploaded photos may differ substantially. The models must choose one of ten labels even for an unfamiliar object. An earlier pilot run's test results were inspected before increasing the training set and epoch count, so this final test result should be treated as descriptive rather than fully blind. Only one random seed was run; no uncertainty interval is reported.
