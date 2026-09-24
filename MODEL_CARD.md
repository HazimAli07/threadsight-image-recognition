# ThreadSight model card

## Intended use

Explore the difference between a classical pixel-based classifier and a convolutional neural network on Fashion-MNIST. The browser demo is for local learning and portfolio review.

## Dataset

Fashion-MNIST has 60,000 official training images and 10,000 official test images. Each is a 28 × 28 grayscale image labeled as one of ten clothing categories. The project uses a stratified 55,000/5,000 fit/validation split from the official training set. The official test set is separate from model fitting and checkpoint selection.

The four files are downloaded from the dataset maintainers' repository and checked against the maintainers' published MD5 checksums. Sample images shown in the app come from the test set and are used only for demonstration, never fitting or model selection.

## Methods

- **Classical ML:** standardized normalized pixels, followed by scikit-learn's `SGDClassifier` with logistic loss.
- **Deep learning:** two convolution/ReLU/max-pooling blocks and a small dense classifier, trained with Adam and cross-entropy. The checkpoint with the highest validation accuracy across ten epochs is retained.

Both use random seed 42 and identical data splits. The classical classifier sees the image flattened into 784 values; the CNN retains spatial layout.

## Evaluation

Exact measured results are saved in [`reports/metrics.json`](reports/metrics.json). Test accuracy and macro F1 compare overall performance; per-class precision, recall, F1, and confusion matrices reveal where errors cluster. An earlier pilot run's test results were inspected before increasing the training set and epoch count, so the final test result is descriptive rather than a fully blind estimate.

## Limits

- Uploaded images are converted to grayscale, optionally inverted to put a light object on a dark background, cropped, centered, and reduced to 28 × 28. This preprocessing can change the appearance of a real photo.
- The benchmark images have simple backgrounds and predefined categories. Predictions on real-world photographs are exploratory and may be wrong.
- The ten classes exclude many kinds of clothing and accessories. The models will still choose one of the available labels for an unfamiliar object.
- The output scores are not calibrated confidence estimates.
- Results may vary slightly when retrained on a different machine, especially when using GPU acceleration.

## Privacy

The app binds to `127.0.0.1` by default. Uploaded images are processed in memory for a prediction and are not saved by the application.
