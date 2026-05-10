# Handwritten Math Symbol Classification

A CNN-based image classification model using PyTorch to recognize handwritten mathematical symbols. The model classifies 32×32 grayscale images into 18 classes: digits (0–9) and common math operators (+, −, ×, ÷, ·, ±, [, ]). Built using a subset of the HASYv2 dataset.

## Dataset

[HASYv2](https://zenodo.org/record/259444) (Handwritten Symbol database v2) contains 168,233 images across 369 symbol classes. For this project, I filtered it down to 18 classes relevant to basic math notation, resulting in ~6,530 images.

The dataset is imbalanced — digit classes have around 60–130 samples each, while operator classes like `×` and `±` have over 1,500. The model handles this reasonably well without explicit rebalancing, though it's something worth addressing in a more thorough version.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download and extract HASYv2 from Zenodo (~33MB)
python3 download_data.py
```

## Usage

Train the CNN:

```bash
python3 train_model.py
```

Run predictions on sample images:

```bash
python3 predict.py
```

## Model

A simple two-layer CNN:

```
Conv2d(1, 32, 3×3) → ReLU → MaxPool
Conv2d(32, 64, 3×3) → ReLU → MaxPool
Flatten → Linear(4096, 128) → ReLU → Dropout(0.3)
Linear(128, 18)
```

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.97 |
| F1 (macro) | 0.94 |
| F1 (weighted) | 0.97 |

Digits and common operators are classified with high confidence. The `+` symbol has the lowest recall (0.56) — likely because some handwritten `+` symbols look similar to `×`, and the `+` class only has ~90 training samples. Most operator classes with more data (like `×`, `±`) perform very well.

Confusion matrix and sample predictions are saved to `results/`.

## Project Structure

```
├── data/                    # HASYv2 dataset (downloaded)
├── notebooks/
│   └── exploration.ipynb    # EDA and visualization
├── src/
│   ├── dataset.py           # Image loading, filtering, Dataset class
│   ├── model.py             # CNN architecture
│   ├── train.py             # Training and validation loops
│   └── evaluate.py          # Metrics and plotting
├── models/                  # Saved weights (.pt)
├── results/                 # Confusion matrix and prediction plots
├── train_model.py           # Main training script
├── predict.py               # Inference on samples
└── download_data.py         # Download HASYv2 from Zenodo
```

## What I'd improve next

- Address class imbalance with weighted loss or oversampling for underrepresented classes
- Add basic data augmentation (small rotations, slight shifts) to improve generalization
- Try BatchNormalization layers to stabilize training
- Experiment with a deeper architecture or different kernel sizes
