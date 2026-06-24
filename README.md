# Visibility Project — Fog Detection using Deep Learning

A binary image classification system that automatically detects **fog vs. no-fog** conditions from surveillance camera images using transfer learning with MobileNetV2.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Data Augmentation](#data-augmentation)
- [Evaluation & Metrics](#evaluation--metrics)
- [Results & Observations](#results--observations)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Project Overview

This project addresses the problem of automated **visibility detection** from road/traffic camera footage. Given an image captured by a surveillance camera, the model classifies it as either:

- `fog` — low visibility conditions
- `no_fog` — clear visibility conditions

The pipeline covers the full machine learning workflow: dataset preparation, model training with fine-tuned transfer learning, and comprehensive evaluation including ROC curves, precision-recall curves, and confusion matrices.

---

## Dataset

Images were sourced from multiple camera feeds across different years, split as follows:

| Split | Source Folders |
|-------|----------------|
| **Train** | `2023_2024 CAM1`, `2024_2025 CAM1`, `2024_2025 CAM2` |
| **Validation** | First half of `2022_2023 CAM1` (random 50/50 split) |
| **Test** | Second half of `2022_2023 CAM1` (random 50/50 split) |

The 2022–2023 data was intentionally held out for validation and testing to evaluate generalization to an earlier, unseen season. The train/val/test split was done using `split_dataset.py` with a fixed random seed (`seed=42`) for reproducibility.

**Classes:** `fog` | `no_fog`

**Image input size:** 224 × 224 pixels (RGB)

---

## Model Architecture

The model uses **MobileNetV2** as a pretrained backbone (ImageNet weights), followed by a custom classification head:

```
MobileNetV2 (pretrained on ImageNet)
  └─ GlobalAveragePooling2D
  └─ Dense(128, activation='relu')
  └─ Dropout(0.5)
  └─ Dense(1, activation='sigmoid')   ← binary output
```

**Fine-tuning strategy:** The last 40 layers of MobileNetV2 were unfrozen and trained alongside the classification head. All earlier layers were frozen to preserve low-level feature representations.

**Why MobileNetV2?**
- Lightweight and efficient for binary classification tasks
- Strong ImageNet pretraining makes it well-suited for texture/appearance-based tasks like fog detection
- Depth-wise separable convolutions keep parameter count low

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-5) |
| Loss | Binary Cross-Entropy |
| Batch Size | 16 |
| Epochs | 10 (with Early Stopping) |
| Early Stopping | `patience=5`, monitors `val_loss`, restores best weights |
| Class Weights | `fog: 1.8`, `no_fog: 1.0` |
| Saved Model | `fog_no_fog_model_FINAL.keras` |

**Class weighting** was applied to compensate for likely class imbalance — fog images tend to be rarer than clear-sky images in surveillance datasets, so the `fog` class was upweighted by a factor of 1.8.

---

## Data Augmentation

Applied only to the training set to improve generalization:

| Augmentation | Value |
|---|---|
| Rescale | 1/255 (pixel normalization) |
| Rotation Range | ±10° |
| Zoom Range | 15% |
| Brightness Range | [0.6, 1.0] |
| Horizontal Flip | Yes |

Validation and test sets were only rescaled (no augmentation) to reflect real-world inference conditions.

---

## Evaluation & Metrics

Evaluation is performed using `evaluate_model.py` with a **decision threshold of 0.45** (slightly below 0.5 to improve fog recall, since missing a fog condition is more costly than a false alarm).

The script generates:

- **Test Accuracy & Loss** — overall performance summary
- **Confusion Matrix** — saved as `confusion_matrix.png`
- **Classification Report** — per-class precision, recall, and F1-score
- **ROC Curve** (`roc_curve.png`) — with AUC score
- **Precision-Recall Curve** (`precision_recall_curve.png`) — particularly informative for imbalanced datasets

Output plots included in the repository:

| File | Description |
|------|-------------|
| `fog_no_fog.png` | Sample fog vs. no-fog image comparison |
| `roc_curve.png` | ROC curve from model evaluation |
| `precision_recall_curve.png` | Precision-recall tradeoff curve |
| `Final_values.png` | Summary of final evaluation metrics |

---

## Results & Observations

Based on the code, configuration choices, and the included result images:

- **Transfer learning with MobileNetV2** proved effective for this task due to the rich visual texture difference between foggy and clear scenes.
- **Class weighting (fog: 1.8)** helped the model avoid defaulting to the majority class, improving fog recall.
- **Threshold tuning to 0.45** (instead of 0.5) was a deliberate decision to bias the model toward detecting fog, reducing the risk of missed fog detections.
- **Fine-tuning the last 40 layers** of MobileNetV2 allowed the model to adapt to the domain-specific appearance of fog in surveillance camera imagery while retaining general visual features from ImageNet pretraining.
- **Early stopping (patience=5)** prevented overfitting, automatically restoring the best-performing checkpoint based on validation loss.
- The dataset split was designed to test **cross-season generalization** — training on 2023–2025 data and validating/testing on 2022–2023 data ensures the model isn't evaluated on data from the same time period it was trained on.

---

## Project Structure

```
Visibility-project/
│
├── split_dataset.py           # Splits raw images into train/val/test sets
├── train_model.py             # Builds, fine-tunes, and trains the MobileNetV2 model
├── evaluate_model.py          # Evaluates the trained model; generates metrics & plots
│
├── fog_no_fog_model_FINAL.keras   # Saved trained model
│
├── fog_no_fog.png             # Sample fog vs. no-fog images
├── roc_curve.png              # ROC curve plot
├── precision_recall_curve.png # Precision-recall curve plot
├── Final_values.png           # Final evaluation metrics summary
│
└── README.md
```

---

## Requirements

```
tensorflow >= 2.x
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## How to Run

**Step 1 — Prepare the dataset**

Update `BASE_DIR` in `split_dataset.py` to point to your local image folder, then run:

```bash
python split_dataset.py
```

This creates a `dataset/` directory with `train/`, `val/`, and `test/` subdirectories, each containing `fog/` and `no_fog/` class folders.

**Step 2 — Train the model**

Update `DATASET_DIR` in `train_model.py` if needed, then run:

```bash
python train_model.py
```

The trained model is saved as `fog_no_fog_model_FINAL.keras`.

**Step 3 — Evaluate the model**

```bash
python evaluate_model.py
```

This prints accuracy, loss, and a classification report to the console, and saves `confusion_matrix.png`, `roc_curve.png`, and `precision_recall_curve.png` to the working directory.

---

## Notes

- The dataset folder paths in the scripts are set to a local macOS path (`/Users/mac/Desktop/VISIBILITY_PROJECT/`). Update these paths before running on a different machine.
- The model is saved in the `.keras` format (TensorFlow/Keras native format, recommended over `.h5` for TF 2.x+).
