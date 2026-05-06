# U-Net for Seismic Data Interpolation & Reconstruction

A PyTorch implementation of a **U-Net** network that reconstructs missing
traces in 2-D seismic sections.  Missing traces are simulated by applying a
random (or regular) subsampling mask during training; the network learns to
fill in the gaps from the surrounding context.

---

## Overview

| File | Purpose |
|---|---|
| `unet_model.py` | U-Net architecture (encoder → bottleneck → decoder with skip connections) |
| `dataset.py` | `SeismicDataset` + mask generators (`random_mask`, `regular_mask`) |
| `utils.py` | Loss function, SNR / PSNR metrics, visualisation, checkpoint helpers |
| `train.py` | End-to-end training script (synthetic data built-in for quick demos) |
| `predict.py` | Inference script – reconstruct and evaluate on new data |
| `tests/` | Pytest unit tests for all components |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Train (synthetic data, no files needed)

```bash
python train.py --epochs 50 --batch_size 8
```

Checkpoints and a CSV training log are saved to `./outputs/`.

### 3 – Train on your own data

Your data should be a NumPy `.npy` file of shape `(N, T, X)` (N sections,
T time samples, X traces) or `(T, X)` for a single section.
SEG-Y support is available when `segyio` is installed.

```bash
python train.py \
    --train_data data/train.npy \
    --val_data   data/val.npy   \
    --missing_ratio 0.5        \
    --mask_type random         \
    --epochs 100
```

### 4 – Reconstruct missing traces

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --input      data/incomplete.npy    \
    --output     data/reconstructed.npy \
    --plot
```

---

## Model Architecture

```
Input  (B, 2, T, X)      ← seismic data  +  binary mask
  │
  ▼  Encoder
  DoubleConv  64  ──────────────────────────────────────┐ skip 4
  MaxPool → DoubleConv 128  ────────────────────────────┤ skip 3
  MaxPool → DoubleConv 256  ────────────────────────────┤ skip 2
  MaxPool → DoubleConv 512  ────────────────────────────┤ skip 1
  MaxPool → DoubleConv 512 (bottleneck)
  │
  ▼  Decoder
  Up → DoubleConv 256  ←─ skip 1
  Up → DoubleConv 128  ←─ skip 2
  Up → DoubleConv  64  ←─ skip 3
  Up → DoubleConv  64  ←─ skip 4
  │
  OutConv 1×1
  │
Output (B, 1, T, X)      ← reconstructed seismic section
```

The input is a **2-channel** tensor: channel 0 is the incomplete seismic
section (missing traces zeroed out) and channel 1 is the binary acquisition
mask (1 = present, 0 = missing).

---

## Training Details

| Hyperparameter | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻³ |
| LR schedule | Cosine annealing |
| Loss | 0.5 × L1 + 0.5 × MSE (missing traces up-weighted ×2) |
| Missing ratio | 50 % (random mask) |

---

## Metrics

* **SNR** (Signal-to-Noise Ratio) in dB – higher is better
* **PSNR** (Peak SNR) in dB – higher is better

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## References

* Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
* Liu, D., Wang, J., et al. (2022). *Seismic Data Reconstruction Using Deep Learning*. Geophysics.
