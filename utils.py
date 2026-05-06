"""
Utility functions for seismic data interpolation / reconstruction.

Covers:
  - Loss functions (MSE, L1, perceptual, combined)
  - Metrics      (SNR, SSIM, PSNR)
  - Visualisation helpers
  - Checkpoint save / load
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """
    Weighted combination of L1 and MSE losses.

    The loss is evaluated on *all* output samples but can optionally
    up-weight the contribution from the missing (masked-out) traces.

    Parameters
    ----------
    l1_weight : float
        Weight of the L1 component.
    mse_weight : float
        Weight of the MSE component.
    missing_weight : float
        Extra multiplier applied to the loss at missing-trace locations.
        Set to 1.0 to treat all locations equally.
    """

    def __init__(
        self,
        l1_weight: float = 0.5,
        mse_weight: float = 0.5,
        missing_weight: float = 2.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.missing_weight = missing_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor  (B, 1, T, X)
        target : Tensor  (B, 1, T, X)
        mask : Tensor  (B, 1, T, X), optional
            Binary mask – 1 for observed traces, 0 for missing.
        """
        diff = pred - target

        if mask is not None:
            # Up-weight missing trace locations
            weight = torch.ones_like(mask)
            weight[mask == 0] = self.missing_weight
        else:
            weight = torch.ones_like(diff)

        l1 = (weight * diff.abs()).mean()
        mse = (weight * diff.pow(2)).mean()

        return self.l1_weight * l1 + self.mse_weight * mse


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def signal_to_noise_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (SNR) in dB.

    SNR = 10 * log10( ||target||^2 / ||pred - target||^2 )
    """
    noise_power = np.mean((pred - target) ** 2)
    signal_power = np.mean(target ** 2)
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def peak_signal_to_noise_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

    Uses the actual dynamic range of the target as peak value.
    """
    data_range = target.max() - target.min()
    if data_range == 0:
        return float("inf")
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute SNR and PSNR for a batch.

    Parameters
    ----------
    pred, target : Tensor  (B, 1, T, X)

    Returns
    -------
    dict with keys "snr" and "psnr" (mean over batch, in dB).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    snrs, psnrs = [], []
    for p, t in zip(pred_np, target_np):
        snrs.append(signal_to_noise_ratio(p, t))
        psnrs.append(peak_signal_to_noise_ratio(p, t))
    finite_snrs = [s for s in snrs if np.isfinite(s)]
    finite_psnrs = [s for s in psnrs if np.isfinite(s)]
    return {
        "snr": float(np.mean(finite_snrs)) if finite_snrs else float("inf"),
        "psnr": float(np.mean(finite_psnrs)) if finite_psnrs else float("inf"),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(
    data: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
    clip_percentile: float = 98.0,
):
    """
    Side-by-side wiggle / image plot of input, prediction, and target.

    Parameters
    ----------
    data, pred, target : ndarray  (T, X)
        2-D seismic sections.
    title : str
        Figure title.
    save_path : str, optional
        If provided, save the figure to this path.
    clip_percentile : float
        Amplitude clip percentile for display.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualisation.") from exc

    vmax = np.percentile(np.abs(target), clip_percentile)
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (data, "Incomplete Input"),
        (pred, "U-Net Reconstruction"),
        (target, "Ground Truth"),
    ]
    for ax, (arr, label) in zip(axes, panels):
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Trace index")
        ax.set_ylabel("Time sample")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **extra,
):
    """Save a training checkpoint to *save_path*."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **extra,
    }
    torch.save(state, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device = None,
) -> dict:
    """
    Load a checkpoint saved by :func:`save_checkpoint`.

    Returns
    -------
    dict
        The full checkpoint dict (useful for resuming training state).
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
