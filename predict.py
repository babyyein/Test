"""
Inference script: apply a trained U-Net to reconstruct missing seismic traces.

Usage
-----
    python predict.py --checkpoint outputs/best_model.pth \
                      --input path/to/incomplete.npy \
                      --output path/to/reconstructed.npy \
                      --mask_path path/to/mask.npy        # optional

If ``--mask_path`` is not supplied, a random mask is generated using
``--missing_ratio`` (default 0.5).

The script can also visualise results with ``--plot``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from unet_model import UNet
from utils import load_checkpoint, compute_metrics, plot_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_to_divisible(arr: np.ndarray, divisor: int = 16) -> tuple[np.ndarray, tuple]:
    """
    Pad a 2-D array (T, X) so that both dimensions are divisible by *divisor*.

    Returns the padded array and the original shape for later cropping.
    """
    t, x = arr.shape
    pt = (divisor - t % divisor) % divisor
    px = (divisor - x % divisor) % divisor
    padded = np.pad(arr, ((0, pt), (0, px)), mode="reflect")
    return padded, (t, x)


def crop_to_original(arr: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Remove zero-padding added by :func:`pad_to_divisible`."""
    t, x = original_shape
    return arr[:t, :x]


# ---------------------------------------------------------------------------
# Reconstruct a single 2-D section
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    data: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    normalize: bool = True,
) -> np.ndarray:
    """
    Reconstruct a single seismic section.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net.
    data : ndarray  (T, X)
        Incomplete seismic section (missing traces already zeroed).
    mask : ndarray  (T, X)
        Binary mask – 1 = present, 0 = missing.
    device : torch.device
    normalize : bool
        Normalise the section before inference (mirrors training behaviour).

    Returns
    -------
    ndarray  (T, X)
        Reconstructed seismic section (in the original amplitude scale
        when ``normalize=True``).
    """
    # Remember original statistics for de-normalisation
    mean_ = data.mean()
    std_ = data.std()

    if normalize:
        if std_ > 0:
            data_n = (data - mean_) / std_
        else:
            data_n = data.copy()
    else:
        data_n = data.copy()

    # Pad so spatial dims are divisible by 16 (4 pooling layers × 2)
    data_padded, orig_shape = pad_to_divisible(data_n)
    mask_padded, _ = pad_to_divisible(mask.astype(np.float32))

    # Build input tensor: (1, 2, T, X)
    data_t = torch.from_numpy(data_padded[np.newaxis, np.newaxis]).float().to(device)
    mask_t = torch.from_numpy(mask_padded[np.newaxis, np.newaxis]).float().to(device)
    net_in = torch.cat([data_t, mask_t], dim=1)

    model.eval()
    pred = model(net_in)   # (1, 1, T_pad, X_pad)

    pred_np = pred.squeeze().cpu().numpy()   # (T_pad, X_pad)
    pred_np = crop_to_original(pred_np, orig_shape)

    # De-normalise
    if normalize and std_ > 0:
        pred_np = pred_np * std_ + mean_

    return pred_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct missing seismic traces with a trained U-Net"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True,
                   help="Path to input .npy file  (shape: (T,X) or (N,T,X))")
    p.add_argument("--output", default="reconstructed.npy",
                   help="Path to save reconstructed data (.npy)")
    p.add_argument("--mask_path", default=None,
                   help="Path to binary mask .npy file (same shape as input). "
                        "If omitted, a random mask is generated.")
    p.add_argument("--missing_ratio", type=float, default=0.5,
                   help="Missing-trace ratio when --mask_path is not provided")
    p.add_argument("--mask_type", choices=["random", "regular"], default="random")
    p.add_argument("--keep_every", type=int, default=2,
                   help="Trace decimation factor for regular mask")
    p.add_argument("--base_features", type=int, default=32)
    p.add_argument("--no_bilinear", action="store_true", default=False,
                   help="Use transposed convolutions instead of bilinear upsampling")
    p.add_argument("--no_normalize", action="store_true", default=False,
                   help="Skip per-section amplitude normalisation")
    p.add_argument("--plot", action="store_true", default=False,
                   help="Save comparison figures to --output_dir/figures/")
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ---------------------------------------------------------
    model = UNet(
        in_channels=2,
        out_channels=1,
        base_features=args.base_features,
        bilinear=not args.no_bilinear,
    ).to(device)

    ckpt = load_checkpoint(model, optimizer=None, checkpoint_path=args.checkpoint, device=device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Load input data ----------------------------------------------------
    raw = np.load(args.input).astype(np.float32)
    if raw.ndim == 2:
        raw = raw[np.newaxis]          # (1, T, X)
    n_samples, n_time, n_traces = raw.shape
    print(f"Input shape: {raw.shape}")

    # ---- Load or generate masks ---------------------------------------------
    if args.mask_path is not None:
        masks = np.load(args.mask_path).astype(np.float32)
        if masks.ndim == 2:
            masks = np.broadcast_to(masks[np.newaxis], raw.shape).copy()
    else:
        from dataset import random_mask, regular_mask
        masks = np.ones_like(raw)
        for i in range(n_samples):
            if args.mask_type == "random":
                m1d = random_mask(n_traces, args.missing_ratio, seed=i)
            else:
                m1d = regular_mask(n_traces, args.keep_every)
            masks[i] = np.broadcast_to(m1d[np.newaxis, :], (n_time, n_traces))

    # ---- Reconstruct --------------------------------------------------------
    reconstructed = np.zeros_like(raw)
    for i in range(n_samples):
        section = raw[i] * masks[i]   # apply mask
        pred = reconstruct(model, section, masks[i], device, normalize=not args.no_normalize)
        reconstructed[i] = pred
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Reconstructed {i + 1}/{n_samples} …")

    # ---- Save results -------------------------------------------------------
    np.save(args.output, reconstructed)
    print(f"Saved reconstructed data → {args.output}")

    # ---- Optional metrics ---------------------------------------------------
    from utils import signal_to_noise_ratio, peak_signal_to_noise_ratio
    snrs = [signal_to_noise_ratio(reconstructed[i], raw[i]) for i in range(n_samples)]
    psnrs = [peak_signal_to_noise_ratio(reconstructed[i], raw[i]) for i in range(n_samples)]
    print(f"Mean SNR  : {np.mean(snrs):.2f} dB")
    print(f"Mean PSNR : {np.mean(psnrs):.2f} dB")

    # ---- Optional visualisation --------------------------------------------
    if args.plot:
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        n_plot = min(5, n_samples)
        for i in range(n_plot):
            incomplete = raw[i] * masks[i]
            plot_comparison(
                incomplete,
                reconstructed[i],
                raw[i],
                title=f"Sample {i}  |  SNR={snrs[i]:.1f} dB",
                save_path=os.path.join(fig_dir, f"sample_{i:04d}.png"),
            )
        print(f"Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
