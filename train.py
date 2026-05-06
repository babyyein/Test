"""
Training script for the U-Net seismic interpolation / reconstruction model.

Usage
-----
Train with default options (uses synthetic data when no paths are given):

    python train.py

Train on real data:

    python train.py --train_data path/to/train.npy --val_data path/to/val.npy

See ``python train.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SeismicDataset, build_loaders
from unet_model import UNet
from utils import ReconstructionLoss, compute_metrics, save_checkpoint, set_seed


# ---------------------------------------------------------------------------
# Synthetic data generator (for quick demos / smoke tests)
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_samples: int = 200, n_time: int = 128, n_traces: int = 128) -> str:
    """
    Generate a small synthetic seismic dataset and save it to a temp .npy file.

    Each section is a superposition of dipping plane waves with random slopes
    and amplitudes – a simple but representative seismic model.

    Returns the path to the saved .npy file.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n_time)
    x = np.arange(n_traces)
    data = np.zeros((n_samples, n_time, n_traces), dtype=np.float32)

    for i in range(n_samples):
        n_events = rng.integers(2, 6)
        for _ in range(n_events):
            slope = rng.uniform(-0.5, 0.5)   # samples / trace
            t0 = rng.integers(10, n_time - 10)
            amp = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
            freq = rng.uniform(0.05, 0.2)    # cycles / sample
            tt = t0 + slope * x              # (n_traces,)  broadcast below
            envelope = np.exp(-0.5 * ((t[:, None] - tt[None, :]) / 8) ** 2)
            wavelet = np.sin(2 * np.pi * freq * (t[:, None] - tt[None, :]))
            data[i] += amp * (envelope * wavelet).astype(np.float32)

    path = "/tmp/seismic_synthetic.npy"
    np.save(path, data)
    return path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ReconstructionLoss,
    device: torch.device,
    scaler,
) -> dict:
    model.train()
    total_loss = 0.0
    total_snr = 0.0
    total_psnr = 0.0

    for batch in loader:
        data = batch["data"].to(device)    # (B, 1, T, X)  incomplete
        mask = batch["mask"].to(device)    # (B, 1, T, X)
        target = batch["target"].to(device)

        # Network input: concatenate data + mask along channel axis → (B, 2, T, X)
        net_input = torch.cat([data, mask], dim=1)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(net_input)
                loss = criterion(pred, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(net_input)
            loss = criterion(pred, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = compute_metrics(pred, target)
        total_loss += loss.item()
        total_snr += metrics["snr"]
        total_psnr += metrics["psnr"]

    n = len(loader)
    return {
        "loss": total_loss / n,
        "snr": total_snr / n,
        "psnr": total_psnr / n,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    total_psnr = 0.0

    for batch in loader:
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        net_input = torch.cat([data, mask], dim=1)
        pred = model(net_input)
        loss = criterion(pred, target, mask)

        metrics = compute_metrics(pred, target)
        total_loss += loss.item()
        total_snr += metrics["snr"]
        total_psnr += metrics["psnr"]

    n = len(loader)
    return {
        "loss": total_loss / n,
        "snr": total_snr / n,
        "psnr": total_psnr / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a U-Net for seismic data interpolation / reconstruction"
    )
    # Data
    p.add_argument("--train_data", default=None,
                   help="Path to training .npy or .segy file (or comma-separated list)")
    p.add_argument("--val_data", default=None,
                   help="Path to validation .npy or .segy file (or comma-separated list)")
    p.add_argument("--missing_ratio", type=float, default=0.5,
                   help="Fraction of traces to remove (0–1)")
    p.add_argument("--mask_type", choices=["random", "regular"], default="random")
    p.add_argument("--patch_size", type=int, nargs=2, default=None, metavar=("T", "X"),
                   help="Patch height and width, e.g. --patch_size 128 128")

    # Model
    p.add_argument("--base_features", type=int, default=32,
                   help="Base feature maps in the first U-Net encoder stage")
    p.add_argument("--no_bilinear", action="store_true", default=False,
                   help="Use transposed convolutions instead of bilinear upsampling")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", default=False,
                   help="Use automatic mixed precision (AMP) training")
    p.add_argument("--num_workers", type=int, default=4)

    # Output
    p.add_argument("--output_dir", default="./outputs",
                   help="Directory for checkpoints and logs")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a checkpoint every N epochs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    if args.train_data is None or args.val_data is None:
        print("No data paths provided – generating synthetic seismic data …")
        syn_path = _make_synthetic_data()
        # Use 80 / 20 split of the synthetic data
        data = np.load(syn_path)
        n = len(data)
        split = int(0.8 * n)
        np.save("/tmp/seismic_train.npy", data[:split])
        np.save("/tmp/seismic_val.npy", data[split:])
        train_path = "/tmp/seismic_train.npy"
        val_path = "/tmp/seismic_val.npy"
    else:
        train_path = args.train_data.split(",") if "," in args.train_data else args.train_data
        val_path = args.val_data.split(",") if "," in args.val_data else args.val_data

    patch_size = tuple(args.patch_size) if args.patch_size is not None else None

    train_loader, val_loader = build_loaders(
        train_path,
        val_path,
        batch_size=args.batch_size,
        missing_ratio=args.missing_ratio,
        mask_type=args.mask_type,
        patch_size=patch_size,
        num_workers=args.num_workers,
    )
    print(f"Training samples : {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # ---- Model --------------------------------------------------------------
    model = UNet(
        in_channels=2,
        out_channels=1,
        base_features=args.base_features,
        bilinear=not args.no_bilinear,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Optimiser & loss ---------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ReconstructionLoss()
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    # ---- Training loop ------------------------------------------------------
    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_stats['loss']:.4f}  SNR={train_stats['snr']:.2f} dB | "
            f"val loss={val_stats['loss']:.4f}  SNR={val_stats['snr']:.2f} dB | "
            f"{elapsed:.1f}s"
        )
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_snr": train_stats["snr"],
            "train_psnr": train_stats["psnr"],
            "val_loss": val_stats["loss"],
            "val_snr": val_stats["snr"],
            "val_psnr": val_stats["psnr"],
        })

        # Save best model
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(
                model, optimizer, epoch, val_stats["loss"],
                os.path.join(args.output_dir, "best_model.pth"),
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_stats["loss"],
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pth"),
            )

    # ---- Save training log --------------------------------------------------
    log_path = os.path.join(args.output_dir, "training_log.csv")
    import csv
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining complete.  Log saved to {log_path}")


if __name__ == "__main__":
    main()
