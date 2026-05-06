"""
Dataset utilities for seismic data interpolation / reconstruction.

Supports loading seismic data from:
  - NumPy .npy files  (shape: [time, trace] or [n_samples, time, trace])
  - SEG-Y files via the `segyio` library (optional dependency)

A random subsampling mask is applied during training to simulate missing
traces (irregular or regular undersampling).
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def random_mask(n_traces: int, missing_ratio: float, seed: int = None) -> np.ndarray:
    """
    Return a 1-D binary mask of shape (n_traces,).

    1 = trace present, 0 = trace missing.

    Parameters
    ----------
    n_traces : int
        Total number of traces.
    missing_ratio : float
        Fraction of traces to remove (0 < missing_ratio < 1).
    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    mask = np.ones(n_traces, dtype=np.float32)
    n_missing = int(n_traces * missing_ratio)
    idx = rng.choice(n_traces, size=n_missing, replace=False)
    mask[idx] = 0.0
    return mask


def regular_mask(n_traces: int, keep_every: int = 2) -> np.ndarray:
    """
    Return a 1-D binary mask that keeps every *keep_every*-th trace.

    Parameters
    ----------
    n_traces : int
        Total number of traces.
    keep_every : int
        Decimation factor (2 = keep 50 %, 4 = keep 25 %, …).
    """
    mask = np.zeros(n_traces, dtype=np.float32)
    mask[::keep_every] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class SeismicDataset(Dataset):
    """
    PyTorch Dataset for seismic interpolation / reconstruction.

    Each sample is a 2-D seismic section (time × traces).  The dataset
    applies a random subsampling mask at sample time, returning:

        ``data``   – incomplete section (masked traces zeroed out)
        ``mask``   – binary mask  (1 = present, 0 = missing)
        ``target`` – full (ground-truth) section

    Parameters
    ----------
    data_path : str or list of str
        Path to a .npy file that contains a 2-D array of shape
        ``(n_time, n_traces)`` or a 3-D array of shape
        ``(n_samples, n_time, n_traces)``.  Alternatively, pass a list
        of paths so that multiple files are concatenated along the
        sample axis.
    missing_ratio : float
        Fraction of traces to randomly mask during training (0–1).
    mask_type : {"random", "regular"}
        Strategy used to generate the subsampling mask.
    keep_every : int
        Used only when ``mask_type == "regular"``.
    patch_size : tuple of int, optional
        If provided, each section is split into non-overlapping patches of
        shape ``(patch_time, patch_traces)``.
    normalize : bool
        If True, each section is normalised to zero mean / unit variance.
    transform : callable, optional
        Additional transform applied to the sample dict after masking.
    """

    def __init__(
        self,
        data_path: Union[str, list],
        missing_ratio: float = 0.5,
        mask_type: str = "random",
        keep_every: int = 2,
        patch_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.missing_ratio = missing_ratio
        self.mask_type = mask_type
        self.keep_every = keep_every
        self.patch_size = patch_size
        self.normalize = normalize
        self.transform = transform

        # Load data
        paths = [data_path] if isinstance(data_path, str) else data_path
        arrays = [self._load(p) for p in paths]
        data = np.concatenate(arrays, axis=0)   # (N, T, X)
        self.data = data.astype(np.float32)

        # Optionally split into patches
        if patch_size is not None:
            self.data = self._patchify(self.data, patch_size)

        self.n_samples, self.n_time, self.n_traces = self.data.shape

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".npy":
            arr = np.load(path)
            if arr.ndim == 2:
                arr = arr[np.newaxis]      # (1, T, X)
            elif arr.ndim != 3:
                raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")
            return arr
        elif ext in (".segy", ".sgy"):
            return SeismicDataset._load_segy(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_segy(path: str) -> np.ndarray:
        try:
            import segyio
        except ImportError as exc:
            raise ImportError(
                "segyio is required to read SEG-Y files.  "
                "Install it with:  pip install segyio"
            ) from exc
        with segyio.open(path, ignore_geometry=True) as f:
            data = segyio.tools.collect(f.trace[:])   # (n_traces, n_time)
        return data.T[np.newaxis]   # (1, n_time, n_traces)

    @staticmethod
    def _patchify(data: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
        """Split each section into non-overlapping patches."""
        n, t, x = data.shape
        pt, px = patch_size
        t_patches = t // pt
        x_patches = x // px
        patches = []
        for i in range(n):
            for ti in range(t_patches):
                for xi in range(x_patches):
                    patch = data[i,
                                 ti * pt:(ti + 1) * pt,
                                 xi * px:(xi + 1) * px]
                    patches.append(patch)
        return np.stack(patches, axis=0)   # (N_patches, pt, px)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        section = self.data[idx].copy()   # (T, X)

        # Normalise
        if self.normalize:
            std = section.std()
            if std > 0:
                section = (section - section.mean()) / std

        # Build mask  (1-D over trace axis → broadcast to (T, X))
        if self.mask_type == "random":
            mask_1d = random_mask(self.n_traces, self.missing_ratio)
        else:
            mask_1d = regular_mask(self.n_traces, self.keep_every)

        mask_2d = np.broadcast_to(mask_1d[np.newaxis, :], section.shape).copy()

        # Apply mask
        masked = section * mask_2d

        # Convert to tensors and add channel dimension → (1, T, X)
        target = torch.from_numpy(section[np.newaxis])
        data_tensor = torch.from_numpy(masked[np.newaxis])
        mask_tensor = torch.from_numpy(mask_2d[np.newaxis])

        sample = {
            "data": data_tensor,      # (1, T, X)  incomplete input
            "mask": mask_tensor,      # (1, T, X)  binary mask
            "target": target,         # (1, T, X)  ground truth
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_loaders(
    train_path: Union[str, list],
    val_path: Union[str, list],
    batch_size: int = 8,
    missing_ratio: float = 0.5,
    mask_type: str = "random",
    patch_size: Optional[Tuple[int, int]] = None,
    num_workers: int = 4,
    **dataset_kwargs,
):
    """
    Return (train_loader, val_loader) DataLoader pair.

    Parameters
    ----------
    train_path, val_path : str or list of str
        Paths to training / validation .npy or .segy files.
    batch_size : int
        Mini-batch size.
    missing_ratio : float
        Fraction of traces to mask.
    mask_type : str
        "random" or "regular".
    patch_size : tuple, optional
        Patch dimensions ``(n_time, n_traces)``.
    num_workers : int
        DataLoader worker processes.
    **dataset_kwargs
        Extra keyword arguments forwarded to :class:`SeismicDataset`.
    """
    from torch.utils.data import DataLoader

    train_ds = SeismicDataset(
        train_path,
        missing_ratio=missing_ratio,
        mask_type=mask_type,
        patch_size=patch_size,
        **dataset_kwargs,
    )
    val_ds = SeismicDataset(
        val_path,
        missing_ratio=missing_ratio,
        mask_type=mask_type,
        patch_size=patch_size,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
