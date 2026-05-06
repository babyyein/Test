"""
Tests for the U-Net seismic interpolation project.

Run with:
    python -m pytest tests/ -v

These tests are designed to be fast (CPU-only, small tensor sizes) and cover:
  - U-Net forward pass and output shape
  - Mask generators
  - Dataset loading
  - Loss functions
  - Metric calculations
  - Predict-time padding helpers
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unet_model import UNet, DoubleConv, Down, Up, OutConv
from dataset import SeismicDataset, random_mask, regular_mask
from utils import (
    ReconstructionLoss,
    signal_to_noise_ratio,
    peak_signal_to_noise_ratio,
    compute_metrics,
)
from predict import pad_to_divisible, crop_to_original


# ---------------------------------------------------------------------------
# U-Net model tests
# ---------------------------------------------------------------------------

class TestUNet:
    """Tests for the U-Net architecture."""

    @pytest.mark.parametrize("in_ch,out_ch", [(1, 1), (2, 1)])
    def test_output_shape(self, in_ch, out_ch):
        model = UNet(in_channels=in_ch, out_channels=out_ch, base_features=8, bilinear=True)
        x = torch.randn(2, in_ch, 64, 64)
        y = model(x)
        assert y.shape == (2, out_ch, 64, 64), f"Expected (2,{out_ch},64,64), got {y.shape}"

    def test_non_square_input(self):
        """Model should handle non-square spatial dimensions."""
        model = UNet(in_channels=2, out_channels=1, base_features=8)
        x = torch.randn(1, 2, 96, 64)
        y = model(x)
        assert y.shape == (1, 1, 96, 64)

    def test_bilinear_false(self):
        """Transposed-convolution upsampling path."""
        model = UNet(in_channels=2, out_channels=1, base_features=8, bilinear=False)
        x = torch.randn(1, 2, 64, 64)
        y = model(x)
        assert y.shape == (1, 1, 64, 64)

    def test_no_nan_in_output(self):
        model = UNet(in_channels=2, out_channels=1, base_features=8)
        x = torch.randn(2, 2, 64, 64)
        y = model(x)
        assert not torch.isnan(y).any(), "Output contains NaN values"

    def test_gradients_flow(self):
        """Ensure backward pass does not raise and gradients are non-zero."""
        model = UNet(in_channels=2, out_channels=1, base_features=8)
        x = torch.randn(1, 2, 64, 64, requires_grad=False)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grad_sum = sum(p.grad.abs().sum().item()
                       for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0, "Gradients are all zero"

    def test_double_conv_shape(self):
        layer = DoubleConv(4, 8)
        x = torch.randn(2, 4, 32, 32)
        assert layer(x).shape == (2, 8, 32, 32)

    def test_down_shape(self):
        layer = Down(8, 16)
        x = torch.randn(2, 8, 32, 32)
        assert layer(x).shape == (2, 16, 16, 16)

    def test_out_conv_shape(self):
        layer = OutConv(16, 1)
        x = torch.randn(2, 16, 32, 32)
        assert layer(x).shape == (2, 1, 32, 32)


# ---------------------------------------------------------------------------
# Mask generator tests
# ---------------------------------------------------------------------------

class TestMaskGenerators:
    def test_random_mask_shape(self):
        m = random_mask(100, 0.5)
        assert m.shape == (100,)

    def test_random_mask_missing_ratio(self):
        m = random_mask(100, 0.3, seed=0)
        assert abs(m.sum() - 70) <= 1   # ~70 present

    def test_random_mask_all_ones_zero_ratio(self):
        m = random_mask(50, 0.0)
        assert m.sum() == 50

    def test_random_mask_reproducibility(self):
        m1 = random_mask(100, 0.4, seed=42)
        m2 = random_mask(100, 0.4, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_regular_mask_pattern(self):
        m = regular_mask(10, keep_every=2)
        expected = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(m, expected)

    def test_regular_mask_keep_every_1(self):
        m = regular_mask(8, keep_every=1)
        assert m.sum() == 8


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_npy(tmp_path):
    """Write a small synthetic seismic .npy to tmp_path and return its path."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 64, 64)).astype(np.float32)
    p = str(tmp_path / "data.npy")
    np.save(p, data)
    return p


class TestSeismicDataset:
    def test_length(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, missing_ratio=0.5)
        assert len(ds) == 20

    def test_sample_keys(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, missing_ratio=0.5)
        sample = ds[0]
        assert set(sample.keys()) == {"data", "mask", "target"}

    def test_sample_shapes(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, missing_ratio=0.5)
        sample = ds[0]
        assert sample["data"].shape == (1, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)
        assert sample["target"].shape == (1, 64, 64)

    def test_mask_values(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, missing_ratio=0.5)
        sample = ds[0]
        unique = torch.unique(sample["mask"])
        assert set(unique.tolist()).issubset({0.0, 1.0}), "Mask contains values other than 0 and 1"

    def test_masked_traces_are_zero(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, missing_ratio=0.5)
        sample = ds[0]
        # Where mask is 0 the data should be exactly 0
        missing = sample["mask"] == 0
        assert (sample["data"][missing] == 0).all()

    def test_regular_mask(self, synthetic_npy):
        ds = SeismicDataset(synthetic_npy, mask_type="regular", keep_every=2)
        sample = ds[0]
        unique = torch.unique(sample["mask"])
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_2d_npy_input(self, tmp_path):
        """A 2-D .npy file should be treated as a single sample."""
        data = np.random.randn(64, 64).astype(np.float32)
        p = str(tmp_path / "single.npy")
        np.save(p, data)
        ds = SeismicDataset(p, missing_ratio=0.3)
        assert len(ds) == 1

    def test_patch_size(self, tmp_path):
        data = np.random.randn(4, 128, 128).astype(np.float32)
        p = str(tmp_path / "large.npy")
        np.save(p, data)
        ds = SeismicDataset(p, patch_size=(64, 64))
        # 4 sections × (128/64)^2 = 4 × 4 = 16 patches
        assert len(ds) == 16
        assert ds[0]["data"].shape == (1, 64, 64)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestReconstructionLoss:
    def test_zero_loss_for_perfect_prediction(self):
        crit = ReconstructionLoss()
        t = torch.randn(2, 1, 32, 32)
        loss = crit(t, t)
        assert loss.item() < 1e-6

    def test_loss_decreases_toward_target(self):
        crit = ReconstructionLoss()
        t = torch.ones(2, 1, 32, 32)
        far = torch.zeros(2, 1, 32, 32)
        close = torch.full((2, 1, 32, 32), 0.9)
        assert crit(close, t).item() < crit(far, t).item()

    def test_loss_with_mask(self):
        crit = ReconstructionLoss(missing_weight=2.0)
        t = torch.ones(2, 1, 32, 32)
        p = torch.zeros(2, 1, 32, 32)
        mask = torch.zeros(2, 1, 32, 32)          # all missing
        mask_ones = torch.ones(2, 1, 32, 32)      # all present
        # Higher weight on missing traces → larger loss
        loss_missing = crit(p, t, mask)
        loss_present = crit(p, t, mask_ones)
        assert loss_missing.item() > loss_present.item()

    def test_loss_is_tensor(self):
        crit = ReconstructionLoss()
        loss = crit(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0   # scalar


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_snr_perfect(self):
        a = np.ones((32, 32))
        assert signal_to_noise_ratio(a, a) == float("inf")

    def test_snr_positive_for_good_reconstruction(self):
        t = np.random.randn(32, 32)
        p = t + 0.01 * np.random.randn(32, 32)
        assert signal_to_noise_ratio(p, t) > 10

    def test_psnr_perfect(self):
        a = np.random.randn(32, 32)
        assert peak_signal_to_noise_ratio(a, a) == float("inf")

    def test_compute_metrics_keys(self):
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        m = compute_metrics(pred, target)
        assert "snr" in m and "psnr" in m

    def test_compute_metrics_returns_floats(self):
        pred = torch.randn(2, 1, 32, 32)
        target = pred + 0.01 * torch.randn_like(pred)
        m = compute_metrics(pred, target)
        assert isinstance(m["snr"], float)
        assert isinstance(m["psnr"], float)


# ---------------------------------------------------------------------------
# Padding helpers tests (predict.py)
# ---------------------------------------------------------------------------

class TestPaddingHelpers:
    @pytest.mark.parametrize("t,x,div", [
        (100, 100, 16),
        (128, 128, 16),
        (63, 77, 16),
        (1, 1, 16),
    ])
    def test_padded_dims_divisible(self, t, x, div):
        arr = np.zeros((t, x))
        padded, _ = pad_to_divisible(arr, divisor=div)
        assert padded.shape[0] % div == 0
        assert padded.shape[1] % div == 0

    def test_round_trip(self):
        arr = np.random.randn(100, 77).astype(np.float32)
        padded, orig_shape = pad_to_divisible(arr, divisor=16)
        recovered = crop_to_original(padded, orig_shape)
        np.testing.assert_array_equal(recovered, arr)

    def test_already_divisible(self):
        arr = np.zeros((128, 64))
        padded, orig = pad_to_divisible(arr, divisor=16)
        assert padded.shape == (128, 64)
        assert orig == (128, 64)
