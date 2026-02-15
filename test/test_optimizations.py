"""Unit tests for performance optimizations in emulator_utils, emulator, and v2e_utils.

Tests verify that optimized code paths produce identical results to the
original implementations.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from v2ecore.emulator_utils import (
    Map_linear_to_log_luminance,
    LowPassFilter,
    compute_event_map,
)
from v2ecore.v2e_utils import hist2d_numba_seq, hist2d_numba_parallel, hist2d_numba
from v2ecore.emulator import EventEmulator


# ---------------------------------------------------------------------------
# Map_linear_to_log_luminance tests
# ---------------------------------------------------------------------------

class TestLinLog:
    """Verify Map_linear_to_log_luminance behaviour and precision properties."""

    def test_output_dtype_is_float32(self):
        x = torch.randint(0, 256, (16, 16), dtype=torch.float32)
        y = Map_linear_to_log_luminance(x)
        assert y.dtype == torch.float32

    def test_known_values(self):
        """Below threshold → linear scaled; above → log."""
        x = torch.tensor([1.0, 10.0, 20.0, 100.0, 255.0])
        y = Map_linear_to_log_luminance(x, threshold=20)
        # Below threshold: y = x * (1/threshold) * ln(threshold)
        import math
        f = (1.0 / 20) * math.log(20)
        assert torch.allclose(y[:3], torch.tensor([1 * f, 10 * f, 20 * f], dtype=torch.float32), atol=1e-5)
        # Above threshold: y = ln(x)
        assert torch.allclose(y[3:], torch.tensor([math.log(100), math.log(255)], dtype=torch.float32), atol=1e-5)

    def test_monotonicity(self):
        """Map_linear_to_log_luminance should be monotonically increasing."""
        x = torch.arange(1, 256, dtype=torch.float32)
        y = Map_linear_to_log_luminance(x)
        assert torch.all(torch.diff(y) >= 0)

    def test_continuity_at_threshold(self):
        """Values just below and at threshold should be very close."""
        t = 20.0
        x = torch.tensor([t - 0.01, t, t + 0.01])
        y = Map_linear_to_log_luminance(x, threshold=t)
        # Check no discontinuity
        assert (y[2] - y[0]).abs() < 0.01


# ---------------------------------------------------------------------------
# LowPassFilter in-place tests
# ---------------------------------------------------------------------------

class TestLowPassFilterInPlace:
    """Verify the in-place IIR filter produces correct results."""

    def _reference_lpf(self, log_new, lp_log, eps):
        """Non-in-place reference: lp + eps * (new - lp)."""
        return lp_log + eps * (log_new - lp_log)

    def test_scalar_eps_path(self):
        """inten01=None → scalar eps, fully in-place."""
        lpf = LowPassFilter(cutoff_hz=200.0)
        log_new = torch.rand(32, 32)
        lp_log = torch.rand(32, 32)
        lp_log_copy = lp_log.clone()

        delta_time = 0.001
        tau = lpf.tau
        eps = delta_time / tau

        expected = self._reference_lpf(log_new, lp_log_copy, eps)
        result = lpf(log_new, lp_log, inten01=None, delta_time=delta_time)

        # result should be the same object as lp_log (in-place)
        assert result.data_ptr() == lp_log.data_ptr()
        assert torch.allclose(result, expected, atol=1e-6)

    def test_tensor_eps_path(self):
        """inten01 provided → per-pixel eps, fused expression (non-in-place
        because PyTorch fuses the expression into fewer CUDA kernels)."""
        lpf = LowPassFilter(cutoff_hz=200.0)
        log_new = torch.rand(32, 32)
        lp_log = torch.rand(32, 32)
        lp_log_copy = lp_log.clone()
        inten01 = torch.rand(32, 32)

        delta_time = 0.001
        tau = lpf.tau
        eps = torch.clamp(inten01 * (delta_time / tau), max=1.0)

        expected = self._reference_lpf(log_new, lp_log_copy, eps)
        result = lpf(log_new, lp_log, inten01=inten01, delta_time=delta_time)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_eps_clamped(self):
        """When eps > 1, it should be clamped to 1.0."""
        lpf = LowPassFilter(cutoff_hz=200.0)
        log_new = torch.ones(8, 8) * 5.0
        lp_log = torch.zeros(8, 8)
        inten01 = torch.ones(8, 8)  # max intensity

        # Use a large delta_time to make eps > 1
        result = lpf(log_new, lp_log, inten01=inten01, delta_time=10.0)

        # With eps clamped to 1.0: result = lp + 1.0*(new - lp) = new
        assert torch.allclose(result, log_new, atol=1e-6)

    def test_negative_cutoff_returns_input(self):
        """cutoff_hz=0 should return log_new_frame unchanged."""
        lpf = LowPassFilter(cutoff_hz=0.0)
        log_new = torch.rand(8, 8)
        lp_log = torch.rand(8, 8)
        result = lpf(log_new, lp_log, inten01=None, delta_time=0.001)
        assert torch.equal(result, log_new)


# ---------------------------------------------------------------------------
# compute_event_map tests
# ---------------------------------------------------------------------------

class TestComputeEventMap:
    """Verify event map quantization."""

    def test_basic_event_counts(self):
        pos_thres = torch.tensor([[0.2]])
        neg_thres = torch.tensor([[0.2]])
        diff = torch.tensor([[0.5]])
        pos_evts, neg_evts = compute_event_map(diff, pos_thres, neg_thres)
        # 0.5 / 0.2 = 2.5, floor = 2
        assert pos_evts.item() == 2
        assert neg_evts.item() == 0

    def test_negative_diff(self):
        pos_thres = torch.tensor([[0.2]])
        neg_thres = torch.tensor([[0.2]])
        diff = torch.tensor([[-0.5]])
        pos_evts, neg_evts = compute_event_map(diff, pos_thres, neg_thres)
        assert pos_evts.item() == 0
        # 0.5 / 0.2 = 2.5, floor = 2
        assert neg_evts.item() == 2

    def test_zero_diff_no_events(self):
        pos_thres = torch.full((4, 4), 0.2)
        neg_thres = torch.full((4, 4), 0.2)
        diff = torch.zeros(4, 4)
        pos_evts, neg_evts = compute_event_map(diff, pos_thres, neg_thres)
        assert pos_evts.sum().item() == 0
        assert neg_evts.sum().item() == 0

    def test_output_dtype_int32(self):
        diff = torch.rand(8, 8) * 0.5
        pos_thres = torch.full((8, 8), 0.1)
        neg_thres = torch.full((8, 8), 0.1)
        pos_evts, neg_evts = compute_event_map(diff, pos_thres, neg_thres)
        assert pos_evts.dtype == torch.int32
        assert neg_evts.dtype == torch.int32


# ---------------------------------------------------------------------------
# hist2d parallel tests
# ---------------------------------------------------------------------------

class TestHist2dParallel:
    """Verify parallel histogram matches sequential version."""

    def test_identical_output(self):
        rng = np.random.default_rng(42)
        n_tracks = 10_000
        height, width = 64, 64
        tracks = np.stack([
            rng.uniform(0, height, n_tracks),
            rng.uniform(0, width, n_tracks),
        ]).astype(np.float64)
        bins = np.array([height, width], dtype=np.int64)
        ranges = np.array([[0, height], [0, width]], dtype=np.int64)

        h_seq = hist2d_numba_seq(tracks, bins, ranges)
        h_par = hist2d_numba_parallel(tracks, bins, ranges)

        np.testing.assert_array_equal(h_seq, h_par)

    def test_empty_tracks(self):
        tracks = np.zeros((2, 0), dtype=np.float64)
        bins = np.array([16, 16], dtype=np.int64)
        ranges = np.array([[0, 16], [0, 16]], dtype=np.int64)

        h_seq = hist2d_numba_seq(tracks, bins, ranges)
        h_par = hist2d_numba_parallel(tracks, bins, ranges)

        np.testing.assert_array_equal(h_seq, h_par)
        assert h_seq.sum() == 0

    def test_auto_select_small(self):
        """Small track count → should use sequential."""
        tracks = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        bins = np.array([10, 10], dtype=np.int64)
        ranges = np.array([[0, 10], [0, 10]], dtype=np.int64)
        h = hist2d_numba(tracks, bins, ranges)
        assert h.sum() == 2

    def test_auto_select_large(self):
        """Large track count → should use parallel (still correct)."""
        rng = np.random.default_rng(99)
        n = 100_000
        tracks = np.stack([
            rng.uniform(0, 32, n),
            rng.uniform(0, 32, n),
        ]).astype(np.float64)
        bins = np.array([32, 32], dtype=np.int64)
        ranges = np.array([[0, 32], [0, 32]], dtype=np.int64)
        h = hist2d_numba(tracks, bins, ranges)
        assert h.sum() == n


# ---------------------------------------------------------------------------
# Event generation sanity tests (not tied to specific optimization)
# ---------------------------------------------------------------------------

def _assert_event_packet_valid(events: np.ndarray, height: int, width: int) -> None:
    assert events is not None
    assert events.ndim == 2
    assert events.shape[1] == 4
    assert np.all(np.diff(events[:, 0]) >= 0)
    assert np.all(events[:, 1] >= 0)
    assert np.all(events[:, 1] < width)
    assert np.all(events[:, 2] >= 0)
    assert np.all(events[:, 2] < height)
    assert set(np.unique(events[:, 3])).issubset({-1.0, 1.0})


class TestEventGenerationSanity:
    """Basic sanity tests for event generation (not specific to buffer optimization)."""

    def test_basic_event_generation(self):
        height, width = 24, 32
        emu = EventEmulator(
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.0,
            cutoff_hz=0.0,
            leak_rate_hz=0.0,
            shot_noise_rate_hz=0.0,
            photoreceptor_noise=False,
            refractory_period_s=0.0,
            seed=42,
            output_width=width,
            output_height=height,
            device="cpu",
        )

        frame_0 = np.zeros((height, width), dtype=np.uint8)
        frame_1 = np.full((height, width), 255, dtype=np.uint8)

        assert emu.generate_events(frame_0, 0.0) is None
        events = emu.generate_events(frame_1, 1.0 / 30.0)
        emu.cleanup()

        _assert_event_packet_valid(events, height=height, width=width)

    def test_zero_contrast_no_signal_events(self):
        """Same frame twice should produce zero signal events."""
        height, width = 16, 16
        emu = EventEmulator(
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.0,
            cutoff_hz=0.0,
            leak_rate_hz=0.0,
            shot_noise_rate_hz=0.0,
            photoreceptor_noise=False,
            refractory_period_s=0.0,
            seed=42,
            output_width=width,
            output_height=height,
            device="cpu",
        )

        frame = np.full((height, width), 128, dtype=np.uint8)
        emu.generate_events(frame, 0.0)
        events = emu.generate_events(frame, 1.0 / 30.0)
        emu.cleanup()

        assert events is None
