"""Micro-benchmarks for the optimised code paths.

Each test exercises one optimised unit, compares it against the *original*
implementation (inlined here as a reference), and prints a speedup ratio.

Run with:
    pytest test/test_perf_benchmarks.py -v -s       # -s to see printed timings
    pytest test/test_perf_benchmarks.py -k lpf -v -s # single benchmark
"""
import math
import time
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from v2ecore.emulator_utils import LowPassFilter, Map_linear_to_log_luminance, compute_event_map
from v2ecore.v2e_utils import hist2d_numba_seq, hist2d_numba_parallel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WARMUP = 5
_REPEATS = 50


def _benchmark(fn: Callable, warmup: int = _WARMUP, repeats: int = _REPEATS) -> float:
    """Return median wall-clock time (seconds) over *repeats* calls."""
    for _ in range(warmup):
        fn()
    if _DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if _DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if _DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _print_result(name: str, t_old: float, t_new: float) -> None:
    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(
        f"\n  {name}:"
        f"\n    old  = {t_old*1e6:10.1f} µs"
        f"\n    new  = {t_new*1e6:10.1f} µs"
        f"\n    speedup = {speedup:.2f}x"
    )


# ===================================================================
# 1. LowPassFilter – in-place vs allocating
# ===================================================================

def _lpf_old_impl(
    log_new_frame: torch.Tensor,
    lp_log_frame: torch.Tensor,
    inten01: torch.Tensor | None,
    delta_time: float,
    tau: float,
) -> torch.Tensor:
    """Original LowPassFilter.__call__ logic (pre-optimization).

    Includes the torch.max(eps) warning check to match real overhead.
    """
    delta_over_tau = delta_time / tau
    if inten01 is not None:
        eps = inten01 * delta_over_tau
        _max_eps = torch.max(eps)  # warning-check overhead (present in both old & new)
        eps = torch.clamp(eps, max=1.0)
    else:
        # Old: created a scalar tensor (small allocation)
        eps = torch.tensor(
            delta_over_tau, dtype=torch.float32, device=log_new_frame.device
        )
    # Old: always created a new output tensor
    return lp_log_frame + eps * (log_new_frame - lp_log_frame)


class TestLowPassFilterBenchmark:
    """Benchmark in-place LPF vs the original allocating version."""

    @pytest.fixture(params=["scalar_eps", "tensor_eps"])
    def setup(self, request):
        H, W = 260, 346  # DAVIS346 resolution
        log_new = torch.rand(H, W, device=_DEVICE)
        inten01 = torch.rand(H, W, device=_DEVICE) if request.param == "tensor_eps" else None
        delta_time = 1.0 / 1000.0
        lpf = LowPassFilter(cutoff_hz=200.0)
        return log_new, inten01, delta_time, lpf, request.param

    def test_lpf_speedup(self, setup):
        log_new, inten01, delta_time, lpf, label = setup

        # --- OLD ---
        def run_old():
            lp = torch.rand_like(log_new)
            _lpf_old_impl(log_new, lp, inten01, delta_time, lpf.tau)

        # --- NEW (in-place) ---
        def run_new():
            lp = torch.rand_like(log_new)
            lpf(log_new, lp, inten01, delta_time)

        t_old = _benchmark(run_old)
        t_new = _benchmark(run_new)
        _print_result(f"LowPassFilter ({label})", t_old, t_new)


# ===================================================================
# 2. Event-buffer pre-allocation vs list-append + torch.cat
# ===================================================================

class TestEventBufferBenchmark:
    """Benchmark pre-allocated buffer vs list-append + cat.

    NOTE: Pre-allocated buffer optimization was REVERTED because torch.cat()
    is highly optimized on CUDA. Results showed 0.51x (2x slower) for buffer.
    This benchmark is kept for reference and future exploration.

    Creates fresh event tensors each iteration to match the real emulator
    pattern where events are generated on-the-fly, not pre-existing.
    """

    @pytest.fixture(params=[10, 50, 200])
    def setup(self, request):
        n_iters = request.param
        events_per_iter = 500
        total = n_iters * events_per_iter
        return n_iters, events_per_iter, total

    def test_event_buffer_speedup(self, setup):
        n_iters, events_per_iter, total = setup

        # --- OLD: list append + torch.cat ---
        def run_old():
            collected = []
            for i in range(n_iters):
                # Create fresh event tensor each iteration (matches real pattern)
                ev = torch.rand(events_per_iter, 4, device=_DEVICE)
                collected.append(ev)
            _ = torch.cat(collected, dim=0)

        # --- NEW: pre-allocated buffer + index write ---
        def run_new():
            buf = torch.empty(total, 4, device=_DEVICE)
            write_idx = 0
            for i in range(n_iters):
                # Create fresh event tensor each iteration (matches real pattern)
                ev = torch.rand(events_per_iter, 4, device=_DEVICE)
                n = ev.shape[0]
                buf[write_idx : write_idx + n] = ev
                write_idx += n
            _ = buf[:write_idx]

        t_old = _benchmark(run_old)
        t_new = _benchmark(run_new)
        _print_result(f"Event buffer (iters={n_iters})", t_old, t_new)


# ===================================================================
# 3. CPU-GPU transfer patterns
# ===================================================================

class TestCpuGpuTransferBenchmark:
    """Benchmark cleaned-up transfer patterns."""

    def test_item_vs_cpu_numpy_item(self):
        """Scalar extraction: .item() vs .cpu().numpy().item()"""
        t = torch.tensor(42.0, device=_DEVICE)

        def run_old():
            _ = t.cpu().numpy().item()

        def run_new():
            _ = t.item()

        t_old = _benchmark(run_old, repeats=200)
        t_new = _benchmark(run_new, repeats=200)
        _print_result(".item() vs .cpu().numpy().item()", t_old, t_new)

    def test_cpu_numpy_vs_cpu_data_numpy(self):
        """Array transfer: .cpu().numpy() vs .cpu().data.numpy()"""
        t = torch.rand(260, 346, device=_DEVICE)

        def run_old():
            _ = np.array(t.cpu().data.numpy())

        def run_new():
            _ = t.cpu().numpy()

        t_old = _benchmark(run_old, repeats=200)
        t_new = _benchmark(run_new, repeats=200)
        _print_result(".cpu().numpy() vs np.array(.cpu().data.numpy())", t_old, t_new)


# ===================================================================
# 4. Histogram: sequential vs parallel
# ===================================================================

class TestHistogramBenchmark:
    """Benchmark hist2d sequential vs parallel."""

    @pytest.fixture(params=[10_000, 100_000, 500_000])
    def setup(self, request):
        n = request.param
        rng = np.random.default_rng(42)
        height, width = 260, 346
        tracks = np.stack(
            [rng.uniform(0, height, n), rng.uniform(0, width, n)]
        ).astype(np.float64)
        bins = np.array([height, width], dtype=np.int64)
        ranges = np.array([[0, height], [0, width]], dtype=np.int64)
        return tracks, bins, ranges, n

    def test_histogram_speedup(self, setup):
        tracks, bins, ranges, n = setup

        # warmup JIT compilation (first call compiles)
        hist2d_numba_seq(tracks[:, :10], bins, ranges)
        hist2d_numba_parallel(tracks[:, :10], bins, ranges)

        def run_seq():
            hist2d_numba_seq(tracks, bins, ranges)

        def run_par():
            hist2d_numba_parallel(tracks, bins, ranges)

        t_seq = _benchmark(run_seq)
        t_par = _benchmark(run_par)
        _print_result(f"hist2d (n={n:,})", t_seq, t_par)


# ===================================================================
# 5. Map_linear_to_log_luminance baseline (for reference, since float64 is kept)
# ===================================================================

class TestLinLogBenchmark:
    """Baseline Map_linear_to_log_luminance timing at DAVIS346 resolution."""

    def test_lin_log_timing(self):
        x = torch.randint(0, 256, (260, 346), dtype=torch.float32, device=_DEVICE)

        def run():
            Map_linear_to_log_luminance(x)

        t = _benchmark(run)
        print(f"\n  Map_linear_to_log_luminance (346x260, {_DEVICE}): {t*1e6:.1f} µs")


# ===================================================================
# 6. compute_event_map baseline
# ===================================================================

class TestComputeEventMapBenchmark:
    """Baseline compute_event_map timing."""

    def test_compute_event_map_timing(self):
        H, W = 260, 346
        diff = (torch.rand(H, W, device=_DEVICE) - 0.5) * 2.0
        pos_thres = torch.full((H, W), 0.2, device=_DEVICE)
        neg_thres = torch.full((H, W), 0.2, device=_DEVICE)

        def run():
            compute_event_map(diff, pos_thres, neg_thres)

        t = _benchmark(run)
        print(f"\n  compute_event_map (346x260, {_DEVICE}): {t*1e6:.1f} µs")
