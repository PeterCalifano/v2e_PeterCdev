"""End-to-end emulator performance benchmark.

Measures actual emulator throughput with real workload to validate
that micro-optimizations translate to real-world speedup.

Run with:
    pytest test/test_end_to_end_perf.py -v -s
"""
import time
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from v2ecore.emulator import EventEmulator


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WARMUP_FRAMES = 3
_BENCHMARK_FRAMES = 30


def _benchmark_emulator(
    width: int,
    height: int,
    n_frames: int,
    pos_thres: float = 0.15,
    neg_thres: float = 0.15,
    sigma_thres: float = 0.03,
    cutoff_hz: float = 200.0,
    shot_noise_rate_hz: float = 1.0,
) -> tuple[float, int]:
    """Benchmark emulator processing N frames and return (time_seconds, total_events)."""

    emu = EventEmulator(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=sigma_thres,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=shot_noise_rate_hz,
        photoreceptor_noise=False,
        refractory_period_s=0.001,  # 1ms refractory
        seed=42,
        output_width=width,
        output_height=height,
        device=_DEVICE,
    )

    # Generate synthetic frames with varying content
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n_frames):
        # Mix of gradients, noise, and motion
        base = int(128 + 64 * np.sin(i * 0.5))
        frame = np.full((height, width), base, dtype=np.uint8)
        # Add some spatial structure
        for _ in range(5):
            y, x = rng.integers(0, height), rng.integers(0, width)
            r = rng.integers(10, 30)
            val = rng.integers(0, 256)
            y1, y2 = max(0, y-r), min(height, y+r)
            x1, x2 = max(0, x-r), min(width, x+r)
            frame[y1:y2, x1:x2] = val
        frames.append(frame)

    # Warmup
    frame_idx = 0
    for i in range(min(_WARMUP_FRAMES, n_frames)):
        emu.generate_events(frames[i % len(frames)], frame_idx * (1.0 / 60.0))
        frame_idx += 1

    if _DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark (continue from where warmup left off)
    t0 = time.perf_counter()
    total_events = 0
    for i in range(n_frames):
        events = emu.generate_events(frames[i % len(frames)], frame_idx * (1.0 / 60.0))
        if events is not None:
            total_events += len(events)
        frame_idx += 1

    if _DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    emu.cleanup()

    return elapsed, total_events


class TestEndToEndPerformance:
    """Measure real-world emulator throughput."""

    def test_davis346_resolution(self):
        """DAVIS346 (346x260) at 60 FPS equivalent."""
        width, height = 346, 260
        elapsed, total_events = _benchmark_emulator(
            width, height, _BENCHMARK_FRAMES,
            pos_thres=0.15, neg_thres=0.15, sigma_thres=0.03,
            cutoff_hz=200.0, shot_noise_rate_hz=1.0,
        )

        frames_per_sec = _BENCHMARK_FRAMES / elapsed
        events_per_sec = total_events / elapsed
        ms_per_frame = (elapsed / _BENCHMARK_FRAMES) * 1000

        print(f"\n  DAVIS346 (346x260) Emulator Performance:")
        print(f"    Frames:         {_BENCHMARK_FRAMES}")
        print(f"    Total events:   {total_events:,}")
        print(f"    Elapsed:        {elapsed:.3f} s")
        print(f"    Throughput:     {frames_per_sec:.1f} frames/s")
        print(f"    Event rate:     {events_per_sec:,.0f} events/s")
        print(f"    Per-frame:      {ms_per_frame:.2f} ms/frame")
        print(f"    Device:         {_DEVICE}")

    def test_davis240_resolution(self):
        """DAVIS240 (240x180) at 60 FPS equivalent."""
        width, height = 240, 180
        elapsed, total_events = _benchmark_emulator(
            width, height, _BENCHMARK_FRAMES,
            pos_thres=0.15, neg_thres=0.15, sigma_thres=0.03,
            cutoff_hz=200.0, shot_noise_rate_hz=1.0,
        )

        frames_per_sec = _BENCHMARK_FRAMES / elapsed
        events_per_sec = total_events / elapsed
        ms_per_frame = (elapsed / _BENCHMARK_FRAMES) * 1000

        print(f"\n  DAVIS240 (240x180) Emulator Performance:")
        print(f"    Frames:         {_BENCHMARK_FRAMES}")
        print(f"    Total events:   {total_events:,}")
        print(f"    Elapsed:        {elapsed:.3f} s")
        print(f"    Throughput:     {frames_per_sec:.1f} frames/s")
        print(f"    Event rate:     {events_per_sec:,.0f} events/s")
        print(f"    Per-frame:      {ms_per_frame:.2f} ms/frame")
        print(f"    Device:         {_DEVICE}")

    def test_high_event_rate_scenario(self):
        """High event rate: low thresholds + high noise."""
        width, height = 346, 260
        elapsed, total_events = _benchmark_emulator(
            width, height, _BENCHMARK_FRAMES,
            pos_thres=0.1,  # Lower threshold → more events
            neg_thres=0.1,
            sigma_thres=0.05,
            cutoff_hz=300.0,
            shot_noise_rate_hz=5.0,  # Higher noise
        )

        frames_per_sec = _BENCHMARK_FRAMES / elapsed
        events_per_sec = total_events / elapsed
        ms_per_frame = (elapsed / _BENCHMARK_FRAMES) * 1000

        print(f"\n  High Event Rate Scenario (346x260):")
        print(f"    Frames:         {_BENCHMARK_FRAMES}")
        print(f"    Total events:   {total_events:,}")
        print(f"    Elapsed:        {elapsed:.3f} s")
        print(f"    Throughput:     {frames_per_sec:.1f} frames/s")
        print(f"    Event rate:     {events_per_sec:,.0f} events/s")
        print(f"    Per-frame:      {ms_per_frame:.2f} ms/frame")
        print(f"    Device:         {_DEVICE}")
