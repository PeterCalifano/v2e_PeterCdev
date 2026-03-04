#!/usr/bin/env python3
"""Unified benchmark for EventEmulator.

Supports:
1) Legacy single-run benchmark (+optional cProfile)
2) Repeated direct benchmark for branch/performance comparisons
"""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime
import io
from pathlib import Path
import pstats
import statistics
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecore.emulator import EventEmulator


def _choose_device(requested: str) -> str:
    """Resolve requested runtime device and validate CUDA availability."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available")
    return requested


def _build_frames(scenario: str, num_frames: int, height: int, width: int,
        motion_amplitude: float) -> list[np.ndarray]:
    """Generate deterministic synthetic frame sequences for benchmarking."""
    x_ramp = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    base = np.repeat(x_ramp, height, axis=0)
    frames: list[np.ndarray] = []

    for i in range(num_frames):
        # Keep scenarios simple and deterministic to isolate emulator runtime changes.
        if scenario == "motion":
            frame = np.roll(base, i % max(1, width // 20), axis=1)
            frame = np.clip(frame + motion_amplitude * np.sin(i * 0.2), 0, 255)
        elif scenario == "flicker":
            delta = motion_amplitude if (i % 2 == 0) else -motion_amplitude
            frame = np.clip(base + delta, 0, 255)
        elif scenario == "step":
            frame = np.where(base > 127, 255, 0).astype(np.float32)
            if i > (num_frames // 2):
                frame = np.flip(frame, axis=1).copy()
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        frames.append(frame.astype(np.uint8))
    return frames


def _create_emulator(args: argparse.Namespace) -> EventEmulator:
    """Create an `EventEmulator` from parsed CLI parameters."""
    return EventEmulator(
        pos_thres=args.pos_thres,
        neg_thres=args.neg_thres,
        sigma_thres=args.sigma_thres,
        cutoff_hz=args.cutoff_hz,
        leak_rate_hz=args.leak_rate_hz,
        shot_noise_rate_hz=args.shot_noise_rate_hz,
        photoreceptor_noise=args.photoreceptor_noise,
        refractory_period_s=args.refractory_period_s,
        seed=args.seed,
        output_width=args.width,
        output_height=args.height,
        device=args.device,
    )


def _run_once_per_frame_timing(args: argparse.Namespace, frames: list[np.ndarray]) -> dict[str, float]:
    """Run one benchmark pass with per-frame wall-clock timing collection."""
    emulator = _create_emulator(args)
    dt = 1.0 / args.fps
    t_frame = 0.0
    per_frame_ms: list[float] = []
    total_events = 0
    nonempty_packets = 0

    with torch.no_grad():
        for frame in frames:
            start = time.perf_counter()
            events = emulator.generate_events(frame, t_frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_frame_ms.append(elapsed_ms)
            if events is not None and events.shape[0] > 0:
                total_events += int(events.shape[0])
                nonempty_packets += 1
            t_frame += dt

    emulator.cleanup()

    runtime_s = sum(per_frame_ms) / 1000.0
    return {
        "runtime_s": runtime_s,
        "frames": float(len(frames)),
        "fps_effective": float(len(frames)) / runtime_s if runtime_s > 0 else 0.0,
        "frame_ms_mean": statistics.fmean(per_frame_ms),
        "frame_ms_median": statistics.median(per_frame_ms),
        "frame_ms_p95": float(np.percentile(np.asarray(per_frame_ms), 95)),
        "events_total": float(total_events),
        "events_per_frame": float(total_events) / len(frames),
        "nonempty_packets": float(nonempty_packets),
    }


def _run_once_wall_timing(args: argparse.Namespace, frames: list[np.ndarray]) -> dict[str, float]:
    """Run one benchmark pass with end-to-end loop wall timing."""
    emulator = _create_emulator(args)
    dt = 1.0 / args.fps
    t_frame = 0.0
    total_events = 0
    nonempty_packets = 0
    do_cuda_sync = args.cuda_sync and args.device == "cuda"

    with torch.no_grad():
        # Optional sync yields more comparable CUDA timing across runs.
        if do_cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for frame in frames:
            events = emulator.generate_events(frame, t_frame)
            if events is not None and events.shape[0] > 0:
                total_events += int(events.shape[0])
                nonempty_packets += 1
            t_frame += dt
        if do_cuda_sync:
            torch.cuda.synchronize()
    runtime_s = time.perf_counter() - start
    emulator.cleanup()

    return {
        "runtime_s": runtime_s,
        "frames": float(len(frames)),
        "fps_effective": float(len(frames)) / runtime_s if runtime_s > 0 else 0.0,
        "events_total": float(total_events),
        "events_per_frame": float(total_events) / len(frames),
        "nonempty_packets": float(nonempty_packets),
    }


def _run_once(args: argparse.Namespace, frames: list[np.ndarray]) -> dict[str, float]:
    """Dispatch one run according to selected timing mode."""
    if args.timing_mode == "per_frame":
        return _run_once_per_frame_timing(args, frames)
    return _run_once_wall_timing(args, frames)


def _format_single_summary(args: argparse.Namespace, stats: dict[str, float]) -> list[str]:
    """Format human-readable single-run summary lines."""
    lines = [
        "[BENCHMARK] EventEmulator single-run summary",
        f"scenario={args.scenario} size={args.width}x{args.height} frames={args.frames}",
        f"device={args.device} timing_mode={args.timing_mode} photoreceptor_noise={args.photoreceptor_noise}",
        f"runtime_s={stats['runtime_s']:.6f}",
        f"fps_effective={stats['fps_effective']:.2f}",
        f"events_total={int(stats['events_total'])}",
        f"events_per_frame={stats['events_per_frame']:.2f}",
        f"nonempty_packets={int(stats['nonempty_packets'])}",
    ]
    if "frame_ms_mean" in stats:
        lines.extend([
            f"frame_ms_mean={stats['frame_ms_mean']:.3f}",
            f"frame_ms_median={stats['frame_ms_median']:.3f}",
            f"frame_ms_p95={stats['frame_ms_p95']:.3f}",
        ])
    return lines


def _format_repeated_summary(args: argparse.Namespace, runs: list[dict[str, float]]) -> list[str]:
    """Format aggregate summary for repeated benchmark runs."""
    runtimes = np.asarray([r["runtime_s"] for r in runs], dtype=np.float64)
    warmup = min(max(args.warmup_runs, 0), max(0, len(runs) - 1))
    # Exclude startup transients in allocator, cache, and JIT-like warmup paths.
    steady = runtimes[warmup:] if warmup > 0 else runtimes
    fps_values = args.frames / steady
    all_events = sorted({int(r["events_total"]) for r in runs})
    lines = [
        "[BENCHMARK] EventEmulator repeated-run summary",
        f"scenario={args.scenario} size={args.width}x{args.height} frames={args.frames}",
        f"device={args.device} timing_mode={args.timing_mode} runs={args.runs} warmup_runs={warmup}",
        f"summary_runtime_mean_s={steady.mean():.6f}",
        f"summary_runtime_median_s={np.median(steady):.6f}",
        f"summary_runtime_p95_s={np.percentile(steady, 95):.6f}",
        f"summary_runtime_min_s={steady.min():.6f}",
        f"summary_runtime_max_s={steady.max():.6f}",
        f"summary_runtime_std_s={statistics.pstdev(steady):.6f}",
        f"summary_fps_mean={fps_values.mean():.2f}",
        "summary_events_total_unique=" + ",".join(map(str, all_events)),
    ]
    if "frame_ms_mean" in runs[0]:
        frame_means = np.asarray([r["frame_ms_mean"] for r in runs], dtype=np.float64)
        steady_frame_means = frame_means[warmup:] if warmup > 0 else frame_means
        lines.append(f"summary_frame_ms_mean={steady_frame_means.mean():.3f}")
    return lines


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark execution and optional profiling."""
    parser = argparse.ArgumentParser(
        description="Unified EventEmulator benchmark (single + repeated).")
    parser.add_argument("--benchmark_mode", choices=["single", "repeated"], default="single")
    parser.add_argument("--timing_mode", choices=["per_frame", "wall"], default="per_frame")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--print_per_run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda_sync", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--scenario", choices=["motion", "flicker", "step"], default="motion")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--motion_amplitude", type=float, default=25.0)

    parser.add_argument("--pos_thres", type=float, default=0.08)
    parser.add_argument("--neg_thres", type=float, default=0.08)
    parser.add_argument("--sigma_thres", type=float, default=0.02)
    parser.add_argument("--cutoff_hz", type=float, default=0.5)
    parser.add_argument("--leak_rate_hz", type=float, default=0.01)
    parser.add_argument("--shot_noise_rate_hz", type=float, default=0.001)
    parser.add_argument("--refractory_period_s", type=float, default=0.0005)
    parser.add_argument(
        "--photoreceptor_noise", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_top", type=int, default=30)
    parser.add_argument(
        "--profile_dump", type=Path, default=None,
        help="Optional .prof output path for offline analysis.")
    parser.add_argument(
        "--log_dir", type=Path, default=REPO_ROOT / "output" / "benchmarks",
        help="Directory where timestamped benchmark logs are stored.")
    return parser.parse_args()


def _run_single(args: argparse.Namespace, frames: list[np.ndarray]) -> tuple[list[str], list[str]]:
    """Execute single-run benchmark path, optionally under `cProfile`."""
    profile_lines: list[str] = []
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        stats = _run_once(args, frames)
        profiler.disable()

        sio = io.StringIO()
        pstats.Stats(profiler, stream=sio).sort_stats("cumtime").print_stats(args.profile_top)
        profile_lines.append("\n[PROFILE] Top cumulative functions")
        profile_lines.append(sio.getvalue())
        if args.profile_dump is not None:
            profiler.dump_stats(str(args.profile_dump))
            profile_lines.append(f"[PROFILE] Saved raw profile: {args.profile_dump}")
    else:
        stats = _run_once(args, frames)
    return profile_lines, _format_single_summary(args, stats)


def _run_repeated(args: argparse.Namespace, frames: list[np.ndarray]) -> list[str]:
    """Execute repeated-run benchmark path and return formatted summary lines."""
    if args.runs < 1:
        raise ValueError("--runs must be >= 1 for repeated benchmark mode")

    runs: list[dict[str, float]] = []
    for run_idx in range(args.runs):
        run_stats = _run_once(args, frames)
        runs.append(run_stats)
        if args.print_per_run:
            print(
                f"run={run_idx + 1} runtime_s={run_stats['runtime_s']:.6f} "
                f"fps_effective={run_stats['fps_effective']:.2f} "
                f"events_total={int(run_stats['events_total'])}")
    return _format_repeated_summary(args, runs)


def main() -> None:
    """CLI entry point for unified emulator benchmark script."""
    args = parse_args()
    args.device = _choose_device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frames = _build_frames(
        scenario=args.scenario,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        motion_amplitude=args.motion_amplitude,
    )

    if args.benchmark_mode == "single":
        profile_lines, summary_lines = _run_single(args, frames)
        lines = profile_lines + summary_lines
    else:
        if args.profile:
            raise ValueError("--profile is only supported with --benchmark_mode single")
        lines = _run_repeated(args, frames)

    for line in lines:
        print(line)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"benchmark_emulator_{timestamp}.log"
    log_lines = [f"timestamp={timestamp}", f"command_args={args}", *lines]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[BENCHMARK] Saved log to {log_path}")


if __name__ == "__main__":
    main()
