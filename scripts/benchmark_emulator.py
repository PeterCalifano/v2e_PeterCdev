#!/usr/bin/env python3
"""Benchmark and profile the EventEmulator core loop.

Examples
--------
python scripts/benchmark_emulator.py
python scripts/benchmark_emulator.py --scenario flicker --frames 200 --profile
python scripts/benchmark_emulator.py --width 346 --height 260 --device cpu
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecore.emulator import EventEmulator


def _choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available")
    return requested


def _build_frames(
        scenario: str, num_frames: int, height: int, width: int) -> list[np.ndarray]:
    x_ramp = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    base = np.repeat(x_ramp, height, axis=0)
    frames: list[np.ndarray] = []

    for i in range(num_frames):
        if scenario == "motion":
            frame = np.roll(base, i % max(1, width // 20), axis=1)
            frame = np.clip(frame + 3.0 * np.sin(i * 0.2), 0, 255)

        elif scenario == "flicker":
            delta = 25 if (i % 2 == 0) else -25
            frame = np.clip(base + delta, 0, 255)

        elif scenario == "step":
            frame = np.where(base > 127, 255, 0).astype(np.float32)
            if i > (num_frames // 2):
                frame = np.flip(frame, axis=1).copy()
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        frames.append(frame.astype(np.uint8))

    return frames


def _run_emulation(
        args: argparse.Namespace, frames: list[np.ndarray]) -> dict[str, float]:
    
    dt = 1.0 / args.fps
    # Create the emulator instance with the specified parameters
    emulator = EventEmulator(
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

    per_frame_ms: list[float] = []
    total_events = 0
    nonempty_packets = 0
    t_frame = 0.0

    with torch.no_grad():
        # Process each frame through the emulator and measure time taken
        for frame in frames:

            start = time.perf_counter()
            events = emulator.generate_events(frame, t_frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_frame_ms.append(elapsed_ms)
            
            if events is not None and events.shape[0] > 0:
                total_events += int(events.shape[0])
                nonempty_packets += 1
            
            t_frame += dt

    # Reset the emulator to flush any remaining events and perform cleanup
    emulator.cleanup()

    runtime_s = sum(per_frame_ms) / 1000.0

    return {
        "runtime_s": runtime_s,
        "frames": float(len(frames)),
        "fps_effective": float(len(frames)) / runtime_s if runtime_s > 0 else 0.0,
        "frame_ms_mean": statistics.fmean(per_frame_ms),
        "frame_ms_median": statistics.median(per_frame_ms),
        "frame_ms_p95": np.percentile(np.asarray(per_frame_ms), 95),
        "events_total": float(total_events),
        "events_per_frame": float(total_events) / len(frames),
        "nonempty_packets": float(nonempty_packets),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and profile EventEmulator performance.")
    parser.add_argument(
        "--scenario", choices=["motion", "flicker", "step"],
        default="motion")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)

    # Core emulator knobs
    parser.add_argument("--pos_thres", type=float, default=0.08)
    parser.add_argument("--neg_thres", type=float, default=0.08)
    parser.add_argument("--sigma_thres", type=float, default=0.02)
    parser.add_argument("--cutoff_hz", type=float, default=0.5)
    parser.add_argument("--leak_rate_hz", type=float, default=0.01)
    parser.add_argument("--shot_noise_rate_hz", type=float, default=0.001)
    parser.add_argument("--refractory_period_s", type=float, default=0.0005)
    parser.add_argument(
        "--photoreceptor_noise", action=argparse.BooleanOptionalAction,
        default=True)

    # Profiling output
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_top", type=int, default=30)
    parser.add_argument(
        "--profile_dump", type=Path, default=None,
        help="Optional .prof output path for offline analysis.")

    return parser.parse_args()


def main() -> None:

    # Parse command-line arguments and choose device
    args = parse_args()
    args.device = _choose_device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build synthetic frames based on the chosen scenario
    frames = _build_frames(
        scenario=args.scenario,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
    )

    if args.profile:

        # Run the emulation with cProfile to collect detailed performance data
        profiler = cProfile.Profile()
        profiler.enable()
        stats = _run_emulation(args, frames)
        profiler.disable()

        sio = io.StringIO()
        pstats.Stats(profiler, stream=sio).sort_stats(
            "cumtime").print_stats(args.profile_top)
        
        print("\n[PROFILE] Top cumulative functions")
        print(sio.getvalue())

        if args.profile_dump is not None:
            profiler.dump_stats(str(args.profile_dump))
            print(f"[PROFILE] Saved raw profile: {args.profile_dump}")
    else:
        stats = _run_emulation(args, frames)

    print("[BENCHMARK] EventEmulator summary")
    print(f"scenario={args.scenario} size={args.width}x{args.height} frames={args.frames}")
    print(f"device={args.device} photoreceptor_noise={args.photoreceptor_noise}")
    print(f"runtime_s={stats['runtime_s']:.4f}")
    print(f"fps_effective={stats['fps_effective']:.2f}")
    print(f"frame_ms_mean={stats['frame_ms_mean']:.3f}")
    print(f"frame_ms_median={stats['frame_ms_median']:.3f}")
    print(f"frame_ms_p95={stats['frame_ms_p95']:.3f}")
    print(f"events_total={int(stats['events_total'])}")
    print(f"events_per_frame={stats['events_per_frame']:.2f}")
    print(f"nonempty_packets={int(stats['nonempty_packets'])}")


if __name__ == "__main__":
    main()
