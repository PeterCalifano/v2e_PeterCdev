#!/usr/bin/env python3
"""Run a short end-to-end v2e CLI benchmark.

This benchmark is intended for quick throughput comparisons of the full
`v2e.py` pipeline (argument parsing + frame handling + event simulation).
By default it uses a short input clip and disables expensive outputs.

Examples
--------
python scripts/benchmark_v2e_cli.py
python scripts/benchmark_v2e_cli.py --input input/box-moving-2.mp4 --stop_time 1.0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Short end-to-end benchmark for v2e.py")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "input" / "box-moving-white.mp4")
    parser.add_argument("--stop_time", type=float, default=0.5)
    parser.add_argument("--input_frame_rate", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--pos_thres", type=float, default=0.08)
    parser.add_argument("--neg_thres", type=float, default=0.08)
    parser.add_argument("--sigma_thres", type=float, default=0.02)
    parser.add_argument("--cutoff_hz", type=float, default=0.5)
    parser.add_argument("--keep_output", action="store_true")
    parser.add_argument(
        "--print_log_tail", type=int, default=20,
        help="Print the last N lines of v2e stdout/stderr logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    v2e_py = repo_root / "v2e.py"

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not v2e_py.exists():
        raise FileNotFoundError(f"v2e.py not found at: {v2e_py}")

    with tempfile.TemporaryDirectory(prefix="v2e-bench-") as tmp_dir:
        output_dir = Path(tmp_dir) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(v2e_py),
            "-i", str(args.input),
            "--output_folder", str(output_dir),
            "--overwrite",
            "--skip_video_output",
            "--no_preview",
            "--disable_slomo",
            "--stop_time", str(args.stop_time),
            "--input_frame_rate", str(args.input_frame_rate),
            "--output_width", str(args.width),
            "--output_height", str(args.height),
            "--pos_thres", str(args.pos_thres),
            "--neg_thres", str(args.neg_thres),
            "--sigma_thres", str(args.sigma_thres),
            "--cutoff_hz", str(args.cutoff_hz),
        ]

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start

        print("[BENCHMARK] v2e.py short pipeline run")
        print(f"command={' '.join(cmd)}")
        print(f"exit_code={proc.returncode}")
        print(f"runtime_s={elapsed:.4f}")
        print(f"output_dir={output_dir}")

        if args.print_log_tail > 0:
            lines = proc.stdout.splitlines()
            tail = lines[-args.print_log_tail:]
            print(f"\n[LOG TAIL] last {len(tail)} lines")
            for line in tail:
                print(line)

        if proc.returncode != 0:
            raise RuntimeError("v2e benchmark run failed. See log tail above.")

        if args.keep_output:
            keep_dir = repo_root / "output" / "benchmark_v2e_cli_last"
            if keep_dir.exists():
                shutil.rmtree(keep_dir)
            shutil.copytree(output_dir, keep_dir)
            print(f"[BENCHMARK] Copied output to {keep_dir}")


if __name__ == "__main__":
    main()
