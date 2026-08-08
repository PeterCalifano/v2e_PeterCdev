#!/usr/bin/env python3
"""Comparative event-stream benchmark + visualization runner.

This script benchmarks baseline/V2CE/IEBCS profiles on a deterministic,
paper-inspired synthetic stimulus and emits machine-readable reports with plots.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecore.benchmarks.eventstream_compare import (
    BenchmarkConfig,
    Build_artifacts_from_report_and_npz,
    Read_report_json,
    Run_comparative_benchmark,
    Save_artifacts_npz,
    Write_report_json,
    Write_summary_csv,
)
from v2ecore.benchmarks.eventstream_plots import Make_comparative_plots


def Parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark and plot-only modes."""
    parser = argparse.ArgumentParser(
        description="Comparative event-stream benchmark (baseline/V2CE/IEBCS + plots)."
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks_comparative",
        help="Directory where JSON/CSV/NPZ reports and plots are stored.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--duration_s", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--reference_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)

    parser.add_argument("--cuda_sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--plot_format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Plot export format.",
    )
    parser.add_argument(
        "--quality_metric_for_pareto",
        type=str,
        default="timestamp_w1_us",
        help="Metric name to use as y-axis in quality-vs-runtime plot.",
    )
    parser.add_argument(
        "--include_all_features_nofile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include extra profile with all V2CE+IEBCS flags except histogram-noise.",
    )

    parser.add_argument(
        "--plot_only_from_json",
        type=Path,
        default=None,
        help="If set, skip benchmark run and regenerate plots from an existing JSON report.",
    )
    parser.add_argument(
        "--artifacts_npz",
        type=Path,
        default=None,
        help="Optional NPZ artifact path for plot-only mode (otherwise read from JSON field artifacts_npz).",
    )
    return parser.parse_args()


def _Resolve_artifacts_path(*, report_path: Path, report: dict, cli_path: Path | None) -> Path:
    """Resolve NPZ artifacts path for plot-only mode.

    Resolution order:
    1) explicit CLI path,
    2) `artifacts_npz` field in report JSON,
    3) report-relative path if stored path is relative.
    """
    if cli_path is not None:
        return cli_path
    raw = report.get("artifacts_npz")
    if raw is None:
        raise ValueError(
            "plot-only mode requires --artifacts_npz or artifacts_npz field in report JSON"
        )
    path = Path(raw)
    if not path.is_absolute():
        if path.exists():
            return path
        path = report_path.parent / path
    return path


def _Print_result_summary(report: dict) -> None:
    """Print a compact per-profile summary to stdout."""
    print("[BENCHMARK] Comparative summary")
    for profile in report.get("profiles", []):
        summary = profile.get("summary", {})
        print(
            f"profile={profile['name']} "
            f"runtime_mean_s={summary.get('runtime_mean_s', 0.0):.6f} "
            f"fps_mean={summary.get('fps_mean', 0.0):.2f} "
            f"events_total_mean={summary.get('events_total_mean', 0.0):.1f}"
        )


def main() -> None:
    """Run comparative benchmark or regenerate plots from an existing report."""
    args = Parse_args()
    # Ensure output location exists for both full benchmark and plot-only flows.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only_from_json is not None:
        # Plot-only flow: reconstruct plotting inputs from serialized report/artifacts.
        report_path = args.plot_only_from_json
        report = Read_report_json(report_path)
        artifacts_path = _Resolve_artifacts_path(
            report_path=report_path,
            report=report,
            cli_path=args.artifacts_npz,
        )
        artifacts = Build_artifacts_from_report_and_npz(report, artifacts_path)

        plot_paths = Make_comparative_plots(
            report=report,
            artifacts=artifacts,
            output_dir=args.output_dir,
            plot_format=args.plot_format,
            quality_metric_for_pareto=args.quality_metric_for_pareto,
        )
        report["plots"] = plot_paths
        # Persist updated plot file list back into the same report JSON.
        Write_report_json(report, report_path)
        print(f"[BENCHMARK] Regenerated plots from {report_path}")
        print(f"[BENCHMARK] Updated report JSON: {report_path}")
        for name, paths in plot_paths.items():
            print(f"plot={name} files={','.join(paths)}")
        return

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    config = BenchmarkConfig(
        width=args.width,
        height=args.height,
        duration_s=args.duration_s,
        fps=args.fps,
        reference_factor=args.reference_factor,
        seed=args.seed,
        device=args.device,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        cuda_sync=args.cuda_sync,
        include_all_features_nofile=args.include_all_features_nofile,
    )

    report, artifacts = Run_comparative_benchmark(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = args.output_dir / f"benchmark_error_models_eventstream_{timestamp}.json"
    csv_path = args.output_dir / f"benchmark_error_models_eventstream_{timestamp}.csv"
    npz_path = args.output_dir / f"benchmark_error_models_eventstream_{timestamp}_artifacts.npz"

    Save_artifacts_npz(artifacts, npz_path)
    report["artifacts_npz"] = str(npz_path.resolve())

    Write_summary_csv(report, csv_path)
    report["summary_csv"] = str(csv_path.resolve())

    if args.make_plots:
        # Plots are optional to support headless/fast benchmark runs.
        plot_paths = Make_comparative_plots(
            report=report,
            artifacts=artifacts,
            output_dir=args.output_dir,
            plot_format=args.plot_format,
            quality_metric_for_pareto=args.quality_metric_for_pareto,
        )
        report["plots"] = plot_paths

    Write_report_json(report, json_path)

    _Print_result_summary(report)
    print(f"[BENCHMARK] Saved report JSON: {json_path}")
    print(f"[BENCHMARK] Saved summary CSV: {csv_path}")
    print(f"[BENCHMARK] Saved artifacts NPZ: {npz_path}")


if __name__ == "__main__":
    main()
