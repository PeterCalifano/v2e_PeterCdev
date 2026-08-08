"""Smoke and output tests for comparative event-stream benchmark suite."""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("seaborn")

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


def _Run_small_benchmark(*, include_all_features_nofile: bool) -> tuple[dict, object]:
    config = BenchmarkConfig(
        width=48,
        height=36,
        duration_s=0.2,
        fps=30.0,
        reference_factor=2,
        seed=7,
        device="cpu",
        runs=2,
        warmup_runs=1,
        cuda_sync=False,
        include_all_features_nofile=include_all_features_nofile,
    )
    return Run_comparative_benchmark(config)


def test_comparative_benchmark_smoke_runs_all_expected_profiles() -> None:
    report, _ = _Run_small_benchmark(include_all_features_nofile=True)

    profile_names = [profile["name"] for profile in report["profiles"]]
    assert profile_names == [
        "baseline_optimized",
        "v2ce_random",
        "v2ce_slope",
        "iebcs_base_nofile",
        "all_features_nofile",
    ]

    assert len(report["pairwise_vs_baseline"]) == len(profile_names) - 1
    assert len(report["pseudo_reference"]["metrics"]) == len(profile_names)

    for profile in report["profiles"]:
        summary = profile["summary"]
        assert summary["runtime_mean_s"] > 0
        assert summary["fps_mean"] > 0
        for run in profile["runs"]:
            assert run["timestamp_monotonic"] is True


def test_outputs_and_plots_are_generated_and_nonempty(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        width=40,
        height=30,
        duration_s=0.2,
        fps=25.0,
        reference_factor=2,
        seed=9,
        device="cpu",
        runs=1,
        warmup_runs=0,
        cuda_sync=False,
        include_all_features_nofile=True,
    )
    report, artifacts = Run_comparative_benchmark(config)

    json_path = tmp_path / "comparative_report.json"
    csv_path = tmp_path / "comparative_summary.csv"
    npz_path = tmp_path / "comparative_artifacts.npz"

    Save_artifacts_npz(artifacts, npz_path)
    report["artifacts_npz"] = str(npz_path)
    Write_report_json(report, json_path)
    Write_summary_csv(report, csv_path)

    loaded_report = Read_report_json(json_path)
    loaded_artifacts = Build_artifacts_from_report_and_npz(loaded_report, npz_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_paths = Make_comparative_plots(
            report=loaded_report,
            artifacts=loaded_artifacts,
            output_dir=tmp_path,
            plot_format="png",
            quality_metric_for_pareto="timestamp_w1_us",
        )

    expected_keys = {
        "runtime_vs_profile",
        "quality_vs_runtime_pareto",
        "timestamp_cdf_overlay",
        "event_rate_map_diff_heatmaps",
        "metric_radar_or_groupedbars",
        "all_features_nofile_delta_panel",
    }
    assert set(plot_paths.keys()) == expected_keys

    assert json_path.exists() and json_path.stat().st_size > 0
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert npz_path.exists() and npz_path.stat().st_size > 0

    for files in plot_paths.values():
        assert len(files) == 1
        plot_file = Path(files[0])
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0
