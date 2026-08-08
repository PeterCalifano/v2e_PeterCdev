"""Build comparative event-stream benchmark visualizations.

Empty streams and undefined metrics are valid benchmark results. Plot builders
therefore preserve the complete figure set while omitting non-finite points or
rendering an explicit empty-data annotation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from v2ecore.benchmarks.eventstream_compare import BenchmarkArtifacts


sns.set_theme(style="whitegrid", context="talk")


def _Save_figure(*,
                 fig: plt.Figure,
                 output_dir: Path,
                 stem: str,
                 plot_format: str,
                 ) -> list[str]:
    """Save a Matplotlib figure in one or multiple formats and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_format not in {"png", "pdf", "both"}:
        raise ValueError("plot_format must be one of: png, pdf, both")

    suffixes = [".png", ".pdf"] if plot_format == "both" else [
        f".{plot_format}"]
    paths: list[str] = []
    for suffix in suffixes:
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _Get_pairwise_row(report: dict[str, Any], profile_name: str) -> dict[str, Any] | None:
    """Return pairwise-vs-baseline row for a profile, if available."""
    for row in report.get("pairwise_vs_baseline", []):
        if row.get("profile") == profile_name:
            return row
    return None


def _Get_reference_row(report: dict[str, Any], profile_name: str) -> dict[str, Any] | None:
    """Return pseudo-reference metrics row for a profile, if available."""
    for row in report.get("pseudo_reference", {}).get("metrics", []):
        if row.get("profile") == profile_name:
            return row
    return None


def _Plot_runtime_vs_profile(report: dict[str, Any]) -> plt.Figure:
    """Plot runtime and effective-FPS bars across all runs per profile."""
    runtime_x: list[str] = []
    runtime_y: list[float] = []
    fps_x: list[str] = []
    fps_y: list[float] = []

    for profile in report.get("profiles", []):
        name = profile["name"]
        for run in profile.get("runs", []):
            runtime_x.append(name)
            runtime_y.append(float(run["runtime_s"]))
            fps_x.append(name)
            fps_y.append(float(run["fps_effective"]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=runtime_x, y=runtime_y, estimator=np.mean,
                errorbar=("ci", 95), ax=axes[0])
    sns.barplot(x=fps_x, y=fps_y, estimator=np.mean,
                errorbar=("ci", 95), ax=axes[1])

    axes[0].set_title("Runtime by Profile")
    axes[0].set_ylabel("runtime_s")
    axes[0].set_xlabel("profile")

    axes[1].set_title("Effective FPS by Profile")
    axes[1].set_ylabel("fps_effective")
    axes[1].set_xlabel("profile")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Comparative Runtime Summary", y=1.02)
    fig.tight_layout()
    return fig


def _Plot_quality_vs_runtime_pareto(report: dict[str, Any],
                                    quality_metric_for_pareto: str,
                                    ) -> plt.Figure:
    """Plot finite runtime-versus-quality pairs from comparison metrics."""
    names: list[str] = []
    x_runtime: list[float] = []
    y_quality: list[float] = []

    baseline_name = "baseline_optimized"
    for profile in report.get("profiles", []):
        name = profile["name"]
        summary = profile.get("summary", {})
        runtime = float(summary.get("runtime_mean_s", 0.0))

        pair = _Get_pairwise_row(report, name)
        ref = _Get_reference_row(report, name)

        # Metric lookup priority: baseline anchor -> pairwise -> reference -> fallback.
        if name == baseline_name:
            quality = 0.0
        elif pair is not None and quality_metric_for_pareto in pair:
            quality = float(pair[quality_metric_for_pareto])
        elif ref is not None and quality_metric_for_pareto in ref:
            quality = float(ref[quality_metric_for_pareto])
        elif pair is not None and "timestamp_w1_us" in pair:
            quality = float(pair["timestamp_w1_us"])
        else:
            quality = float("nan")

        names.append(name)
        x_runtime.append(runtime)
        y_quality.append(quality)

    # Undefined quality metrics are valid for empty streams. Omit only those
    # points rather than passing NaNs into Matplotlib's coordinate machinery.
    finite_points_ = [
        (name_, runtime_, quality_)
        for name_, runtime_, quality_ in zip(names, x_runtime, y_quality)
        if np.isfinite(runtime_) and np.isfinite(quality_)
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    if finite_points_:
        finite_names_, finite_runtime_, finite_quality_ = zip(*finite_points_)
        sns.scatterplot(
            x=finite_runtime_, y=finite_quality_, hue=finite_names_, s=140,
            ax=ax)
        for name_, runtime_, quality_ in finite_points_:
            ax.text(
                runtime_, quality_, f" {name_}", va="center", ha="left")
    else:
        ax.text(
            0.5, 0.5, "No finite quality metrics available",
            ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Quality vs Runtime Pareto View")
    ax.set_xlabel("runtime_mean_s")
    ax.set_ylabel(quality_metric_for_pareto)
    fig.tight_layout()
    return fig


def _Plot_timestamp_cdf_overlay(report: dict[str, Any],
                                artifacts: BenchmarkArtifacts,
                                ) -> plt.Figure:
    """Overlay available timestamp CDFs or annotate an empty event set."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for profile in report.get("profiles", []):
        name = profile["name"]
        events = artifacts.comparison_events[name]
        if events.size == 0:
            continue
        ts = np.sort(events[:, 0])
        cdf = np.linspace(0.0, 1.0, ts.size, dtype=np.float64)
        ax.plot(ts, cdf, label=name, linewidth=1.8)

    if artifacts.reference_events.size > 0:
        ts_ref = np.sort(artifacts.reference_events[:, 0])
        cdf_ref = np.linspace(0.0, 1.0, ts_ref.size, dtype=np.float64)
        ax.plot(ts_ref, cdf_ref, label="pseudo_reference",
                linewidth=2.2, linestyle="--", color="black")

    ax.set_title("Event Timestamp CDF Overlay")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("cdf")
    handles_, _labels_ = ax.get_legend_handles_labels()
    if handles_:
        ax.legend(loc="lower right", fontsize=10)
    else:
        ax.text(
            0.5, 0.5, "No events available",
            ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def _Plot_event_rate_map_diff_heatmaps(report: dict[str, Any],
                                       artifacts: BenchmarkArtifacts,
                                       ) -> plt.Figure:
    """Plot per-profile event-rate-map differences relative to baseline."""
    baseline = artifacts.comparison_event_rate_maps["baseline_optimized"]
    names = [p["name"] for p in report.get(
        "profiles", []) if p["name"] != "baseline_optimized"]

    if not names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No non-baseline profiles available",
                ha="center", va="center")
        ax.axis("off")
        return fig

    diffs = [artifacts.comparison_event_rate_maps[name] -
             baseline for name in names]
    # Use a symmetric color scale to preserve sign interpretation across subplots.
    vmax = max(float(np.max(np.abs(d))) for d in diffs)
    vmax = max(vmax, 1e-9)

    ncols = 2
    nrows = int(math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        14, 4.8 * nrows), squeeze=False)

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if idx >= len(names):
                ax.axis("off")
                continue
            diff = diffs[idx]
            sns.heatmap(
                diff,
                cmap="coolwarm",
                center=0.0,
                vmin=-vmax,
                vmax=vmax,
                ax=ax,
                cbar=True,
                xticklabels=False,
                yticklabels=False,
            )
            ax.set_title(f"{names[idx]} - baseline")
            idx += 1

    fig.suptitle("Per-pixel Event-rate Map Differences", y=1.01)
    fig.tight_layout()
    return fig


def _Plot_metric_groupedbars(report: dict[str, Any]) -> plt.Figure:
    """Plot normalized metrics, mapping undefined metric values to zero."""
    metrics = [
        "runtime_mean_s",
        "timestamp_w1_us",
        "rate_map_l1",
        "METE_like_us",
        "NOE_pct",
    ]

    records_x: list[str] = []
    records_y: list[float] = []
    records_profile: list[str] = []

    values_by_metric: dict[str, list[float]] = {
        metric: [] for metric in metrics}
    profile_names = [p["name"] for p in report.get("profiles", [])]

    for profile_name in profile_names:
        summary = next(p["summary"]
                       for p in report["profiles"] if p["name"] == profile_name)
        pair = _Get_pairwise_row(report, profile_name)
        ref = _Get_reference_row(report, profile_name)

        metric_values = {
            "runtime_mean_s": float(summary.get("runtime_mean_s", 0.0)),
            "timestamp_w1_us": float(pair.get("timestamp_w1_us", 0.0)) if pair else 0.0,
            "rate_map_l1": float(pair.get("rate_map_l1", 0.0)) if pair else 0.0,
            "METE_like_us": float(ref.get("METE_like_us", 0.0)) if ref else 0.0,
            "NOE_pct": float(ref.get("NOE_pct", 0.0)) if ref else 0.0,
        }
        for key, value in metric_values.items():
            values_by_metric[key].append(value)

    normalized: dict[str, list[float]] = {}
    for metric, values in values_by_metric.items():
        # Min-max normalize each metric to [0, 1] for a shared visual scale.
        arr_ = np.asarray(values, dtype=np.float64)
        finite_arr_ = arr_[np.isfinite(arr_)]
        if finite_arr_.size == 0:
            normalized[metric] = [0.0 for _ in values]
        else:
            lo_ = float(np.min(finite_arr_))
            hi_ = float(np.max(finite_arr_))
            if hi_ - lo_ < 1e-12:
                normalized[metric] = [0.0 for _ in values]
            else:
                normalized[metric] = [
                    float((value_ - lo_) / (hi_ - lo_))
                    if np.isfinite(value_) else 0.0
                    for value_ in values
                ]

    for profile_idx, profile_name in enumerate(profile_names):
        for metric in metrics:
            records_x.append(metric)
            records_y.append(normalized[metric][profile_idx])
            records_profile.append(profile_name)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=records_x, y=records_y, hue=records_profile, ax=ax)
    ax.set_title("Normalized Comparative Metric Overview")
    ax.set_ylabel("normalized score [0,1]")
    ax.set_xlabel("metric")
    ax.legend(title="profile", fontsize=9)
    fig.tight_layout()
    return fig


def _Plot_all_features_delta_panel(report: dict[str, Any]) -> plt.Figure:
    """Plot two-panel view for baseline vs all_features_nofile deltas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline = next((p for p in report.get("profiles", [])
                    if p["name"] == "baseline_optimized"), None)
    all_features = next((p for p in report.get("profiles", [])
                        if p["name"] == "all_features_nofile"), None)
    pair = _Get_pairwise_row(report, "all_features_nofile")
    ref = _Get_reference_row(report, "all_features_nofile")

    if baseline is None or all_features is None:
        # Keep plot generation robust even when optional profile is not present.
        for ax in axes:
            ax.text(0.5, 0.5, "all_features_nofile profile not present",
                    ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return fig

    runtime_names = ["baseline_optimized", "all_features_nofile"]
    runtime_mean = [
        float(baseline["summary"].get("runtime_mean_s", 0.0)),
        float(all_features["summary"].get("runtime_mean_s", 0.0)),
    ]
    runtime_std = [
        float(baseline["summary"].get("runtime_std_s", 0.0)),
        float(all_features["summary"].get("runtime_std_s", 0.0)),
    ]
    axes[0].bar(runtime_names, runtime_mean, yerr=runtime_std,
                capsize=4, color=["#4c78a8", "#f58518"])
    axes[0].set_title("Runtime Delta Panel")
    axes[0].set_ylabel("runtime_s")
    axes[0].tick_params(axis="x", rotation=20)

    delta_labels = ["delta_events_pct", "timestamp_w1_us",
                    "rate_map_l1", "METE_like_us", "NOE_pct"]
    delta_vals = [
        float(pair.get("delta_events_pct", 0.0)) if pair else 0.0,
        float(pair.get("timestamp_w1_us", 0.0)) if pair else 0.0,
        float(pair.get("rate_map_l1", 0.0)) if pair else 0.0,
        float(ref.get("METE_like_us", 0.0)) if ref else 0.0,
        float(ref.get("NOE_pct", 0.0)) if ref else 0.0,
    ]
    sns.barplot(x=delta_labels, y=delta_vals, ax=axes[1], color="#72b7b2")
    axes[1].set_title("all_features_nofile vs baseline deltas")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    return fig


def Make_comparative_plots(*,
                           report: dict[str, Any],
                           artifacts: BenchmarkArtifacts,
                           output_dir: Path,
                           plot_format: str,
                           quality_metric_for_pareto: str,
                           ) -> dict[str, list[str]]:
    """Create and save all requested comparative benchmark plots.

    Returns a mapping from logical plot name to emitted file path list.
    """
    plots: dict[str, list[str]] = {}

    fig_runtime = _Plot_runtime_vs_profile(report)
    plots["runtime_vs_profile"] = _Save_figure(
        fig=fig_runtime,
        output_dir=output_dir,
        stem="runtime_vs_profile",
        plot_format=plot_format,
    )

    fig_pareto = _Plot_quality_vs_runtime_pareto(
        report=report,
        quality_metric_for_pareto=quality_metric_for_pareto,
    )
    plots["quality_vs_runtime_pareto"] = _Save_figure(
        fig=fig_pareto,
        output_dir=output_dir,
        stem="quality_vs_runtime_pareto",
        plot_format=plot_format,
    )

    fig_cdf = _Plot_timestamp_cdf_overlay(report=report, artifacts=artifacts)
    plots["timestamp_cdf_overlay"] = _Save_figure(
        fig=fig_cdf,
        output_dir=output_dir,
        stem="timestamp_cdf_overlay",
        plot_format=plot_format,
    )

    fig_heatmaps = _Plot_event_rate_map_diff_heatmaps(
        report=report, artifacts=artifacts)
    plots["event_rate_map_diff_heatmaps"] = _Save_figure(
        fig=fig_heatmaps,
        output_dir=output_dir,
        stem="event_rate_map_diff_heatmaps",
        plot_format=plot_format,
    )

    fig_group = _Plot_metric_groupedbars(report=report)
    plots["metric_radar_or_groupedbars"] = _Save_figure(
        fig=fig_group,
        output_dir=output_dir,
        stem="metric_radar_or_groupedbars",
        plot_format=plot_format,
    )

    fig_panel = _Plot_all_features_delta_panel(report=report)
    plots["all_features_nofile_delta_panel"] = _Save_figure(
        fig=fig_panel,
        output_dir=output_dir,
        stem="all_features_nofile_delta_panel",
        plot_format=plot_format,
    )

    return plots
