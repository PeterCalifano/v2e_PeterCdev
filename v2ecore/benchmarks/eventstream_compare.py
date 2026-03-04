"""Comparative event-stream benchmark utilities for v2e error-model profiles.

This module provides:
- deterministic paper-inspired synthetic stimulus generation,
- profile definitions for baseline/V2CE/IEBCS configurations,
- runtime and event-stream metrics,
- pseudo-reference comparisons,
- report and artifact serialization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from v2ecore.emulator import EventEmulator


@dataclass(frozen=True)
class BenchmarkProfile:
    """Configuration profile for one benchmark run.

    Each profile overrides a subset of `EventEmulator` keyword arguments
    to represent one error-model configuration in the comparative study.
    """

    name: str
    description: str
    emulator_kwargs: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark runtime/configuration controls.

    This captures stimulus geometry, run controls, pseudo-reference settings,
    and device/synchronization behavior.
    """

    width: int = 346
    height: int = 260
    duration_s: float = 1.0
    fps: float = 120.0
    reference_factor: int = 8
    seed: int = 1
    device: str = "auto"
    runs: int = 3
    warmup_runs: int = 1
    cuda_sync: bool = True
    include_all_features_nofile: bool = True


@dataclass
class BenchmarkArtifacts:
    """Binary artifacts needed for visualization and plot regeneration.

    These arrays are intentionally separated from JSON report content to keep
    reports readable while preserving full-resolution numerical data.
    """

    comparison_events: dict[str, np.ndarray]
    comparison_event_rate_maps: dict[str, np.ndarray]
    reference_events: np.ndarray


def Choose_device(requested: str) -> str:
    """Resolve device string from requested mode."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available")
    return requested


def Build_common_emulator_kwargs(*,
    width: int,
    height: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Build shared emulator kwargs across benchmark profiles.

    Profile-specific options are layered on top of this baseline dictionary.
    """
    return {
        "pos_thres": 0.20,
        "neg_thres": 0.20,
        "sigma_thres": 0.03,
        "cutoff_hz": 1.0,
        "leak_rate_hz": 0.0,
        "shot_noise_rate_hz": 0.0,
        "photoreceptor_noise": False,
        "refractory_period_s": 0.0005,
        "seed": seed,
        "output_width": width,
        "output_height": height,
        "device": device,
    }


def Build_profiles(*, include_all_features_nofile: bool) -> list[BenchmarkProfile]:
    """Return the configured comparison profiles.

    The profile set is fixed and intentionally explicit to keep benchmark
    comparisons reproducible across branches and machines.
    """
    profiles: list[BenchmarkProfile] = [
        BenchmarkProfile(
            name="baseline_optimized",
            description="Optimized v2e baseline; all IEBCS/V2CE extensions disabled.",
            emulator_kwargs={
                "v2ce_nonuniform_burst_timestamps": False,
                "iebcs_latency_jitter_model": False,
                "iebcs_resample_thresholds_on_event": False,
                "iebcs_contrast_latency_model": False,
                "iebcs_refractory_state_coupling": False,
                "iebcs_hist_noise_model": False,
            },
        ),
        BenchmarkProfile(
            name="v2ce_random",
            description="V2CE-inspired non-uniform timestamps (random mode).",
            emulator_kwargs={
                "v2ce_nonuniform_burst_timestamps": True,
                "v2ce_burst_timestamps_mode": "random",
                "iebcs_latency_jitter_model": False,
                "iebcs_resample_thresholds_on_event": False,
                "iebcs_contrast_latency_model": False,
                "iebcs_refractory_state_coupling": False,
                "iebcs_hist_noise_model": False,
            },
        ),
        BenchmarkProfile(
            name="v2ce_slope",
            description="V2CE-inspired non-uniform timestamps (slope mode).",
            emulator_kwargs={
                "v2ce_nonuniform_burst_timestamps": True,
                "v2ce_burst_timestamps_mode": "slope",
                "iebcs_latency_jitter_model": False,
                "iebcs_resample_thresholds_on_event": False,
                "iebcs_contrast_latency_model": False,
                "iebcs_refractory_state_coupling": False,
                "iebcs_hist_noise_model": False,
            },
        ),
        BenchmarkProfile(
            name="iebcs_base_nofile",
            description=(
                "IEBCS Stage-1 + no-file Stage-2 base features: latency+jitter, "
                "threshold-resample, contrast-latency, refractory coupling; histogram disabled."
            ),
            emulator_kwargs={
                "iebcs_latency_jitter_model": True,
                "iebcs_latency_mean_us": 100.0,
                "iebcs_latency_jitter_us": 30.0,
                "iebcs_resample_thresholds_on_event": True,
                "iebcs_contrast_latency_model": True,
                "iebcs_latency_tau_us": 300.0,
                "iebcs_latency_clamp_us": 10000.0,
                "iebcs_latency_slope_jitter": True,
                "iebcs_refractory_state_coupling": True,
                "iebcs_refractory_us": 700.0,
                "iebcs_hist_noise_model": False,
                "v2ce_nonuniform_burst_timestamps": False,
            },
        ),
    ]
    if include_all_features_nofile:
        profiles.append(
            BenchmarkProfile(
                name="all_features_nofile",
                description=(
                    "All V2CE+IEBCS flags enabled except IEBCS histogram-noise. "
                    "This is a stress profile for combined extensions without external files."
                ),
                emulator_kwargs={
                    "iebcs_latency_jitter_model": True,
                    "iebcs_latency_mean_us": 100.0,
                    "iebcs_latency_jitter_us": 30.0,
                    "iebcs_resample_thresholds_on_event": True,
                    "iebcs_contrast_latency_model": True,
                    "iebcs_latency_tau_us": 300.0,
                    "iebcs_latency_clamp_us": 10000.0,
                    "iebcs_latency_slope_jitter": True,
                    "iebcs_refractory_state_coupling": True,
                    "iebcs_refractory_us": 700.0,
                    "iebcs_hist_noise_model": False,
                    "v2ce_nonuniform_burst_timestamps": True,
                    "v2ce_burst_timestamps_mode": "slope",
                },
            )
        )
    return profiles


def Generate_paper_inspired_sequence(*,
    width: int,
    height: int,
    duration_s: float,
    fps: float,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate a deterministic synthetic sequence inspired by IEBCS protocol.

    Design goals:
    - temporal square-wave modulation at 10 Hz,
    - monotonic amplitude ramp from 0 to 0.5 log_e units,
    - structured motion via moving edge and drifting texture.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    rng = np.random.default_rng(seed)
    # We generate N+1 timestamps so each interval boundary has an explicit frame.
    num_intervals = max(1, int(round(duration_s * fps)))
    timestamps = np.linspace(0.0, duration_s, num_intervals + 1, dtype=np.float64)

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_idx = np.arange(width, dtype=np.float32)[None, :]

    phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_y = float(rng.uniform(0.0, 2.0 * np.pi))

    frames: list[np.ndarray] = []
    for t in timestamps:
        # Temporal modulation: signed square-wave with increasing logarithmic amplitude.
        ramp_log = 0.5 * (t / duration_s)
        square_sign = 1.0 if np.sin(2.0 * np.pi * 10.0 * t) >= 0 else -1.0
        gain = float(np.exp(square_sign * ramp_log))

        edge_pos = width * (0.1 + 0.8 * (t / duration_s))
        edge_mask = (x_idx >= edge_pos).astype(np.float32)

        drift_x = 0.45 * t / duration_s
        drift_y = 0.35 * t / duration_s
        texture = (
            0.52
            + 0.22 * np.sin(2.0 * np.pi * (x + drift_x) + phase_x)
            + 0.18 * np.cos(2.0 * np.pi * (y - drift_y) + phase_y)
        )

        radial = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        vignette = 1.0 - 0.25 * np.clip(radial / 0.75, 0.0, 1.0)

        frame_float = texture * vignette
        frame_float = frame_float * (1.0 + edge_mask * (gain - 1.0))
        frame_float = np.clip(frame_float, 0.0, 1.0)
        frames.append(np.round(frame_float * 255.0).astype(np.uint8))

    return frames, timestamps


def Compute_event_rate_map(*,
    events: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Compute normalized per-pixel event-count map.

    Returned map sums to 1.0 when events are present and valid.
    """
    rate_map = np.zeros((height, width), dtype=np.float32)
    if events.size == 0:
        return rate_map

    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return rate_map

    np.add.at(rate_map, (y[valid], x[valid]), 1.0)
    total = float(rate_map.sum())
    if total > 0:
        rate_map /= total
    return rate_map


def Approx_timestamp_w1_us(ts_a: np.ndarray,
    ts_b: np.ndarray,
    *,
    num_quantiles: int = 2048,
) -> float:
    """Approximate 1D Wasserstein distance in microseconds using quantiles.

    This avoids a full optimal-transport solve while preserving a robust
    distributional distance for timestamp comparisons.
    """
    if ts_a.size == 0 or ts_b.size == 0:
        return float("nan")

    q_count = max(16, min(num_quantiles, ts_a.size, ts_b.size))
    q = np.linspace(0.0, 1.0, q_count, dtype=np.float64)
    qa = np.quantile(ts_a, q)
    qb = np.quantile(ts_b, q)
    return float(np.mean(np.abs(qa - qb)) * 1e6)


def Compute_basic_metrics(*,
    events: np.ndarray,
    runtime_s: float,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[dict[str, float | bool], np.ndarray]:
    """Compute per-run runtime + stream-shape metrics.

    Metrics include speed, polarity balance, temporal layering proxy, and
    normalized spatial event-rate map.
    """
    events_total = int(events.shape[0])
    on_events = int(np.count_nonzero(events[:, 3] > 0)) if events_total > 0 else 0
    off_events = int(np.count_nonzero(events[:, 3] < 0)) if events_total > 0 else 0

    timestamp_monotonic = (
        bool(np.all(np.diff(events[:, 0]) >= -1e-12)) if events_total > 1 else True
    )
    if events_total > 0:
        ts_us = np.round(events[:, 0] * 1e6).astype(np.int64)
        unique_ratio = float(np.unique(ts_us).size / events_total)
    else:
        unique_ratio = 0.0

    event_rate_map = Compute_event_rate_map(events=events, height=height, width=width)

    fps_effective = float(num_frames / runtime_s) if runtime_s > 0 else 0.0
    metrics: dict[str, float | bool] = {
        "runtime_s": float(runtime_s),
        "fps_effective": fps_effective,
        "events_total": float(events_total),
        "events_per_frame": float(events_total / max(num_frames, 1)),
        "event_rate_ev_s": float(events_total / runtime_s) if runtime_s > 0 else 0.0,
        "on_events": float(on_events),
        "off_events": float(off_events),
        "on_off_ratio": float(on_events / max(off_events, 1)),
        "timestamp_layering_score": unique_ratio,
        "timestamp_monotonic": timestamp_monotonic,
    }
    return metrics, event_rate_map


def Run_profile_once(*,
    profile: BenchmarkProfile,
    common_kwargs: dict[str, Any],
    frames: list[np.ndarray],
    timestamps: np.ndarray,
    cuda_sync: bool,
) -> tuple[dict[str, float | bool], np.ndarray, np.ndarray]:
    """Execute one profile over one sequence and return metrics/events/rate-map.

    Runtime can optionally include CUDA synchronization for fairer GPU timing.
    """
    kwargs = dict(common_kwargs)
    kwargs.update(profile.emulator_kwargs)

    emulator = EventEmulator(**kwargs)
    use_cuda_sync = cuda_sync and kwargs["device"] == "cuda"

    packets: list[np.ndarray] = []
    with torch.no_grad():
        # CUDA kernels are asynchronous; synchronize to measure kernel time.
        if use_cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for frame, t_frame in zip(frames, timestamps):
            ev = emulator.generate_events(frame, float(t_frame))
            if ev is not None and ev.shape[0] > 0:
                packets.append(ev)
        if use_cuda_sync:
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    emulator.cleanup()

    if packets:
        events = np.concatenate(packets, axis=0).astype(np.float32, copy=False)
    else:
        events = np.empty((0, 4), dtype=np.float32)

    # Enforce monotonic ordering for downstream comparative metrics.
    if events.shape[0] > 1:
        order = np.argsort(events[:, 0], kind="mergesort")
        events = events[order]

    metrics, rate_map = Compute_basic_metrics(
        events=events,
        runtime_s=elapsed_s,
        num_frames=len(frames),
        height=int(common_kwargs["output_height"]),
        width=int(common_kwargs["output_width"]),
    )
    return metrics, events, rate_map


def _Summarize_runs(runs: list[dict[str, float | bool]], *, warmup_runs: int) -> dict[str, float | int]:
    """Summarize steady-state run metrics after warmup exclusion.

    Warmup clipping avoids startup transients in allocator/caching paths.
    """
    if not runs:
        raise ValueError("runs must not be empty")

    warmup = min(max(0, warmup_runs), max(0, len(runs) - 1))
    steady = runs[warmup:]

    def _stats(key: str) -> tuple[float, float]:
        vals = np.asarray([float(run[key]) for run in steady], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=0))

    runtime_mean, runtime_std = _stats("runtime_s")
    fps_mean, fps_std = _stats("fps_effective")
    events_mean, events_std = _stats("events_total")
    epf_mean, epf_std = _stats("events_per_frame")
    layering_mean, layering_std = _stats("timestamp_layering_score")

    summary = {
        "warmup_runs_applied": warmup,
        "num_steady_runs": len(steady),
        "runtime_mean_s": runtime_mean,
        "runtime_std_s": runtime_std,
        "fps_mean": fps_mean,
        "fps_std": fps_std,
        "events_total_mean": events_mean,
        "events_total_std": events_std,
        "events_per_frame_mean": epf_mean,
        "events_per_frame_std": epf_std,
        "timestamp_layering_score_mean": layering_mean,
        "timestamp_layering_score_std": layering_std,
    }
    return summary


def Compute_pairwise_vs_baseline(*,
    baseline_name: str,
    comparison_events: dict[str, np.ndarray],
    comparison_maps: dict[str, np.ndarray],
    comparison_metrics: dict[str, dict[str, float | bool]],
) -> list[dict[str, float | str]]:
    """Compute pairwise metrics for each profile against baseline.

    The baseline row itself is intentionally excluded from the output table.
    """
    baseline_events = comparison_events[baseline_name]
    baseline_map = comparison_maps[baseline_name]
    baseline_metrics = comparison_metrics[baseline_name]

    rows: list[dict[str, float | str]] = []
    for name, events in comparison_events.items():
        if name == baseline_name:
            continue
        metrics = comparison_metrics[name]
        rate_map = comparison_maps[name]
        rows.append(
            {
                "baseline": baseline_name,
                "profile": name,
                "delta_events_pct": float(
                    100.0
                    * (float(metrics["events_total"]) - float(baseline_metrics["events_total"]))
                    / max(float(baseline_metrics["events_total"]), 1.0)
                ),
                "delta_on_off_ratio": float(metrics["on_off_ratio"])
                - float(baseline_metrics["on_off_ratio"]),
                "timestamp_w1_us": Approx_timestamp_w1_us(events[:, 0], baseline_events[:, 0]),
                "rate_map_l1": float(np.mean(np.abs(rate_map - baseline_map))),
                "layering_delta": float(metrics["timestamp_layering_score"])
                - float(baseline_metrics["timestamp_layering_score"]),
            }
        )
    return rows


def Compute_pseudo_reference_metrics(*,
    comparison_events: dict[str, np.ndarray],
    comparison_maps: dict[str, np.ndarray],
    reference_events: np.ndarray,
    reference_map: np.ndarray,
) -> list[dict[str, float | str]]:
    """Compute profile-to-reference distances.

    The pseudo-reference is generated at higher temporal sampling using the
    baseline profile, then compared against each candidate profile.
    """
    rows: list[dict[str, float | str]] = []
    ref_count = float(reference_events.shape[0])
    for name, events in comparison_events.items():
        count = float(events.shape[0])
        rows.append(
            {
                "profile": name,
                "METE_like_us": Approx_timestamp_w1_us(events[:, 0], reference_events[:, 0]),
                "NOE_pct": float(100.0 * abs(count - ref_count) / max(ref_count, 1.0)),
                "GPER_like_l1": float(np.mean(np.abs(comparison_maps[name] - reference_map))),
            }
        )
    return rows


def Run_comparative_benchmark(config: BenchmarkConfig,
) -> tuple[dict[str, Any], BenchmarkArtifacts]:
    """Run full comparative benchmark and return report + plotting artifacts.

    The selected comparison run per profile is `compare_idx`, derived from
    warmup settings so all profiles are compared using the same run index.
    """
    device = Choose_device(config.device)
    common_kwargs = Build_common_emulator_kwargs(
        width=config.width,
        height=config.height,
        seed=config.seed,
        device=device,
    )
    profiles = Build_profiles(
        include_all_features_nofile=config.include_all_features_nofile
    )

    frames, timestamps = Generate_paper_inspired_sequence(
        width=config.width,
        height=config.height,
        duration_s=config.duration_s,
        fps=config.fps,
        seed=config.seed,
    )

    compare_idx = min(max(0, config.warmup_runs), max(0, config.runs - 1))

    profile_reports: list[dict[str, Any]] = []
    comparison_events: dict[str, np.ndarray] = {}
    comparison_maps: dict[str, np.ndarray] = {}
    comparison_metrics: dict[str, dict[str, float | bool]] = {}

    for profile in profiles:
        run_rows: list[dict[str, float | bool]] = []
        run_events: list[np.ndarray] = []
        run_maps: list[np.ndarray] = []

        for _ in range(config.runs):
            metrics, events, rate_map = Run_profile_once(
                profile=profile,
                common_kwargs=common_kwargs,
                frames=frames,
                timestamps=timestamps,
                cuda_sync=config.cuda_sync,
            )
            run_rows.append(metrics)
            run_events.append(events)
            run_maps.append(rate_map)

        summary = _Summarize_runs(run_rows, warmup_runs=config.warmup_runs)
        # Keep one representative steady-state run for pairwise comparisons and plots.
        comparison_events[profile.name] = run_events[compare_idx]
        comparison_maps[profile.name] = run_maps[compare_idx]
        comparison_metrics[profile.name] = run_rows[compare_idx]

        profile_reports.append(
            {
                "name": profile.name,
                "description": profile.description,
                "emulator_kwargs": profile.emulator_kwargs,
                "runs": run_rows,
                "summary": summary,
                "comparison_run_index": compare_idx,
            }
        )

    baseline_name = "baseline_optimized"
    pairwise = Compute_pairwise_vs_baseline(
        baseline_name=baseline_name,
        comparison_events=comparison_events,
        comparison_maps=comparison_maps,
        comparison_metrics=comparison_metrics,
    )

    reference_frames, reference_timestamps = Generate_paper_inspired_sequence(
        width=config.width,
        height=config.height,
        duration_s=config.duration_s,
        fps=config.fps * float(config.reference_factor),
        seed=config.seed,
    )
    # Pseudo-reference uses baseline settings at higher temporal sampling.
    baseline_profile = next(p for p in profiles if p.name == baseline_name)
    _, reference_events, reference_map = Run_profile_once(
        profile=baseline_profile,
        common_kwargs=common_kwargs,
        frames=reference_frames,
        timestamps=reference_timestamps,
        cuda_sync=config.cuda_sync,
    )
    pseudo_ref = Compute_pseudo_reference_metrics(
        comparison_events=comparison_events,
        comparison_maps=comparison_maps,
        reference_events=reference_events,
        reference_map=reference_map,
    )

    report: dict[str, Any] = {
        "benchmark": "comparative_eventstream_error_models",
        "config": {
            "width": config.width,
            "height": config.height,
            "duration_s": config.duration_s,
            "fps": config.fps,
            "reference_factor": config.reference_factor,
            "seed": config.seed,
            "device": device,
            "runs": config.runs,
            "warmup_runs": config.warmup_runs,
            "cuda_sync": config.cuda_sync,
            "include_all_features_nofile": config.include_all_features_nofile,
            "stimulus": {
                "square_wave_hz": 10.0,
                "amplitude_log_ramp_max": 0.5,
                "description": (
                    "Paper-inspired synthetic: 10Hz square-wave modulation with "
                    "monotonic 0..0.5 log_e amplitude ramp, plus moving edge and drifting texture."
                ),
            },
        },
        "profiles": profile_reports,
        "pairwise_vs_baseline": pairwise,
        "pseudo_reference": {
            "reference_profile": baseline_name,
            "reference_factor": config.reference_factor,
            "metrics": pseudo_ref,
        },
    }

    artifacts = BenchmarkArtifacts(
        comparison_events=comparison_events,
        comparison_event_rate_maps=comparison_maps,
        reference_events=reference_events,
    )
    return report, artifacts


def Save_artifacts_npz(artifacts: BenchmarkArtifacts, output_path: Path) -> None:
    """Persist benchmark artifacts for plot regeneration.

    Arrays are flattened into namespaced NPZ keys for robust round-tripping.
    """
    payload: dict[str, np.ndarray] = {
        "reference_events": artifacts.reference_events,
    }
    for name, events in artifacts.comparison_events.items():
        payload[f"events__{name}"] = events
    for name, rate_map in artifacts.comparison_event_rate_maps.items():
        payload[f"rate_map__{name}"] = rate_map
    np.savez_compressed(output_path, **payload)


def Load_artifacts_npz(input_path: Path) -> BenchmarkArtifacts:
    """Load benchmark artifacts from compressed NPZ.

    Expected keys are `reference_events`, `events__*`, and `rate_map__*`.
    """
    data = np.load(input_path, allow_pickle=False)
    comparison_events: dict[str, np.ndarray] = {}
    comparison_maps: dict[str, np.ndarray] = {}
    reference_events = data["reference_events"]

    for key in data.files:
        if key.startswith("events__"):
            comparison_events[key.split("events__", 1)[1]] = data[key]
        elif key.startswith("rate_map__"):
            comparison_maps[key.split("rate_map__", 1)[1]] = data[key]

    return BenchmarkArtifacts(
        comparison_events=comparison_events,
        comparison_event_rate_maps=comparison_maps,
        reference_events=reference_events,
    )


def Write_report_json(report: dict[str, Any], output_path: Path) -> None:
    """Write report dictionary as pretty JSON."""
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")


def Read_report_json(input_path: Path) -> dict[str, Any]:
    """Read benchmark JSON report."""
    return json.loads(input_path.read_text(encoding="utf-8"))


def Write_summary_csv(report: dict[str, Any], output_path: Path) -> None:
    """Write profile summary table with pairwise/reference metric joins.

    Missing pairwise/reference values are left blank for compatibility with
    downstream spreadsheet and plotting workflows.
    """
    pairwise_map = {
        row["profile"]: row for row in report.get("pairwise_vs_baseline", [])
    }
    pseudo_map = {
        row["profile"]: row for row in report.get("pseudo_reference", {}).get("metrics", [])
    }

    header = [
        "profile",
        "runtime_mean_s",
        "runtime_std_s",
        "fps_mean",
        "events_total_mean",
        "events_per_frame_mean",
        "timestamp_layering_score_mean",
        "delta_events_pct_vs_baseline",
        "timestamp_w1_us_vs_baseline",
        "rate_map_l1_vs_baseline",
        "METE_like_us_vs_reference",
        "NOE_pct_vs_reference",
        "GPER_like_l1_vs_reference",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for profile in report.get("profiles", []):
            name = profile["name"]
            summary = profile["summary"]
            pair = pairwise_map.get(name, {})
            pseudo = pseudo_map.get(name, {})
            writer.writerow(
                {
                    "profile": name,
                    "runtime_mean_s": summary.get("runtime_mean_s", ""),
                    "runtime_std_s": summary.get("runtime_std_s", ""),
                    "fps_mean": summary.get("fps_mean", ""),
                    "events_total_mean": summary.get("events_total_mean", ""),
                    "events_per_frame_mean": summary.get("events_per_frame_mean", ""),
                    "timestamp_layering_score_mean": summary.get(
                        "timestamp_layering_score_mean", ""
                    ),
                    "delta_events_pct_vs_baseline": pair.get("delta_events_pct", ""),
                    "timestamp_w1_us_vs_baseline": pair.get("timestamp_w1_us", ""),
                    "rate_map_l1_vs_baseline": pair.get("rate_map_l1", ""),
                    "METE_like_us_vs_reference": pseudo.get("METE_like_us", ""),
                    "NOE_pct_vs_reference": pseudo.get("NOE_pct", ""),
                    "GPER_like_l1_vs_reference": pseudo.get("GPER_like_l1", ""),
                }
            )


def Build_artifacts_from_report_and_npz(report: dict[str, Any],
    npz_path: Path,
) -> BenchmarkArtifacts:
    """Load artifacts and validate profile alignment with report content.

    Validation prevents silent plot/report mismatches when users combine files
    generated from different benchmark runs.
    """
    artifacts = Load_artifacts_npz(npz_path)
    profile_names = {profile["name"] for profile in report.get("profiles", [])}
    artifact_names = set(artifacts.comparison_events.keys())
    if profile_names != artifact_names:
        missing = profile_names - artifact_names
        extra = artifact_names - profile_names
        raise ValueError(
            "Artifact profiles mismatch report profiles. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return artifacts
