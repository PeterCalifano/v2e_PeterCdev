# Comparative Event-Stream Benchmark

This benchmark suite compares event-stream quality and runtime across the
optimized baseline and new error-model feature sets in this branch.

## Goals

- Compare runtime and event-stream behavior under a single representative test
  case designed to expose timing and burst-placement differences.
- Compare these profiles:
  - `baseline_optimized`
  - `v2ce_random`
  - `v2ce_slope`
  - `iebcs_base_nofile`
  - `all_features_nofile` (optional, enabled by default)
- Produce machine-readable outputs (`JSON`, `CSV`, `NPZ`) and comparative
  visualizations.

## Why This Stimulus

The synthetic scenario is inspired by the IEBCS sensitivity protocol and V2CE
burst-timestamp analysis goals.

- IEBCS reports square-wave stimulation at `10 Hz` and evaluates sensitivity
  versus contrast amplitude ramps up to `0.5 log` units.
- V2CE highlights timestamp-layering artifacts from even/random timestamp
  placement and motivates better temporal placement quality metrics.

The implemented stimulus uses:

- `10 Hz` square-wave temporal modulation,
- monotonic ramp from `0` to `0.5 log_e` units,
- a moving edge and drifting texture to create burst-heavy transitions.

## Implemented Profiles

All profiles run on the same generated frames and timestamps.

### 1) `baseline_optimized`

Current optimized v2e behavior with all new IEBCS/V2CE extensions disabled.

### 2) `v2ce_random`

- `--v2ce_nonuniform_burst_timestamps true`
- `--v2ce_burst_timestamps_mode random`

### 3) `v2ce_slope`

- `--v2ce_nonuniform_burst_timestamps true`
- `--v2ce_burst_timestamps_mode slope`

### 4) `iebcs_base_nofile`

- Stage-1 + no-file Stage-2 IEBCS features:
  - latency + jitter,
  - threshold resampling on signal events,
  - contrast-latency model,
  - refractory-state coupling.
- Histogram noise is explicitly disabled.

### 5) `all_features_nofile`

All IEBCS and V2CE flags enabled except histogram-noise.

## Metrics

### Runtime metrics

- `runtime_s`
- `fps_effective`
- `events_total`
- `events_per_frame`

### Stream-shape metrics

- `on_events`, `off_events`, `on_off_ratio`
- `timestamp_layering_score` (unique timestamp ratio after microsecond
  quantization)
- per-pixel normalized event-rate map

### Pairwise (vs baseline)

- `delta_events_pct`
- `delta_on_off_ratio`
- `timestamp_w1_us`
- `rate_map_l1`
- `layering_delta`

### Pseudo-reference

A high-FPS synthetic reference run (same scenario with finer timestep) is used
as a proxy reference:

- `METE_like_us` (timestamp distribution distance)
- `NOE_pct` (number-of-events error)
- `GPER_like_l1` (rate-map distance)

## Visualizations

The benchmark generates:

1. `runtime_vs_profile.png`
   - Mean runtime and FPS with confidence intervals across runs.
2. `quality_vs_runtime_pareto.png`
   - Runtime-quality tradeoff scatter.
3. `timestamp_cdf_overlay.png`
   - Timestamp ECDF/CDF overlay for all profiles plus pseudo-reference.
4. `event_rate_map_diff_heatmaps.png`
   - Per-pixel event-rate map differences (`profile - baseline`).
5. `metric_radar_or_groupedbars.png`
   - Normalized grouped metric view.
6. `all_features_nofile_delta_panel.png`
   - Focused runtime + quality deltas for all-features-nohist profile.

## CLI

Main script:

```bash
python scripts/benchmark_error_models_eventstream.py --help
```

Key options:

- `--make_plots / --no-make_plots` (default: make plots)
- `--plot_format {png,pdf,both}` (default: `png`)
- `--quality_metric_for_pareto` (default: `timestamp_w1_us`)
- `--include_all_features_nofile / --no-include_all_features_nofile`
- Standard controls: `--device`, `--width`, `--height`, `--duration_s`, `--fps`,
  `--runs`, `--warmup_runs`, `--seed`, `--reference_factor`, `--output_dir`.

## Examples

### 1) Quick smoke run (CPU)

```bash
python scripts/benchmark_error_models_eventstream.py \
  --device cpu \
  --width 128 --height 96 \
  --duration_s 0.3 --fps 40 \
  --runs 1 --warmup_runs 0
```

### 2) Full comparative run (auto device)

```bash
python scripts/benchmark_error_models_eventstream.py \
  --device auto \
  --width 346 --height 260 \
  --duration_s 1.0 --fps 120 \
  --runs 3 --warmup_runs 1 \
  --reference_factor 8
```

### 3) Plot-only regeneration from saved report

```bash
python scripts/benchmark_error_models_eventstream.py \
  --plot_only_from_json output/benchmarks_comparative/benchmark_error_models_eventstream_YYYYMMDD_HHMMSS_xxxxxx.json \
  --output_dir output/benchmarks_comparative/replots
```

If `artifacts_npz` is not available in the JSON report, pass `--artifacts_npz`.

## Output Files

Per run, the script writes:

- `benchmark_error_models_eventstream_<timestamp>.json`
- `benchmark_error_models_eventstream_<timestamp>.csv`
- `benchmark_error_models_eventstream_<timestamp>_artifacts.npz`
- plot files in the chosen format

## Reproducibility Checklist

- Fix `--seed`.
- Keep same `--device` and `--cuda_sync` behavior.
- Keep same `--width`, `--height`, `--fps`, `--duration_s`.
- Keep same `--runs` and `--warmup_runs`.
- Keep same `--reference_factor`.

## Limitations

- The pseudo-reference is synthetic high-FPS output, not hardware GT.
- Absolute metric values depend on stimulus and parameterization.
- Cross-device runtime comparisons (CPU vs GPU) are not directly comparable.

## References

- Joubert et al., *Event Camera Simulator Improvements via Characterized
  Parameters*, Frontiers in Neuroscience, 2021.
- V2CE paper/project resources for event timestamp placement evaluation.
