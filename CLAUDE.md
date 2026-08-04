# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

v2e converts conventional video frames into realistic synthetic DVS (Dynamic Vision Sensor) event streams. It models DVS camera physics including intensity-dependent photoreceptor bandwidth, per-pixel threshold variation, refractory periods, and noise. The project uses PyTorch + OpenCV and includes SuperSloMo for temporal upsampling.

## Environment Setup

The maintained repo environment is described by `environment.yml` and currently
uses Python 3.11 with GPU-accelerated PyTorch:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate v2e

# Install v2e in editable mode
python -m pip install -e .
```

If you build the environment manually instead of using `environment.yml`, use
Python 3.11 to match the checked-in dependency set.

When running commands from an automation shell where the environment is not
already activated, use `conda run -n v2e ...`.

Download the pre-trained SuperSloMo model from Google Drive (SuperSloMo39.ckpt, 151 MB) and save to the `input/` folder.

## Common Commands

### Running v2e

```bash
# Basic conversion (uses file chooser if no --input specified)
python v2e.py

# Example conversion with explicit parameters
python v2e.py -i input/tennis.mov \
  --overwrite \
  --timestamp_resolution=.003 \
  --auto_timestamp_resolution=False \
  --dvs_exposure duration 0.005 \
  --output_folder=output/tennis \
  --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 \
  --dvs_aedat2 tennis.aedat \
  --output_width=346 --output_height=260 \
  --stop_time=3 --cutoff_hz=15

# Common DVS camera presets
python v2e.py --dvs346  # Sets output to 346x260 (DAVIS346)
python v2e.py --dvs240  # Sets output to 240x180 (DAVIS240)
python v2e.py --dvs640  # Sets output to 640x480 (DAVIS640)

# Trial run (use --stop_time to limit processing time)
python v2e.py -i input/video.mov --stop_time=3

# Headless mode (no OpenCV preview windows)
python v2e.py --no_preview
```

### Testing

```bash
# Run all tests
conda run -n v2e python -m pytest -q

# Run emulator regression tests only
conda run -n v2e python -m pytest -q test/test_emulator_*.py

# Run error-model / eventstream benchmark smoke tests
conda run -n v2e python -m pytest -q test/test_eventstream_benchmark.py

# Run specific test file
conda run -n v2e python -m pytest -q test/test_io_regressions.py

# Run optimization correctness tests
conda run -n v2e python -m pytest -q test/test_optimizations.py

# Run micro-benchmarks (prints speedup ratios, use -s)
conda run -n v2e python -m pytest -q test/test_perf_benchmarks.py -s

# Run end-to-end throughput benchmarks
conda run -n v2e python -m pytest -q test/test_end_to_end_perf.py -s
```

### Benchmarking and Profiling

```bash
# Core emulator timing
conda run -n v2e python v2ecore/benchmarks/benchmark_emulator.py

# Core emulator profiling with cProfile
conda run -n v2e python v2ecore/benchmarks/benchmark_emulator.py --profile

# Comparative error model benchmark (baseline / V2CE / IEBCS profiles)
conda run -n v2e python v2ecore/benchmarks/benchmark_error_models_eventstream.py --output_dir output/benchmarks

# 3D event visualization example
conda run -n v2e python scripts/plot_events_3d_example.py --scenario moving_blob
```

## Core Architecture

### Pipeline Flow

1. **v2e.py**: Top-level CLI, argument parsing, pipeline orchestration
2. **Frame preprocessing**: Read video, crop, resize, convert to grayscale
3. **SuperSloMo interpolation** (optional): Temporal upsampling via `v2ecore/slomo.py`
4. **Event generation**: `EventEmulator.generate_events(frame, timestamp)` for each frame
5. **Output**: Render event frames and serialize event streams (AEDAT-2.0, HDF5, text, AVI)

### Key Modules (v2ecore/)

- **emulator.py**: Stateful DVS pixel-array emulator - main event generation logic and primary performance hotspot
- **emulator_utils.py**: Low-level numerical kernels (`Map_linear_to_log_luminance`, IIR lowpass filtering, event quantization, noise)
- **renderer.py**: Event-to-frame accumulation, preview display, video export
- **slomo.py**: SuperSloMo temporal interpolation
- **v2e_args.py**: Centralized CLI argument definitions and exposure mode parsing
- **output/**: Event serialization backends (.txt, .aedat2, .aedat4, .h5)

### Event Model Stages

The DVS event model follows the stages documented in
`docs/core_model_mapping.md`. That document uses stable literal `# KEY[...]`
anchors rather than copied line-number indexes.

1. **Lin-log encoding**: Piecewise linear/log brightness mapping (`#KEY[D-LINLOG]`)
2. **Photoreceptor lowpass**: Intensity-dependent IIR filter (`#KEY[E-LPF-*]`)
3. **Contrast differencing**: `DeltaL = L_lp - L_mem` (`#KEY[F-DIFF]`)
4. **Event quantization**: `k+ = floor(max(DeltaL, 0) / theta+)` (`#KEY[F-EVENT-QUANT]`)
5. **Memory reset**: `L_mem += k+ * theta+ - k- * theta-` (`#KEY[F-LMEM-UPDATE]`)
6. **Noise models**: Photoreceptor noise or shot noise (`#KEY[G-*]`)
7. **Refractory filtering**: Per-pixel minimum inter-spike interval (`#KEY[H-REFRACTORY-EXT]`)

Use the line links in `docs/core_model_mapping.md` to jump from the paper model
to implementation locations.

## IEBCS and V2CE Error Model Extensions

All extensions are **opt-in**, **disabled by default**, and experimental. See
`docs/README_error_models_extensions.md` for the current correctness and
reference-equivalence limits.

### IEBCS Stage-1 Extensions

```bash
# Event timestamp latency + jitter
--iebcs_latency_jitter_model true \
--iebcs_latency_mean_us 100 \
--iebcs_latency_jitter_us 30

# Threshold reset noise (re-sample thresholds after events)
--iebcs_resample_thresholds_on_event true
```

### V2CE Extensions

```bash
# Non-uniform intra-frame timestamp placement for event bursts
--v2ce_nonuniform_burst_timestamps true \
--v2ce_burst_timestamps_mode random  # or 'slope'
```

### IEBCS Stage-2 Extensions

```bash
# Contrast-dependent latency
--iebcs_contrast_latency_model true \
--iebcs_latency_tau_us 300

# Histogram-based background noise from measured distributions
--iebcs_hist_noise_model true \
--iebcs_noise_source preset --iebcs_noise_preset 161lux
# Or use explicit files:
--iebcs_noise_source files \
--iebcs_noise_pos_path /path/to/noise_pos.npy \
--iebcs_noise_neg_path /path/to/noise_neg.npy

# Refractory state coupling with interpolation
--iebcs_refractory_state_coupling true \
--iebcs_refractory_us 700
```

Note: the CLI exposes preset histogram names, but this repository snapshot does
not ship the matching `input/iebcs_noise/*.npy` assets. Use explicit file mode
unless you add those preset files locally.

## Optimization Notes

Historical optimization measurements are archived in
`docs/OPTIMIZATION_SUMMARY.md`. Do not reuse their fixed speedup values as
current results.

- **`LowPassFilter` scalar ε path** (`emulator_utils.py`): uses in-place
  `lerp_()`, but currently lacks the tensor path's stability clamp
- **CPU-GPU transfers** (`emulator.py`): use direct `.item()` and
  `.cpu().numpy()` patterns
- **`Map_linear_to_log_luminance`**: requires float64 internally for OFF-event numerical stability — see `TODO (TBC)` comment before attempting float32 conversion
- **Event buffer pre-allocation**: historical experiment was reverted; current
  code accumulates chunks and concatenates once
- **Parallel histogram** (`v2e_utils.py`): implemented above a one-million-track
  threshold, but current tests do not activate that branch

Active porting/performance roadmap (shared C++/CUDA backend, Python/Julia
interfaces, and parity gates):
`doc/developments/performance_optimization_opportunities.md`

## Development Notes

### Correctness Invariants

When editing core event generation code, preserve:

- Event format: `[timestamp, x, y, polarity]`
- Monotonic timestamps within packets and across the complete output stream
- Polarity values in `{+1, -1}`
- Threshold/reset logic: emitted events must advance/reduce `base_log_frame` consistently
- Refractory logic: events closer than `refractory_period_s` filtered per pixel
- Reproducibility with fixed seeds (`--dvs_emulator_seed`)

Current correctness work is ordered in
`doc/developments/consolidation_staged_plan.md`.

### Performance Hotspots

- `v2ecore/emulator.py`: Iterative event generation loop and per-iteration event extraction
- `v2ecore/emulator_utils.py`: `Map_linear_to_log_luminance` and
  `LowPassFilter`
- `v2e.py` + `v2ecore/slomo.py`: Frame I/O and batch buffering

### Optimization Strategy

1. Reduce Python/container overhead (avoid repeated `np.append`/`torch.cat` in loops)
2. Reduce CPU/GPU sync points
3. Keep numeric behavior unchanged unless guarded by explicit options
4. Add benchmarks and regression tests in the same change

### Working with Code

- The codebase uses the anchor IDs documented in `docs/core_model_mapping.md`
  to map paper math to implementation
- All event timestamps use seconds (float) internally
- CLI arguments in `v2ecore/v2e_args.py` use microseconds for latency/timing parameters (converted to seconds in emulator)
- SuperSloMo checkpoint must be at `input/SuperSloMo39.ckpt` by default (override with `--slomo_model`)

### DVS Timestamp Resolution

- `--auto_timestamp_resolution` (default): Upsample to limit motion to ~1 pixel per frame
- `--timestamp_resolution X`: Force minimum timestamp resolution of X seconds
- `--disable_slomo`: Skip interpolation, use source video frame rate directly
- Check console warnings about photoreceptor filter undersampling if using high cutoff_hz with low frame rates

### DVS Frame Exposure Modes

- `--dvs_exposure duration T`: Fixed duration T seconds per frame
- `--dvs_exposure count N`: N events per frame
- `--dvs_exposure area_count N M`: Frame ends when any MxM block has N events
- `--dvs_exposure source`: One DVS frame per source video frame

### Synthetic Input

Create custom input by subclassing `v2ecore/base_synthetic_input.py` and
override `next_frame()`. Examples: `scripts/particles.py`,
`scripts/moving_dot.py`. Use with:

```bash
python v2e.py --synthetic_input scripts.particles
```

## Dataset Scripts

- `dataset_scripts/ddd/ddd_extract_data.py`: Extract DVS events and APS frames from DDD17/DDD20 recordings
- Use `-m package.script.py` notation when running dataset scripts

## Common Issues

- GPU memory: Reduce `--batch_size` if SuperSloMo runs out of memory
- Slow conversion: v2e runs 50-200X slower than real time even on GPU. Use `--stop_time` for trial runs.
- Photoreceptor filter warnings: Frame rate too low for specified `--cutoff_hz`. Either increase frame rate (lower `--timestamp_resolution`) or reduce cutoff frequency.
- Missing SuperSloMo model: Download from Google Drive and place at `input/SuperSloMo39.ckpt`

## Key References

- v2e paper: Hu, Liu, Delbruck. "v2e: From Video Frames to Realistic DVS Events" (CVPRW 2021) - <https://arxiv.org/abs/2006.07722>
- DVS noise analysis: Graca, Delbruck. "Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise" - <https://arxiv.org/abs/2109.08640>
- IEBCS reference: Joubert et al. "Event Camera Simulator Improvements via Characterized Parameters" (Frontiers in Neuroscience 2021) - <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.702765/full>
- V2CE reference: Zhang et al. "V2CE: Video to Continuous Events Simulator" (ICRA 2024) - <https://arxiv.org/abs/2309.08891>
- v2e home page: <https://sites.google.com/view/video2events/home>
