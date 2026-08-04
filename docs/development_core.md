# Core Development Guide

This document is aimed at contributors working on v2e internals,
especially event-generation performance and correctness.

For a paper-aligned math-to-code map of the event model and the anchor IDs used
throughout the docs,
see [`docs/core_model_mapping.md`](core_model_mapping.md).

## Repository map (core-focused)

- `v2e.py`: top-level CLI and pipeline orchestration.
- `v2ecore/emulator.py`: core DVS event simulation state machine.
- `v2ecore/emulator_utils.py`: low-level numerical helpers.
- `v2ecore/slomo.py`: SuperSloMo interpolation path.
- `v2ecore/renderer.py`: event-to-video frame accumulation.
- `v2ecore/output/*`: event serialization backends.
- `test/`: regression tests.
- `v2ecore/benchmarks/`: repo-native profiling and comparative benchmark
  helpers.
- `doc/developments/`: staged plans and open development checklists.

## End-to-end pipeline

1. Parse CLI args in `v2e.py` and configure models/output.
2. Read and preprocess frames (crop, resize, grayscale conversion).
3. Optionally interpolate frames (SloMo).
4. Run `EventEmulator.generate_events(frame, timestamp)` for each frame.
5. Optionally render accumulated event frames and serialize event streams.

## Correctness invariants

When editing core code, preserve:

- Event format: `[timestamp, x, y, polarity]`.
- Monotonic timestamps within each packet and across the complete output stream.
- Polarity values in `{+1, -1}`.
- Threshold/reset logic:
  emitted ON/OFF events must advance/reduce `base_log_frame` consistently.
- Refractory logic:
  events closer than `refractory_period_s` should be filtered per pixel.
- Reproducibility when a non-zero seed is provided.

Known violations and their staged tests are tracked in
[`consolidation_staged_plan.md`](../doc/developments/consolidation_staged_plan.md).

## Known hotspots

- `v2ecore/emulator.py`:
  iterative event generation loop and per-iteration event extraction.
- `v2ecore/emulator_utils.py`:
  lin-log transform and low-pass processing.
- `v2e.py` + `v2ecore/slomo.py`:
  intermediate frame I/O and batch buffering.

## Safe optimization strategy

Preferred order:

1. Establish a failing correctness/parity test before changing a model path.
2. Reduce Python/container overhead and unnecessary CPU/GPU sync points.
3. Keep numeric behavior unchanged unless a separately reviewed model correction
   requires it.
4. Add benchmarks and regression tests in the same change.

For accelerator work, use
[`doc/developments/performance_optimization_opportunities.md`](../doc/developments/performance_optimization_opportunities.md).
The current direction is a shared C++/CUDA backend callable from Python and
Julia, with Python/PyTorch as the reference fallback.

## Benchmarking workflow

- Core emulator timing:
  `conda run -n v2e python v2ecore/benchmarks/benchmark_emulator.py`
- Core emulator profiling:
  `conda run -n v2e python v2ecore/benchmarks/benchmark_emulator.py --profile`
- Comparative IEBCS / V2CE benchmark:
  `conda run -n v2e python v2ecore/benchmarks/benchmark_error_models_eventstream.py`
- Dummy-motion 3D event visualization:
  `conda run -n v2e python scripts/plot_events_3d_example.py --scenario moving_blob --save output/events_3d_blob.png --no_show`

Record benchmark parameters when sharing results:

- resolution
- number of frames
- DVS parameters (`pos_thres`, `neg_thres`, `sigma_thres`, `cutoff_hz`)
- noise settings (`leak_rate_hz`, `shot_noise_rate_hz`, `photoreceptor_noise`)
- platform (`CPU/GPU`, torch version)

## Testing workflow

- Run all tests:
  `conda run -n v2e python -m pytest -q`
- Run emulator regressions only:
  `conda run -n v2e python -m pytest -q test/test_emulator_*.py`

Recommended regression checks for core changes:

- event shape and coordinate bounds
- packet and global-stream timestamp monotonicity
- deterministic output with fixed seeds
- signal/noise labeling path (if enabled)
- writer round-trip and finalization behavior
- interaction checks for every pair of enabled timing/noise mechanisms

## Notes for `frames2events_quick.sh`

The helper script accepts `-a|--auto_timestamp` for convenience, but it must
expand that option to the real `v2e.py` CLI flag
`--auto_timestamp_resolution` defined in `v2ecore/v2e_args.py`.
