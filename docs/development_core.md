# Core Development Guide

This document is aimed at contributors working on v2e internals,
especially event-generation performance and correctness.

For a paper-aligned math-to-code map of the event model and `#KEY[...]` tags,
see [`docs/core_model_mapping.md`](core_model_mapping.md).

## Repository map (core-focused)

- `v2e.py`: top-level CLI and pipeline orchestration.
- `v2ecore/emulator.py`: core DVS event simulation state machine.
- `v2ecore/emulator_utils.py`: low-level numerical helpers.
- `v2ecore/slomo.py`: SuperSloMo interpolation path.
- `v2ecore/renderer.py`: event-to-video frame accumulation.
- `v2ecore/output/*`: event serialization backends.
- `test/`: regression tests.
- `scripts/benchmark_*.py`: profiling/benchmark helpers.

## End-to-end pipeline

1. Parse CLI args in `v2e.py` and configure models/output.
2. Read and preprocess frames (crop, resize, grayscale conversion).
3. Optionally interpolate frames (SloMo).
4. Run `EventEmulator.generate_events(frame, timestamp)` for each frame.
5. Optionally render accumulated event frames and serialize event streams.

## Correctness invariants

When editing core code, preserve:

- Event format: `[timestamp, x, y, polarity]`.
- Monotonic timestamps within each emitted event packet.
- Polarity values in `{+1, -1}`.
- Threshold/reset logic:
  emitted ON/OFF events must advance/reduce `base_log_frame` consistently.
- Refractory logic:
  events closer than `refractory_period_s` should be filtered per pixel.
- Reproducibility when a non-zero seed is provided.

## Known hotspots

- `v2ecore/emulator.py`:
  iterative event generation loop and per-iteration event extraction.
- `v2ecore/emulator_utils.py`:
  lin-log transform and low-pass processing.
- `v2e.py` + `v2ecore/slomo.py`:
  intermediate frame I/O and batch buffering.

## Safe optimization strategy (Python-only)

Preferred order:

1. Reduce Python/container overhead
   (avoid repeated `np.append`/`torch.cat` in loops).
2. Reduce unnecessary CPU/GPU sync points.
3. Keep numeric behavior unchanged unless guarded behind explicit options.
4. Add benchmarks and regression tests in the same change.

## Benchmarking workflow

- Core emulator timing:
  `python scripts/benchmark_emulator.py`
- Core emulator profiling:
  `python scripts/benchmark_emulator.py --profile`
- Short CLI pipeline timing:
  `python scripts/benchmark_v2e_cli.py`

Record benchmark parameters when sharing results:

- resolution
- number of frames
- DVS parameters (`pos_thres`, `neg_thres`, `sigma_thres`, `cutoff_hz`)
- noise settings (`leak_rate_hz`, `shot_noise_rate_hz`, `photoreceptor_noise`)
- platform (`CPU/GPU`, torch version)

## Testing workflow

- Run all tests:
  `pytest -q`
- Run emulator regressions only:
  `pytest -q test/test_emulator_*.py`

Recommended regression checks for core changes:

- event shape and coordinate bounds
- timestamp monotonicity
- deterministic output with fixed seeds
- signal/noise labeling path (if enabled)

## Notes for `frames2events_quick.sh`

The script must pass CLI argument names that exist in `v2ecore/v2e_args.py`.
Use `--auto_timestamp_resolution`, not `--auto_timestamp`.
