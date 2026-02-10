v2e Core Developer Notes
========================

This folder contains the core simulation pipeline for `v2e`.

Core modules
------------

- `emulator.py`
  Stateful DVS pixel-array emulator. This is the main event-generation
  implementation and the primary performance hotspot for high event rates.
- `emulator_utils.py`
  Low-level numeric kernels used by the emulator:
  lin-log mapping, low-pass filtering, threshold/event maps, leak and shot noise.
- `renderer.py`
  Event-to-frame accumulation and optional preview/video export.
- `slomo.py`, `model.py`, `dataloader.py`
  SuperSloMo interpolation path and model code.
- `v2e_args.py`
  Central CLI argument definitions and exposure mode parsing.
- `output/`
  Event writers (`.txt`, `.aedat2`, `.aedat4`).

Execution flow (high level)
---------------------------

1. `v2e.py` reads and resizes source frames.
2. Optional SloMo interpolation increases temporal resolution.
3. `EventEmulator.generate_events` consumes each frame/timestamp pair and emits
   `[t, x, y, p]` events.
4. Optional rendering/output writers serialize data.

Performance-sensitive paths
---------------------------

- `EventEmulator.generate_events` in `emulator.py`.
- Lin-log and low-pass operations in `emulator_utils.py`.
- Frame conversion and intermediate frame I/O in `v2e.py`/`slomo.py`.

When making performance changes, keep these constraints:

- Preserve event semantics:
  timestamps monotonic per frame packet, correct polarity, and per-pixel
  threshold/reset behavior.
- Preserve reproducibility for fixed seeds where possible.
- Avoid changing defaults unless behavior is explicitly intended to change.

Developer workflow
------------------

- Run unit tests:
  `pytest -q`
- Run focused core benchmarks:
  `python scripts/benchmark_emulator.py`
- Run cProfile for emulator hot path:
  `python scripts/benchmark_emulator.py --profile`
- Run a short end-to-end CLI benchmark:
  `python scripts/benchmark_v2e_cli.py`

For deeper architecture and optimization guidance, see:
`docs/development_core.md`.

For paper-to-code mapping of the event model with `#CORE[...]` tags, see:
`docs/core_model_mapping.md`.
