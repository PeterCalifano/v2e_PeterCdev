# v2e Error Model Extensions (IEBCS + V2CE-Inspired)

This document maps the new optional error-model extensions added to `v2e` while
preserving default behavior.

## Scope

Added extensions:

- IEBCS-inspired event timestamp latency + jitter.
- IEBCS-inspired threshold reset noise (re-sample thresholds after emitted
  signal events).
- V2CE-inspired non-uniform intra-frame timestamp placement for event bursts.

Primary implementation files:

- `v2ecore/v2e_args.py`
- `v2e.py`
- `v2ecore/emulator.py`

Primary tests:

- `test/test_emulator_regression.py`
- `test/test_io_regressions.py`

## Compatibility Guarantee

All new features are opt-in and disabled by default. With default flags,
`v2e` behavior and I/O format remain unchanged.

- Event schema is unchanged: `[t, x, y, p]`.
- Existing CLI options remain valid.
- Default timestamp generation remains linear subdivision.

## CLI Surface

Defined in `v2ecore/v2e_args.py`:

- `--iebcs_latency_jitter_model` (default `false`)
- `--iebcs_latency_mean_us` (default `100.0`)
- `--iebcs_latency_jitter_us` (default `30.0`)
- `--iebcs_resample_thresholds_on_event` (default `false`)
- `--v2ce_nonuniform_burst_timestamps` (default `false`)
- `--v2ce_burst_timestamps_mode` (default `"random"`, choices: `random|slope`)

Validated and forwarded in `v2e.py`, then consumed by `EventEmulator`.

## Extension Map

### 1. IEBCS Latency + Jitter

Intent:

- Simulate event timestamp delay and temporal uncertainty after event formation.

Flag(s):

- Enable with `--iebcs_latency_jitter_model=true`.
- Parameters:
  `--iebcs_latency_mean_us`, `--iebcs_latency_jitter_us`.

Implementation:

- Constructor fields in `EventEmulator.__init__`.
- Applied in `_apply_latency_jitter_and_sort`.
- Called just before conversion/output in `generate_events`.

Behavior:

- Per-event offset sampled as `N(mean_s, jitter_s)`.
- Offsets are clamped to non-negative values.
- Offsets are added to event timestamps.
- Events are sorted by timestamp afterwards to preserve monotonicity.
- `signnoise_label` is reordered with the same permutation.

### 2. IEBCS Threshold Reset Noise

Intent:

- Emulate comparator/reset variability by re-sampling thresholds for pixels that
  actually emitted signal events.

Flag(s):

- Enable with `--iebcs_resample_thresholds_on_event=true`.

Implementation:

- Logic in `_resample_thresholds_after_signal_events`.
- Called in `generate_events` after final signal event masks are known.

Behavior:

- Uses `final_pos_evts_frame > 0` and `final_neg_evts_frame > 0` masks.
- Re-samples ON/OFF thresholds from `N(pos_thres_nominal, sigma_thres)` and
  `N(neg_thres_nominal, sigma_thres)`.
- Clamps re-sampled thresholds to stability lower bound `0.01`.
- Refreshes shot-noise scaling via `_refresh_threshold_probability_scales`.

Notes:

- No-op if `sigma_thres <= 0`.
- No-op if threshold tensors are not initialized as per-pixel tensors.
- Applied to signal-event pixels (not shot-noise-only pixels).

### 3. V2CE Non-Uniform Burst Timestamps

Intent:

- Replace linear intra-frame timestamp spacing for burst iterations with
  non-uniform placement.

Flag(s):

- Enable with `--v2ce_nonuniform_burst_timestamps=true`.
- Mode via `--v2ce_burst_timestamps_mode=random|slope`.

Implementation:

- Centralized in `_sample_signal_timestamps`.
- Used where `ts`/`ts_step` are generated in `generate_events`.

Behavior:

- Default path (`false`): same linear `torch.linspace` timestamps as legacy
  behavior.
- `random` mode: sorted uniform random fractions over the frame interval.
- `slope` mode: sorted `sqrt(U)` fractions, biasing timestamps later in the
  frame.
- `ts_step` remains nominal `delta_time / min_ts_steps` and is still used for
  refractory gating decisions.

Notes:

- Effect is visible when `min_ts_steps > 1` (multi-event bursts in a frame).
- Output remains monotonic.

## Data-Flow Integration

1. Parse flags in `v2ecore/v2e_args.py`.
2. Validate numeric ranges and forward options in `v2e.py`.
3. Store options in `EventEmulator`.
4. During `generate_events`:
   - (optional) non-uniform per-iteration `ts` generation.
   - regular event map + emission pipeline.
   - (optional) threshold re-sampling after final signal masks.
   - (optional) latency+jitter and stable sort before output.

## Implementation Details

### Constructor and State (`v2ecore/emulator.py`)

New constructor parameters are stored as internal state in `EventEmulator.__init__`:

- `iebcs_latency_jitter_model` -> `self.iebcs_latency_jitter_model`
- `iebcs_latency_mean_us` -> `self.iebcs_latency_mean_s = mean_us * 1e-6`
- `iebcs_latency_jitter_us` -> `self.iebcs_latency_jitter_s = jitter_us * 1e-6`
- `iebcs_resample_thresholds_on_event` -> `self.iebcs_resample_thresholds_on_event`
- `v2ce_nonuniform_burst_timestamps` -> `self.v2ce_nonuniform_burst_timestamps`
- `v2ce_burst_timestamps_mode` -> `self.v2ce_burst_timestamps_mode`

`v2ce_burst_timestamps_mode` is validated in the emulator (`random|slope`), and
`v2e.py` validates non-negative latency parameters before construction.

### Timestamp Generation Path (`_sample_signal_timestamps`)

Signal-event iteration timestamps are generated once per frame transition.

- Inputs: `min_ts_steps`, `delta_time`, `t_frame`.
- Returns: `(ts, ts_step)`.
- `ts_step = delta_time / min_ts_steps` is preserved as nominal spacing for
  refractory checks, even when non-uniform timestamp placement is enabled.

Cases:

1. `min_ts_steps == 1`:
   - `ts = [t_frame]`.
2. Default mode (`v2ce_nonuniform_burst_timestamps=false`):
   - `ts = linspace(t_prev + ts_step, t_frame, min_ts_steps)`.
3. Non-uniform mode:
   - Sample `raw ~ U(0,1)` of size `min_ts_steps`.
   - If mode is `slope`, map with `raw = sqrt(raw)` (end-biased).
   - Sort fractions and map to interval:
     `ts = t_prev + delta_time * sort(raw)`.
   - Clamp to `(t_prev, t_frame]` for numerical safety.

### Threshold Reset Noise (`_resample_thresholds_after_signal_events`)

Executed only after final signal event masks are known.

Guard conditions:

- Feature disabled -> immediate return.
- `sigma_thres <= 0` -> immediate return.
- Threshold fields not per-pixel tensors -> immediate return.

For ON/OFF masks independently:

- Mask definition:
  - ON: `final_pos_evts_frame > 0`
  - OFF: `final_neg_evts_frame > 0`
- Resample only masked pixels:
  - ON: `N(pos_thres_nominal, sigma_thres)`
  - OFF: `N(neg_thres_nominal, sigma_thres)`
- Clamp sampled thresholds to `>= 0.01`.
- Recompute shot-noise pre-probability scales only if at least one threshold
  set changed, via `_refresh_threshold_probability_scales()`.

This keeps work proportional to active pixels and avoids full-frame resampling.

### Latency/Jitter Application (`_apply_latency_jitter_and_sort`)

Applied at the end of frame processing, after all signal/noise events are
assembled.

Algorithm:

- If disabled or no events: no-op.
- Sample per-event offset:
  `offset_i ~ N(self.iebcs_latency_mean_s, self.iebcs_latency_jitter_s)`.
- Clamp each offset to `>= 0`.
- Add offset to event timestamps in-place.
- Stable output ordering is restored by sorting `events[:, 0]`.
- If present, `signnoise_label` is permuted with identical indices.

This keeps external packet format unchanged while preserving monotonic output.

### Integration Point in `generate_events`

Call order inside `EventEmulator.generate_events`:

1. Compute event counts (`compute_event_map`).
2. Build per-iteration signal events using `ts` from `_sample_signal_timestamps`.
3. Optionally append shot-noise events (legacy path).
4. Update base memory (`self.base_log_frame`).
5. Optionally resample thresholds for signal-emitting pixels.
6. Optionally apply latency+jitter and sort events.
7. Convert to numpy and write to selected outputs.

This ordering intentionally keeps base v2e internals and outputs unchanged when
all new flags are disabled.

### Complexity Notes

- Signal event assembly avoids repeated `torch.cat` inside the iteration loop:
  per-iteration chunks are collected then concatenated once.
- Threshold reset noise samples only masked pixel subsets (`num_pos`, `num_neg`)
  instead of full-frame random draws.
- Latency model adds one normal sample and one sort per event packet.

### Validation and Error Handling

In `v2e.py`:

- `shot_noise_rate_hz < 0` -> error + exit.
- `iebcs_latency_mean_us < 0` -> error + exit.
- `iebcs_latency_jitter_us < 0` -> error + exit.

In `EventEmulator.__init__`:

- Invalid `v2ce_burst_timestamps_mode` raises `ValueError`.

These checks ensure bad parameterizations fail early.

## Tests Mapping

Feature tests in `test/test_emulator_regression.py`:

- `test_iebcs_latency_jitter_model_delays_timestamps_and_keeps_monotonic`
- `test_iebcs_resample_thresholds_on_event_updates_thresholds_only_when_enabled`
- `test_v2ce_random_burst_timestamps_are_nonuniform_and_monotonic`
- `test_v2ce_slope_mode_biases_events_later_than_random_mode`

CLI wiring test in `test/test_io_regressions.py`:

- `test_main_passes_iebcs_and_v2ce_flags_to_emulator`

## Quick Usage Examples

IEBCS latency+jitter only:

```bash
python v2e.py ... \
  --iebcs_latency_jitter_model true \
  --iebcs_latency_mean_us 120 \
  --iebcs_latency_jitter_us 25
```

IEBCS threshold reset noise only:

```bash
python v2e.py ... \
  --iebcs_resample_thresholds_on_event true
```

V2CE-style burst timestamps:

```bash
python v2e.py ... \
  --v2ce_nonuniform_burst_timestamps true \
  --v2ce_burst_timestamps_mode slope
```
