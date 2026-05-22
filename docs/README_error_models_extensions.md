# v2e Error Model Extensions (IEBCS + V2CE-Inspired)

This document maps the new optional error-model extensions added to `v2e` while
preserving default behavior.

## Scope

Added extensions:

- IEBCS-inspired event timestamp latency + jitter.
- IEBCS-inspired threshold reset noise (re-sample thresholds after emitted
  signal events).
- V2CE-inspired non-uniform intra-frame timestamp placement for event bursts.
- IEBCS Stage-2 inspired contrast-latency, histogram background noise, and
  refractory-state coupling (all optional).

Primary references:

- Joubert et al. (2021), *Event Camera Simulator Improvements via Characterized
  Parameters*, Frontiers in Neuroscience.
- Zhang et al. (2024), *V2CE: Video to Continuous Events Simulator*, ICRA 2024
  / arXiv:2309.08891.

Primary implementation files:

- `v2ecore/v2e_args.py`
- `v2e.py`
- `v2ecore/emulator.py`

Primary tests:

- `test/test_emulator_regression.py`
- `test/test_io_regressions.py`

Comparative benchmark and visualization:

- `README_comparative_benchmark.md`
- Runner:
  `python v2ecore/benchmarks/benchmark_error_models_eventstream.py`

## Compatibility Guarantee

All new features are opt-in and disabled by default. With default flags,
`v2e` behavior and I/O format remain unchanged.

- Event schema is unchanged: `[t, x, y, p]`.
- Existing CLI options remain valid.
- Default timestamp generation remains linear subdivision.

## Behavioral Equivalence Scope

The right standard for these options is behavioral equivalence of the modeled
effect, not source-code identity with the upstream simulator.

- IEBCS-derived options in this repo aim to reproduce the same first-order
  output effect as the corresponding IEBCS mechanisms:
  delayed / jittered timestamps, reset-threshold variability, histogram-driven
  background noise, and refractory release interpolation.
- V2CE-derived options in this repo do **not** aim to reproduce the full V2CE
  simulator output. They target one narrower effect from V2CE: reducing
  timestamp layering inside same-frame event bursts.

As a result:

- IEBCS-derived features are best described as behaviorally aligned mechanisms
  implemented inside the `v2e` architecture.
- The V2CE-derived feature is best described as a behaviorally aligned
  timestamp-placement heuristic, not a full V2CE-equivalent event generator.

## CLI Surface

Defined in `v2ecore/v2e_args.py`:

- `--iebcs_latency_jitter_model` (default `false`)
- `--iebcs_latency_mean_us` (default `100.0`)
- `--iebcs_latency_jitter_us` (default `30.0`)
- `--iebcs_resample_thresholds_on_event` (default `false`)
- `--iebcs_contrast_latency_model` (default `false`)
- `--iebcs_latency_tau_us` (default `300.0`)
- `--iebcs_latency_clamp_us` (default `10000.0`)
- `--iebcs_latency_slope_jitter` (default `true`)
- `--iebcs_hist_noise_model` (default `false`)
- `--iebcs_noise_source` (default `"preset"`, choices: `preset|files`)
- `--iebcs_noise_preset` (default `"161lux"`, choices: `3klux|161lux|0.1lux`)
- `--iebcs_noise_pos_path` (default `None`)
- `--iebcs_noise_neg_path` (default `None`)
- `--iebcs_refractory_state_coupling` (default `false`)
- `--iebcs_refractory_us` (default `None`, then falls back to `--refractory_period`)
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

- Per-event offset sampled from `Normal(mean_s, jitter_s)`.
- Offsets are clamped to non-negative values.
- Offsets are added to event timestamps.
- Events are sorted by timestamp afterwards to preserve monotonicity.
- `signnoise_label` is reordered with the same permutation.

Current code-level summary:

```text
delta_i ~ Normal(mean_s, jitter_s)
t_i,new = t_i + max(delta_i, 0)
```

Behavioral-equivalence assessment:

- Strong for the intended effect.
- The output effect matches the IEBCS mechanism class: event times are delayed
  and jittered after crossing formation.
- Exact parameterization is not identical to upstream IEBCS in every detail, so
  this should not be described as reference-identical output.

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
- Re-samples ON/OFF thresholds from
  `Normal(pos_thres_nominal, sigma_thres)` and
  `Normal(neg_thres_nominal, sigma_thres)`.
- Clamps re-sampled thresholds to stability lower bound `0.01`.
- Refreshes shot-noise scaling via `_refresh_threshold_probability_scales`.

Current code-level summary:

```text
theta_on,new  ~ Normal(theta_on_nominal,  sigma_theta)
theta_off,new ~ Normal(theta_off_nominal, sigma_theta)
theta_* = max(theta_*, 0.01)
```

Notes:

- No-op if `sigma_thres <= 0`.
- No-op if threshold tensors are not initialized as per-pixel tensors.
- Applied to signal-event pixels (not shot-noise-only pixels).

Behavioral-equivalence assessment:

- Strong for the intended effect.
- This reproduces the same comparator-reset noise behavior class as IEBCS:
  thresholds are re-sampled for pixels that emitted signal events.

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

Current code-level summary:

```text
Delta t = t_frame - t_prev

linear:  t_i = t_prev + i * Delta t / N,      i = 1, ..., N
random:  t_i = t_prev + Delta t * sort(U_i)
slope:   t_i = t_prev + Delta t * sort(sqrt(U_i))

U_i ~ Uniform(0, 1)
```

Notes:

- Effect is visible when `min_ts_steps > 1` (multi-event bursts in a frame).
- Output remains monotonic.
- This is a V2CE-inspired timestamp-placement heuristic, not the full learned
  V2CE simulator from Zhang et al. (2024).

Behavioral-equivalence assessment:

- Partial only.
- It reproduces the same narrow output effect that motivated the V2CE timestamp
  stage, namely reducing artificial temporal layering from uniform burst
  placement.
- It does **not** reproduce the full V2CE behavior, because it does not infer
  timestamps from learned event voxels or local dynamic-aware slope estimates in
  the same way as the official V2CE pipeline.

### 4. IEBCS Stage-2: Contrast Latency

Intent:

- Use signal-dependent latency per emitted signal event, instead of only a
  fixed packet-level timestamp shift.

Flags:

- `--iebcs_contrast_latency_model=true`
- Optional tuning:
  `--iebcs_latency_tau_us`,
  `--iebcs_latency_clamp_us`,
  `--iebcs_latency_slope_jitter`

Behavior:

- For each signal event, latency is sampled from an amplitude/slope-dependent
  model and added to its per-iteration base timestamp.
- Existing `--iebcs_latency_jitter_model` remains available as an additional
  post-packet perturbation model for backward compatibility.

Current code-level summary:

```text
drive = |L_photo + eta_photo - L_mem|
amp = clamp(theta / drive, eps, 1 - eps)
lat = mu_lat - tau_lat * log(1 - amp) + xi
lat = clamp(lat, 0, lat_clamp)
t_i,new = t_i + lat
```

with `xi` sampled from a zero-mean Gaussian whose standard deviation is either
constant or slope-scaled, depending on `--iebcs_latency_slope_jitter`.

Behavioral-equivalence assessment:

- Strong for the intended effect.
- The mean latency term follows the same first-order structure as IEBCS
  front-end interpolation / contrast-latency modeling.
- Jitter scaling is not upstream-identical, so the correct claim is behavioral
  alignment rather than exact output equivalence.

### 5. IEBCS Stage-2: Histogram Background Noise

Intent:

- Replace simplified shot-noise probability sampling with scheduled ON/OFF
  background noise events sampled from measured histogram distributions.

Flags:

- Enable: `--iebcs_hist_noise_model=true`
- Source selection:
  - Presets: `--iebcs_noise_source preset --iebcs_noise_preset 161lux`
  - Files: `--iebcs_noise_source files --iebcs_noise_pos_path ... --iebcs_noise_neg_path ...`

Behavior:

- One ON and one OFF noise CDF are assigned per pixel.
- Each pixel keeps next ON/OFF noise timestamps; events due in `(t_prev, t_frame]`
  are emitted and rescheduled.
- Event ordering and optional signal/noise labels are preserved.

Notes:

- This repository snapshot does not include preset histogram `.npy` files under
  `input/iebcs_noise/`. Preset mode therefore requires users to add those files
  themselves; explicit `files` mode works immediately.

Behavioral-equivalence assessment:

- Strong if the same histogram assets are used.
- The implementation uses the same output-level mechanism class as IEBCS:
  assign ON/OFF histogram CDF rows per pixel, schedule next noise timestamps,
  emit due events, then reschedule.
- A local safety cap is added to avoid pathological per-frame explosions, so in
  extreme cases output can diverge intentionally from upstream behavior.

### 6. IEBCS Stage-2: Refractory State Coupling

Intent:

- Couple refractory handling with state release timestamps and interpolate
  memory state on refractory release.

Flags:

- `--iebcs_refractory_state_coupling=true`
- `--iebcs_refractory_us` (optional override of `--refractory_period`)

Behavior:

- Per-pixel refractory-release timestamps gate candidate events.
- Pixels leaving refractory inside a frame interval update memory state using
  release-time interpolation before further event checks.

Behavioral-equivalence assessment:

- Strong for the intended effect.
- This captures the same modeled behavior as IEBCS refractory-state coupling:
  release-time interpolation followed by renewed threshold checking inside the
  same frame interval.

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
   - Sample `raw ~ Uniform(0, 1)` of size `min_ts_steps`.
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
  - ON: `Normal(pos_thres_nominal, sigma_thres)`
  - OFF: `Normal(neg_thres_nominal, sigma_thres)`
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
  `offset_i ~ Normal(self.iebcs_latency_mean_s, self.iebcs_latency_jitter_s)`.
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
- `iebcs_latency_tau_us < 0` -> error + exit.
- `iebcs_latency_clamp_us < 0` -> error + exit.
- `iebcs_refractory_us < 0` -> error + exit.
- `--iebcs_hist_noise_model` with `--iebcs_noise_source=files` requires both
  ON/OFF file paths.
- `--iebcs_hist_noise_model` with missing resolved ON/OFF files (including
  missing preset files) -> actionable error + exit before emulator construction.

In `EventEmulator.__init__`:

- Invalid `v2ce_burst_timestamps_mode` raises `ValueError`.
- Histogram-noise ON/OFF file existence is validated before loading.

In histogram-noise event scheduling:

- Per-frame histogram-noise output is capped at
  `8 * num_pixels` events (internal guardrail) to avoid runaway loops.
- Excess due events are rescheduled strictly after frame time and a warning is logged.

These checks ensure bad parameterizations fail early.

## Tests Mapping

Feature tests in `test/test_emulator_regression.py`:

- `test_iebcs_latency_jitter_model_delays_timestamps_and_keeps_monotonic`
- `test_iebcs_resample_thresholds_on_event_updates_thresholds_only_when_enabled`
- `test_v2ce_disabled_matches_default_linear_path`
- `test_v2ce_random_burst_timestamps_are_nonuniform_and_monotonic`
- `test_v2ce_slope_mode_biases_events_later_than_random_mode`
- `test_invalid_v2ce_burst_timestamps_mode_raises_value_error`
- `test_refractory_release_interpolation_is_idempotent_within_frame`
- `test_hist_noise_event_generation_is_capped_and_reschedules_dropped_due_events`

CLI wiring test in `test/test_io_regressions.py`:

- `test_main_passes_iebcs_and_v2ce_flags_to_emulator`
- `test_cli_rejects_invalid_hist_noise_configuration`
- `test_cli_rejects_missing_preset_hist_noise_files`
- `test_cli_rejects_negative_latency_tau_or_refractory_values`

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

IEBCS Stage-2 contrast latency + refractory coupling:

```bash
python v2e.py ... \
  --iebcs_contrast_latency_model true \
  --iebcs_latency_tau_us 300 \
  --iebcs_refractory_state_coupling true \
  --iebcs_refractory_us 700
```

IEBCS Stage-2 histogram noise from explicit files:

```bash
python v2e.py ... \
  --iebcs_hist_noise_model true \
  --iebcs_noise_source files \
  --iebcs_noise_pos_path /path/to/noise_pos.npy \
  --iebcs_noise_neg_path /path/to/noise_neg.npy
```
