# Findings: IEBCS-Inspired Extensions

Scope: the opt-in IEBCS mechanisms — latency and jitter, threshold reset,
histogram background noise, and refractory-state coupling. All are disabled by
default; every defect here requires an explicit flag.

Reference source compared against: `../IEBCS` (`src/dvs_sensor.py`), a clean
local checkout. Semantics of each mechanism are described in
[`../README_error_models_extensions.md`](../README_error_models_extensions.md).

Conventions and severity definitions: [`README.md`](README.md).

Cross-packet ordering under latency is filed as
[STREAM-001](stream_and_io.md#stream-001) because it is a stream-lifecycle
defect rather than an IEBCS modelling one.

---

## IEBCS-001

**Histogram noise schedule is anchored to absolute zero**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `_init_iebcs_noise_schedule()`

### Issue

The initial next-event times are built as `delay * phase` with
`phase ~ U(0,1)`, measured from absolute zero:

```python
self.iebcs_noise_next_pos_s = (pos_delay * pos_phase).reshape(shape)
```

They are never offset by the timestamp of the first frame. `_sample_hist_noise_events`
then emits every pixel whose scheduled time is `<= t_frame`.

### Why it is an issue

For any stream that does not start at `t = 0`, the entire noise schedule is
already overdue on the first frame. The emulator floods the first packet with
events stamped in the distant past, then keeps emitting until the per-frame cap
stops it. The events are not merely mistimed — they carry timestamps outside the
frame interval that produced them, which breaks the basic invariant that a
packet's events lie in `(t_previous, t_frame]`, and they arrive before the
recording's own start time.

Non-zero start times are normal: any segment extracted from a longer recording,
any resumed sequence, and any caller that passes wall-clock-derived timestamps.

The reference does not have this problem because it defines its own clock:
`init_image()` sets `self.time = 0`, so IEBCS's phase-from-zero is phase from
*its* stream start. v2e keeps absolute input timestamps but copied the
zero-relative initialisation, so the two only agree when the stream happens to
begin at zero.

### Evidence

Stream starting at `t = 100.0 s`, frame interval `(100.0, 100.02]`:

```text
emitted events: 288
min ts = 0.000001   max ts = 100.002480
events BEFORE the first frame time (100.0 s): 285 / 288
```

285 of 288 events precede the start of the recording.

### Suggested fix

Anchor the schedule to the first frame timestamp. `_init_iebcs_noise_schedule`
is already called from `_init()`, which receives the first frame, so the origin
is available:

```python
self.iebcs_noise_next_pos_s = t_first + (pos_delay * pos_phase).reshape(shape)
self.iebcs_noise_next_neg_s = t_first + (neg_delay * neg_phase).reshape(shape)
```

Pass the first timestamp into `_init()` alongside the first frame. Add a
regression that starts a stream at a large non-zero time and asserts every
emitted noise timestamp lies within the frame interval that produced it. Note
this must be settled together with [LIFE-002](state_lifecycle.md#life-002),
since both concern what "the start of the stream" means after a reset.

---

## IEBCS-002

**CDF loader uses a running maximum, not a cumulative sum**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `_normalize_noise_cdf()`

### Issue

The loader enforces monotonicity with `np.maximum.accumulate(arr, axis=1)`,
then divides by the last column. That is correct *only* if the input is already
cumulative. If the input is a per-bin histogram — a probability density — the
running maximum does not integrate it; it flattens it.

### Why it is an issue

The user-facing vocabulary points the wrong way. The flags are
`--iebcs_hist_noise_model`, `--iebcs_noise_pos_path`, the attributes are
`iebcs_hist_noise_*`, and the documentation calls this "histogram-based
background noise from measured distributions". A user supplying an actual
histogram gets a silently wrong distribution: no exception, no warning, just a
noise process sampling from the wrong frequencies for the whole run.

The failure mode is severe because it concentrates probability incorrectly. In
the probe below, a histogram whose mass sits in the middle bins is transformed
into a distribution that assigns *all* probability to the second bin, because
the running maximum saturates at the first peak and normalisation then maps
that plateau to 1.0.

The reference expects cumulative input too — IEBCS annotates its arrays as
"Positive noise cumulative distributions" — so v2e's expectation is right. The
defect is that the code accepts non-cumulative input silently instead of
rejecting it.

### Evidence

```text
input row (a histogram / PDF)          : [0. 5. 3. 2. 0.]
_normalize_noise_cdf result            : [0. 1. 1. 1. 1.]
a true CDF of that PDF would be        : [0.  0.5 0.8 1.  1. ]
```

### Suggested fix

Validate rather than coerce. A genuine CDF is already non-decreasing, so
`maximum.accumulate` should be a no-op on valid input — which makes it a usable
detector:

```python
if not np.all(np.diff(arr, axis=1) >= -tol):
    raise ValueError(
        f"{label} noise histogram rows must be cumulative distributions "
        "(non-decreasing). Pass np.cumsum(hist, axis=-1) if you have "
        "per-bin counts.")
```

Keep the clamp for floating-point noise (`tol`), but reject genuine
non-monotonicity. Document the expected input format in
`README_error_models_extensions.md` explicitly as "cumulative, one row per
pixel spectrum", and say so in the `--iebcs_noise_pos_path` help text. Consider
renaming the flags to `..._cdf_path` if a compatibility break is acceptable.

---

## IEBCS-003

**CDF normalisation divisor differs from the IEBCS reference**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `_normalize_noise_cdf()`; reference
  `../IEBCS/src/dvs_sensor.py`, `init_bgn_hist()`

### Issue

v2e normalises each row by its **last** element and then forces that element to
exactly 1.0:

```python
arr = arr / row_last[:, None]
arr[:, -1] = 1.0
```

IEBCS normalises by the **second-to-last** element:

```python
self.bgn_hist_pos[id_p, :] = self.bgn_hist_pos[id_p, :] / \
                             np.repeat(self.bgn_hist_pos[id_p, -2]...)
```

### Why it is an issue

The two produce different scalings of the same file, so the same measured
distribution yields different sampled frequencies — and therefore different
background noise rates — in v2e than in the reference. Since inverse-CDF
sampling picks `argmax(cdf >= u)`, a uniformly rescaled row shifts which bin a
given `u` selects, biasing the whole noise spectrum.

This matters specifically because it is invisible: both normalisations produce
a valid-looking monotonic row ending near 1, so no validation catches it, and
any future parity comparison against IEBCS would show a systematic rate offset
with no obvious cause.

The divisor choice is not arbitrary in the reference — dividing by `[-2]`
implies the final bin is an overflow or terminator rather than a probability
mass point. If that is the file format, v2e's use of `[-1]` also
double-counts that terminator into the distribution.

### Suggested fix

Establish the file format first, since the correct answer depends on it:

1. Inspect the reference `.npy` layout — 72 bins, and whether the last column is
   an overflow bin. This is blocked on
   [IEBCS-007](#iebcs-007) (the assets are not present).
2. Match the reference divisor once the layout is known, and record the reason
   in a comment — `[-2]` is surprising enough that it will be "corrected" back
   to `[-1]` otherwise.
3. Add a fixture derived from the reference file and assert v2e and IEBCS
   produce the same sampled-frequency distribution for a fixed seed.

Until then, treat measured noise **rates** from this path as uncalibrated
against IEBCS, and say so where the mechanism is described.

---

## IEBCS-004

**Zero-duration refractory coupling is not inert**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `generate_events()` loop and
  `_apply_refractory_release_interpolation()`

### Issue

Enabling `iebcs_refractory_state_coupling` with `iebcs_refractory_us = 0`
changes the event count substantially, although a zero-length refractory period
should impose no constraint at all.

### Why it is an issue

A parameter set to its neutral value must reproduce the unmodified model. That
property is what makes a mechanism testable: it isolates "the feature is
enabled" from "the feature is doing something", and it is the standard way to
show an opt-in path has no side effects at its identity setting. Here the two
are entangled, so any measurement of the refractory model's effect is confounded
by whatever else enabling the flag changes.

The cause is structural rather than a wrong constant. Enabling the flag does
three separate things: it disables the legacy refractory branch entirely (the
guard is `if (not self.iebcs_refractory_state_coupling) and ...`), it gates
event eligibility on `release_ts <= ts[i]`, and it invokes the memory
interpolation described in [IEBCS-005](#iebcs-005). Only the second is
neutralised by a zero period; the other two still apply.

### Evidence

Identical five-frame ramp, fixed seed, `sigma_thres = 0`:

```text
events, coupling disabled            : 324
events, coupling on, refractory=0us  : 216
zero-duration coupling inert?        : False
```

A third of the events disappear at the setting that should change nothing.

### Suggested fix

Separate the three concerns so the period alone controls the gating:

- Keep exactly one refractory implementation active at a time and make the
  choice explicit, rather than having the coupling flag silently disable the
  legacy branch as a side effect.
- Make the release-state update a no-op when `iebcs_refractory_s == 0`, rather
  than relying on the comparison to be vacuous.
- Add the inertness test as a hard gate: for a fixed seed, coupling-on with a
  zero period must reproduce coupling-off **bitwise**.

This is entangled with [IEBCS-005](#iebcs-005) and should be fixed in the same
change, since the interpolation is one of the three behaviours involved.

---

## IEBCS-005

**Refractory interpolation edits memory after quantisation**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `_apply_refractory_release_interpolation()`,
  called from the `generate_events()` iteration loop; anchors `KEY[F-EVENT-MAP-CALL]`
  and `KEY[F-LMEM-UPDATE]`

### Issue

Event counts are quantised once, before the loop:

```python
pos_evts_frame, neg_evts_frame = compute_event_map(
    self.diff_frame, self.pos_thres, self.neg_thres)
```

Inside the loop, `_apply_refractory_release_interpolation()` then modifies
`self.base_log_frame` for pixels leaving refractory. After the loop,
`base_log_frame` is additionally incremented by `n * theta`.

Two things follow: no renewed threshold check is performed after the state
change, and the end-of-frame increment is applied to a memory value that is no
longer the one the quantisation was derived from.

### Why it is an issue

The mechanism is documented as capturing IEBCS's "release-time interpolation
followed by renewed threshold checking". The renewed check does not exist —
counts computed before the interpolation are used unchanged after it. So the
interpolation can move memory across a threshold boundary and no event is
generated for the crossing.

The second effect is a state-consistency bug independent of reference fidelity.
`L_mem += n * theta` is a *reset-by-increment*: it is only correct if `n` was
computed from the same `L_mem`. Once interpolation has moved `L_mem` toward the
photoreceptor value, adding `n * theta` on top applies a partial reset and a
full reset to the same pixel in the same frame, so the comparator memory drifts
away from any value the model defines.

### Evidence

Four-frame ramp with coupling enabled at 700 µs, tracking the two states:

```text
frame 0: mean base_log=4.094345  mean lp_log=4.094345  diff=+0.000000
frame 1: mean base_log=4.625460  mean lp_log=4.499810  diff=-0.125650
frame 2: mean base_log=4.792406  mean lp_log=4.867535  diff=+0.075129
frame 3: mean base_log=5.332777  mean lp_log=5.192957  diff=-0.139820
```

`base_log_frame` overshoots the photoreceptor output and alternates sign — the
signature of two competing reset rules applied to the same state.

The count effect is quantified in [IEBCS-004](#iebcs-004).

### Suggested fix

Restructure to state-first ordering, which is what the reference does and what
makes a renewed check possible at all:

1. Evolve release state for pixels leaving refractory **before** computing
   `diff_frame`.
2. Quantise against the updated memory.
3. Apply the reset-by-increment once, from the same memory the quantisation
   used.

That means the interpolation moves out of the per-iteration loop and above
`compute_event_map`, which is a real restructure of `generate_events` rather
than a local patch — hence "state-first event loop" rather than a one-line fix.
Separately decide whether the release evolution should be linear (current) or
exponential (reference); the choice belongs in
`README_error_models_extensions.md` with its justification, and until it is
made the mechanism cannot be described as behaviourally aligned.

---

## IEBCS-006

**Noise cap is polarity-ordered and spatially biased**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `_sample_hist_noise_events()`

### Issue

The per-frame safety cap (`IEBCS_MAX_NOISE_EVENTS_PER_FRAME_FACTOR * n_pixels`)
is consumed by a loop that drains all due ON candidates first, and only then
processes OFF candidates under whatever budget remains. When a partial batch
must be truncated, it keeps the first `remaining` entries of a `nonzero()`
result, which is row-major.

### Why it is an issue

Two independent biases, both silent:

1. **Polarity bias.** A frame that saturates the cap can emit ON events only,
   because OFF processing is guarded by `if emitted_events < max_events`. The
   ON/OFF balance of background noise is one of the primary things a noise model
   is judged on, so a cap that destroys it converts a resource guard into a
   model artefact.
2. **Spatial bias.** Truncating a `nonzero()` result in row-major order keeps
   events from the top of the sensor and drops events from the bottom. The
   resulting noise is spatially non-uniform in a way that no physical mechanism
   would produce, and that would corrupt any per-pixel event-rate map.

The cap is honestly labelled in the code as a guard rather than a model claim,
and it does log a warning when it engages. The problem is what it does *when* it
engages, not that it exists.

### Suggested fix

Make the cap polarity-fair and spatially unbiased:

- Merge ON and OFF due candidates into one list ordered by timestamp, then apply
  a single cap to the merged list. Earliest-first is both defensible physically
  and free of polarity preference.
- If a partial batch must still be truncated, select uniformly at random from
  the due set using the emulator's private generator, so the choice is unbiased
  and reproducible.
- Expose the capped count as a diagnostic (a counter on the emulator, like the
  existing `h5_writer_stats`) rather than only a log line, so a run can be
  checked for cap engagement after the fact.

Define the cap semantics in the docstring: whether capped events are dropped or
deferred, and what the reschedule does to the pixel's phase.

---

## IEBCS-007

**Preset noise assets are not present in the repository**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/model_options.py`, `IEBCS_NOISE_PRESET_FILES`;
  `--iebcs_noise_source preset` in `v2ecore/v2e_args.py`

### Issue

The CLI offers `--iebcs_noise_source preset` with the choices `3klux`,
`161lux`, `0.1lux`, mapping to six `input/iebcs_noise/*.npy` files. Those files
are not in this repository. `IebcsNoisePreset`'s own docstring records that they
are not bundled.

### Why it is an issue

An advertised CLI option that cannot succeed is a defect in the interface, not
just missing data. A user selecting a documented preset gets a runtime failure
from `_load_iebcs_noise_distributions`, after argument parsing has already
accepted the value as valid. The repository currently carries the caveat in four
separate documents instead of removing the dead choice, which is why it keeps
reappearing in status reviews.

It also blocks [IEBCS-003](#iebcs-003): the normalisation discrepancy against
the reference cannot be settled without an actual reference file to inspect.

### Suggested fix

Choose one, and delete the caveats the other three docs carry:

- **Bundle** the assets if their licence permits, and add a checksum test.
- **Generate** them with a documented script from a cited measurement source.
- **Remove** `preset` from the `--iebcs_noise_source` choices until assets
  exist, leaving `files` as the only accepted value.

Removal is recommended as the immediate step: it is reversible, it makes the CLI
honest, and it costs nothing that currently works. Keep `IebcsNoisePreset` and
`IEBCS_NOISE_PRESET_FILES` in `model_options.py` so the intended names survive
for whoever adds the assets.
