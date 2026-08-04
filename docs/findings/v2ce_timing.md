# Findings: V2CE-Inspired Timestamp Modes

Scope: `--v2ce_nonuniform_burst_timestamps` and
`--v2ce_burst_timestamps_mode {random,slope}`. Both are disabled by default.

The forward design for this area is
[`../developments/v2ce_timing_staged_plan.md`](../developments/v2ce_timing_staged_plan.md);
this file records only what is currently wrong or overclaimed.

Conventions and severity definitions: [`README.md`](README.md).

---

## V2CE-001

**Timestamp layers remain frame-global**

- **Severity:** High
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `_sample_signal_timestamps()` and
  `_sample_v2ce_burst_fractions()`, anchor `KEY[F-TS-SUBDIV]`

### Issue

Both modes draw a single vector of length `min_ts_steps` for the whole frame
transition and sort it:

```python
raw = torch.rand((min_ts_steps,), ...)      # random
raw = torch.sqrt(raw)                       # slope
return torch.sort(raw).values
```

Iteration `i` of the emission loop then stamps **every** active pixel with
`ts[i]`. The number of distinct timestamps in the interval is unchanged from the
legacy linear path; only their positions move.

### Why it is an issue

The stated purpose of the option is to reduce the timestamp layering artefact
that comes from assigning shared sub-frame times. Relocating the layers does not
reduce layering — the same `N` timestamps are still shared by all pixels, so the
unique-timestamp ratio, the per-pixel inter-event interval structure, and every
layering metric are unchanged in kind.

Against the reference this is a different mechanism, not an approximation of the
same one. V2CE infers timestamps per voxel from local dynamics, so independent
pixels receive independent times. Here they cannot, because a single sorted
vector is shared frame-wide.

The `slope` mode's `sqrt(U)` transform compounds this: it is a fixed end-biased
density with no dependence on the local signal, whereas the reference's slope
conditioning is the entire point of the mechanism. The same bias is applied to
every pixel regardless of what that pixel is doing.

### Why it is filed as a defect rather than a limitation

The code and current docs are honest about the mechanism. The finding is
recorded because the capability language elsewhere has repeatedly drifted toward
"de-layering", and because the metric used to evaluate it
(`timestamp_layering_score`) is computed on a stream that is sorted first — see
[TOOL-002](tooling_and_benchmarks.md#tool-002) — so the benchmark cannot
distinguish the two situations.

### Suggested fix

Implement per-pixel candidate timing, which is Stage 2 of the V2CE plan:
flatten candidates to explicit `(x, y, polarity, ordinal)` tuples and sample
timestamps independently per pixel and polarity. Preserve count, coordinates and
polarity when refractory is disabled, so the change is provably timing-only.

Until that lands, keep the description as "relocates frame-global layers" in
every user-facing document, and do not use `timestamp_layering_score` as
evidence of improvement while [TOOL-002](tooling_and_benchmarks.md#tool-002)
stands.

---

## V2CE-002

**Refractory gating uses nominal spacing, not actual gaps**

- **Severity:** High
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `_sample_signal_timestamps()` returns
  `ts_step`; consumed by the legacy refractory guard in `generate_events()`,
  anchor `KEY[H-REFRACTORY-EXT]`

### Issue

`_sample_signal_timestamps` returns `ts_step = delta_time / min_ts_steps` — the
*linear* spacing — regardless of mode. The refractory branch is entered only
when that nominal value is smaller than the configured period:

```python
if (not self.iebcs_refractory_state_coupling) and self.refractory_period_s > ts_step:
```

Under the V2CE modes the actual timestamps are random, so realised gaps between
adjacent layers are unrelated to `ts_step` and can be far smaller.

### Why it is an issue

The decision of *whether* to enforce refractory filtering is made from a number
that no longer describes the stream being filtered. When the nominal spacing
happens to exceed the configured period, the guard concludes filtering is
unnecessary and skips it entirely — while randomly adjacent layers may sit
microseconds apart.

The result is a stream that violates the refractory period the user explicitly
configured, on a code path whose only job is to enforce it. Because the guard is
an optimisation ("only filter when the period could possibly bind"), its
premise has to hold for the optimisation to be sound; under non-uniform timing
it does not.

Sorted uniform draws make small gaps routine rather than exceptional: for `N`
sorted uniforms on an interval of length `T`, the minimum spacing has expected
value on the order of `T/N²`, so with 20 layers in a 20 ms frame the smallest
gap averages tens of microseconds against a nominal spacing of 1 ms.

The prior review reports a probe configured for 3.5 ms refractory producing
same-pixel gaps near 72.5 µs, which is consistent with that.

### Suggested fix

Filter against actual candidate timestamps rather than nominal spacing:

- Remove the `refractory_period_s > ts_step` shortcut, or restrict it to the
  linear path where its premise is true.
- Compare each candidate's own timestamp to that pixel's `timestamp_mem`, which
  the loop already maintains — the per-iteration comparison is already written
  in terms of `ts[i]`, so the fix is mostly deleting the guard rather than
  adding logic.
- Add a test that enables a V2CE mode with a large refractory period and asserts
  the minimum realised same-pixel gap satisfies it.

The cost is that refractory filtering runs on frames where it previously
short-circuited. Measure it before optimising it back — with the guard removed
the branch is a vectorised comparison per iteration, which is unlikely to
dominate.

This interacts with [IEBCS-004](iebcs_extensions.md#iebcs-004): both concern
which refractory implementation is active and on what timestamps, so the two
should be designed together even if fixed separately.
