# Findings: Tooling, Benchmarks, and Packaging

Scope: everything outside the sensor model — benchmark harnesses, the
photoreceptor-noise estimator's numerics, dead code, and packaging metadata.

Conventions and severity definitions: [`README.md`](README.md).

---

## TOOL-001

**Comparative benchmark resolves the wrong repository root**

- **Severity:** Medium
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/benchmarks/benchmark_error_models_eventstream.py`

### Issue

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
```

The file lives at `v2ecore/benchmarks/`, so `parents[1]` is `v2ecore/`, not the
repository root. Its sibling `benchmark_emulator.py` correctly uses
`parents[2]`.

### Why it is an issue

`REPO_ROOT` is used for two things, and both are wrong:

1. It is inserted into `sys.path`, so imports resolve against `v2ecore/` rather
   than the repository. This happens to work when the script is run from the
   repository root because the correct path is already on `sys.path`, which is
   exactly why the bug has survived — it fails only when run from elsewhere.
2. It sets the default output directory to
   `v2ecore/output/benchmarks_comparative`, so benchmark artefacts are written
   *inside the package directory* instead of the repository's `output/` tree.

The inconsistency between two sibling scripts is itself the strongest evidence:
one of them must be wrong, and the one that disagrees with its own directory
depth is the one to fix.

### Evidence

```text
comparative REPO_ROOT (parents[1]): .../v2e/v2ecore
emulator     REPO_ROOT (parents[2]): .../v2e
actual repo root                  : .../v2e
```

### Suggested fix

Change to `parents[2]` to match `benchmark_emulator.py`, then add the test the
consolidation plan already calls for: assert `REPO_ROOT` contains a known
repository-root marker (`pyproject.toml`) so the two scripts cannot drift apart
again. Any existing artefacts under `v2ecore/output/` should be moved or
deleted, and `v2ecore/output/` added to `.gitignore` if it is not already
covered.

---

## TOOL-002

**Benchmark sorts the stream before measuring stream validity**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/benchmarks/benchmark_error_models_eventstream.py`,
  `Run_profile()`

### Issue

Concatenated packets are sorted by timestamp before stream-shape metrics are
computed.

### Why it is an issue

Sorting makes every ordering metric measured afterwards tautological: a sorted
array is monotonic by construction, so the benchmark cannot report the ordering
defect described in [STREAM-001](stream_and_io.md#stream-001) — it repairs it
first, then measures the repair.

This converts the comparative benchmark from a detector into a concealer for
the single highest-severity stream defect currently open. It is also why that
defect went unreported by the benchmarking work even though the benchmark
exercises the exact configurations that trigger it.

`timestamp_layering_score` is affected in a related way: computed on a sorted
stream, it cannot distinguish an emission-order problem from a timing
distribution, which weakens it as evidence for
[V2CE-001](v2ce_timing.md#v2ce-001).

### Suggested fix

Measure raw stream validity **before** any normalisation:

1. On the concatenated raw stream, record monotonicity violations — count,
   largest backward step, and the packet boundaries where they occur — as
   first-class reported metrics.
2. Only then produce a sorted copy, and use it exclusively for metrics that are
   genuinely order-invariant (distribution distances, rate maps, counts).
3. Label sorted-derived metrics in the JSON/CSV output so a reader can tell
   which numbers describe the emitted stream and which describe a repaired one.

---

## TOOL-003

**Dead polynomial helper duplicated inline**

- **Severity:** Low
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator_utils.py`,
  `PhotoreceptorNoiseVoltageEstimator._compute_vn_from_log_rate_per_hz` and
  `__call__`, anchor `KEY[G-PHOTO-VRMS-FIT]`

### Issue

`_compute_vn_from_log_rate_per_hz` implements the Graca–Delbruck fit

```python
y = -0.0026 * x ** 3 - 0.036 * x ** 2 - 0.1949 * x + 0.321
```

and is never called. `__call__` contains the same expression inline, followed by
`# DOUBT what's this?`.

### Why it is an issue

Two copies of a fitted physical relation will diverge the first time one is
corrected. The inline copy carries a comment indicating the author did not know
what it computed — which is precisely the information the named helper (with its
paper citation and axis definitions) exists to supply. The documentation is
attached to the copy nobody reads.

### Suggested fix

Delete the inline duplicate and call the helper:

```python
thr_per_vn = 10 ** self._log10_thr_per_vn(x)   # or reuse the existing helper
vn = float(np.mean(mins / thr_per_vn))
```

Resolve the `# DOUBT` comment by pointing at the helper's docstring and the
`media/noise_event_rate_simulation.xlsx` derivation. Keep the anchor comment
`KEY[G-PHOTO-VRMS-FIT]` on the surviving copy so `core_model_mapping.md` still
resolves.

---

## TOOL-004

**Noise-rate warning text contradicts its own threshold**

- **Severity:** Low
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator_utils.py`,
  `PhotoreceptorNoiseVoltageEstimator.__call__`

### Issue

```python
if rate_per_bw > 0.5:
    logger.warning(
        f'Shot noise rate per hz of bandwidth is larger than 0.1 (...)')
```

The condition tests `0.5`; the message says `0.1`.

### Why it is an issue

A user tuning `--shot_noise_rate_hz` against this warning will target the wrong
value, and will see no warning in the range `0.1 < rate_per_bw <= 0.5` where the
message implies one. Minor in isolation, but it is a warning specifically about
model validity, so acting on it incorrectly degrades the simulation silently.

### Suggested fix

Decide which number is intended — `0.5` matches the `x = log10(rate_per_bw)`
range checks that follow (`x > 0.0` warns separately), suggesting the threshold
is right and the text is stale — then make the message quote the constant rather
than hard-code it:

```python
RATE_PER_BW_WARN = 0.5
if rate_per_bw > RATE_PER_BW_WARN:
    logger.warning(
        f'Shot noise rate per Hz of bandwidth {rate_per_bw:.3g} is larger '
        f'than {RATE_PER_BW_WARN} (...)')
```

---

## TOOL-005

**Noise calibration loop length is unbounded in Python**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator_utils.py`,
  `PhotoreceptorNoiseVoltageEstimator.__call__`

### Issue

The estimator simulates the IIR response over a fixed number of time constants
using a scalar Python loop:

```python
t = np.arange(0, 1000 * tau, dt)
...
for i in range(1, len(rin)):
    rout[i] = rout[i-1] * (1 - eps) + rin[i] * eps
```

Its length is `1000 * tau / dt = 1000 * sample_rate_hz / (2 * pi * f3db)` —
proportional to sample rate and inversely proportional to cutoff frequency, with
no cap.

### Why it is an issue

Both driving parameters are user-controlled and can legitimately take values
that make this enormous. A low photoreceptor cutoff combined with a high
upsampled frame rate — which is the configuration `--auto_timestamp_resolution`
naturally produces — pushes the iteration count into the millions. At
`cutoff_hz = 1` and a 10 kHz effective sample rate the loop runs ~1.6 million
scalar Python iterations, plus two arrays of that length.

The failure mode is a long unexplained stall during setup, before any frame is
processed, with no progress indication. Users would reasonably read it as a
hang. The existing `eps > 0.1` warning fires for the opposite condition (too few
samples per time constant), so nothing warns about the expensive direction.

### Suggested fix

Two independent improvements:

1. **Vectorise.** The recurrence is a first-order IIR whose only purpose is to
   measure the output RMS. `scipy.signal.lfilter([eps], [1, -(1-eps)], rin)`
   computes it in compiled code, or the closed-form variance ratio
   `eps / (2 - eps)` avoids the simulation entirely for white input — worth
   checking against the simulated value before replacing it.
2. **Bound the work.** Cap the sample count and warn when the cap binds, since
   the RMS estimate converges long before 1000 time constants; a few thousand
   samples is ample for a variance estimate.

Also reconsider `sample_rate_rel_tolerance = 0.1` on the cache: a 10 % change in
sample rate currently returns a cached `vn` rather than recomputing. That is a
reasonable trade given the cost above, but it should be a documented choice
rather than an implicit one — and it becomes less necessary once the computation
is cheap.

---

## TOOL-006

**Package version disagrees with the changelog**

- **Severity:** Low
- **Status:** Confirmed (inspection)
- **Where:** `pyproject.toml`, `CHANGELOG.md`

### Issue

`pyproject.toml` reports version `1.5.1`. `CHANGELOG.md` begins at `v1.6.2`.

### Why it is an issue

Anything that resolves the installed version — dependency pins in
`EventDataGenerationLib`, provenance recorded in generated datasets, bug reports
— gets a version that does not correspond to any documented release. Since the
workspace integration story already depends on v2e being declared as a packaged
dependency, an unreliable version string blocks that from being meaningful.

### Suggested fix

Set `pyproject.toml` to the changelog's newest released version, and add the
release step to whatever checklist governs tagging. A test asserting
`importlib.metadata.version("v2e")` matches the top entry of `CHANGELOG.md`
makes the two impossible to desynchronise again.

---

## TOOL-007

**Parallel histogram branch is never exercised**

- **Severity:** Low
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/v2e_utils.py`, `hist2d_numba()` /
  `hist2d_numba_parallel()`

### Issue

```python
if tracks.shape[1] > 1_000_000:
    return hist2d_numba_parallel(tracks, bins, ranges)
return hist2d_numba_seq(tracks, bins, ranges)
```

No test or benchmark supplies more than a million tracks, so the parallel branch
never runs.

### Why it is an issue

Untested compiled-kernel code that is reachable in production is a latent
failure: the first run that crosses the threshold is the first execution ever,
during a large real workload rather than in CI. The parallel implementation
maintains per-chunk partial histograms and merges them, which is exactly the
kind of code where an indexing error survives review and shows up as a subtly
wrong histogram rather than a crash.

The historical justification for leaving it dormant — a TBB version too old to
benefit — no longer applies to the current environment, so the branch is now
both reachable and unvalidated.

### Suggested fix

Add a correctness test that crosses the threshold and asserts the parallel
result equals the sequential result exactly for the same input. Keep the input
synthetic and shaped to stay fast (a narrow bin range with 1,000,001 tracks
suffices). Separately, re-measure whether the parallel path is actually faster
on the current runtime before keeping the auto-selection; if it is not, delete
the branch rather than carrying untested code for a benefit that no longer
exists.
