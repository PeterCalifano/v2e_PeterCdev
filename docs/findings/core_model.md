# Findings: Core Model and Numerics

Scope: the default v2e sensor path — lin-log encoding, photoreceptor low-pass
filtering, and HDR preprocessing. Defects here affect runs with **no** opt-in
extension enabled.

Conventions and severity definitions: [`README.md`](README.md).

---

## CORE-001

**Scalar low-pass path lacked an `eps <= 1` clamp**

- **Severity:** Medium
- **Status:** Resolved (functional regression)
- **Where:** `v2ecore/emulator_utils.py`, `LowPassFilter.__call__`, anchor
  `KEY[E-LPF-IIR]` (scalar branch)

### Issue

The intensity-dependent branch computes `eps = inten01 * delta_time / tau` and
then applies `torch.clamp(eps, max=1.0)`. The scalar branch computes
`eps = delta_time / tau` and passes it straight to `lerp_` with no clamp.

### Why it is an issue

`lerp` with a weight greater than 1 extrapolates instead of interpolating. The
filter state overshoots its target and, for weights above 2, oscillates with
growing amplitude — the update is no longer a low-pass filter but an unstable
recurrence. The scalar branch is the one selected whenever intensity scaling is
off, so this is reachable on a default-ish configuration, not only on an
experimental flag. The tensor branch already treats `eps > 1` as a condition
worth clamping *and* warning about, so the two branches encode contradictory
opinions about the same quantity.

### Evidence

With `cutoff_hz=1` (`tau = 0.159 s`), `delta_time = 1.0`, state `0.0`, target
`1.0`:

```text
scalar eps path -> 6.283185     (expected within [0, 1])
tensor eps path -> 1.000000
```

The existing large-epsilon regression exercises only the tensor branch, so the
suite stays green.

### Suggested fix

Clamp in one place that both branches share, before the branch:

```python
delta_over_tau = delta_time / tau
...
eps = inten01 * delta_over_tau if inten01 is not None else delta_over_tau
eps = torch.clamp(eps, max=1.0) if torch.is_tensor(eps) else min(eps, 1.0)
```

Move the `max_eps > 0.3` diagnostic above the branch too, so an undersampled
scalar-path configuration warns exactly as the tensor path does. Add a
regression asserting the scalar result stays within `[old, target]` for
`delta_time / tau >> 1`.

### Resolution

The scalar and intensity-dependent routes now share the warning and
`eps <= 1` clamp before applying the update. The regression verifies that a
large scalar step reaches, but does not overshoot, its target. The optional
`strict_model_validity` policy raises before warning, clamping, or mutating state
when either route computes `eps > 1`; `0.3 < eps <= 1` remains warning-only.

---

## CORE-002

**`LowPassFilter` mutated caller state on one path only**

- **Severity:** Medium
- **Status:** Resolved (functional regression)
- **Where:** `v2ecore/emulator_utils.py`, `LowPassFilter.__call__`

### Issue

The scalar branch calls `lp_log_frame.lerp_(...)` and returns the same object.
The tensor branch evaluates `lp_log_frame + eps * (...)` and returns a new
tensor, leaving the argument untouched.

### Why it is an issue

The function's aliasing contract depends on the value of an unrelated argument
(`inten01`). Any caller that keeps a reference to the tensor it passed in — for
a diagnostic, a state recording, a comparison against the pre-update value —
silently gets different semantics depending on whether intensity scaling is
enabled. This is exactly the class of bug that survives a green suite: both
paths produce the right *return value*, and only the caller's other reference
disagrees. It also blocks any future attempt to reuse this helper from a shared
backend, where in-place mutation of a caller-owned buffer must be explicit.

### Evidence

```text
scalar path: caller tensor mutated? True  | returns same object? True
tensor path: caller tensor mutated? False | returns same object? False
```

### Suggested fix

Pick one contract and document it in the docstring. The lower-risk choice is
"never mutates the argument": drop `lerp_` for `torch.lerp` on the scalar path
and accept the allocation, since the historical 2.29x micro-benchmark that
motivated `lerp_` is superseded by [TOOL-005](tooling_and_benchmarks.md#tool-005)-era
measurement caveats and was never shown to matter end to end. If the in-place
form is retained for performance, rename the method to make the mutation
visible (`update_` / `apply_inplace`) and make **both** branches in-place so the
contract stops depending on `inten01`.

### Resolution

The selected contract is in-place mutation for both routes. Tests assert that
each route returns the exact supplied state object and produces the expected
filtered values.

---

## CORE-003

**`apply_low_pass_filter` silently ignored `filter_tau_const`**

- **Severity:** High
- **Status:** Resolved (functional regression)
- **Where:** `v2ecore/emulator_utils.py`, `apply_low_pass_filter`

### Issue

`apply_low_pass_filter` accepts both `filter_tau_const` and an optional
pre-built `low_pass_filter`. It uses `filter_tau_const` only when it constructs
a filter itself. When a `low_pass_filter` instance is supplied, the final call

```python
return low_pass_filter(
    log_new_frame=..., lp_log_frame=..., inten01=..., delta_time=delta_time)
```

omits `filter_tau_const` entirely, even though `LowPassFilter.__call__` accepts
that parameter and would honour it.

### Why it is an issue

A caller that passes an explicit time constant gets the instance's own `tau`
instead, with no error and no warning. The parameter is accepted, type-checked
by the signature, and then discarded. Because `tau` sets the photoreceptor
bandwidth, the resulting filter can be off by orders of magnitude while every
downstream value still looks plausible. Silent parameter drops are worse than
crashes here: the run completes, the events look like events, and nothing in
the output indicates the requested model was not the model that ran.

### Evidence

Filter built with `cutoff_hz=100` (`tau ≈ 1.59e-3`), then asked for
`filter_tau_const=1.0`, with `delta_time = 1e-4`:

```text
instance supplied, no filter_tau_const : 0.06283186
instance supplied + filter_tau_const=1 : 0.06283186   <- request ignored
no instance,        filter_tau_const=1 : 0.00010000   <- request honoured
```

A 628x difference in the update weight, from a parameter the API advertises.

### Suggested fix

Forward the argument:

```python
return low_pass_filter(
    log_new_frame=log_new_frame,
    lp_log_frame=lp_log_frame,
    inten01=inten01,
    delta_time=delta_time,
    filter_tau_const=filter_tau_const)
```

Then add a test asserting the two supply routes agree. If overriding an
instance's `tau` per call is *not* intended, the opposite fix is equally valid
and arguably cleaner — raise `ValueError` when both `low_pass_filter` and
`filter_tau_const` are given — but the current silent-drop behaviour is not
defensible either way.

Note the cache in `_LOW_PASS_FILTER_CACHE` is keyed on
`(cutoff_hz, filter_tau_const)`, so the self-constructed route is already
correct; only the injected-instance route is wrong.

### Resolution

The wrapper forwards `filter_tau_const` and `strict_model_validity` to supplied
filter instances. Regressions compare the supplied-instance and internally
constructed routes for the same explicit time constant and strict policy.

---

## CORE-004

**float64 lin-log rationale is not supported by its own rounding grid**

- **Severity:** Open question
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator_utils.py`, `Map_linear_to_log_luminance`,
  anchor `KEY[D-LINLOG]`

### Issue

The function casts input to float64, rounds the result onto a `1e-8` decimal
grid "to prevent numerical drift that suppresses OFF events", then returns
`.float()` (float32). A `TODO (TBC)` warns that moving to float32 throughout
would break OFF-event generation.

The stated mechanism does not survive inspection: the `1e-8` grid is finer than
the float32 spacing the result is immediately quantised onto.

### Why it is an issue

This is a documented, load-bearing constraint — `CLAUDE.md`, the performance
plan, and the historical optimization report all cite it as the reason a float32
lin-log path is off the table. If the rationale as written is wrong, the
constraint may be either unnecessary (a real speedup is being declined for no
reason) or *right for a different reason* that nobody has recorded, in which
case the next person to touch it will remove the wrong line. Either way the
repository is carrying a performance restriction it cannot currently justify.

This is filed as an open question rather than a defect: the rounding may still
matter through a mechanism other than final precision — for example by making
values that differ in the last float64 bits land on a common grid point
*before* the cast, so they convert to the identical float32 rather than
straddling a rounding boundary.

### Evidence

```text
returned dtype                             : torch.float32
max |round(1e-8 grid) - raw| in float64    : 4.84e-09
float32 spacing near log(255) = 5.541      : 4.77e-07
1e-8 grid finer than float32 spacing?      : True
```

The rounding moves values by at most ~5e-9, roughly 100x below the ~4.8e-7
float32 quantum they are then snapped to.

### Suggested fix

Do not change the dtype on the strength of this note alone. Instead settle it
with an experiment and record the answer next to the code:

1. Build a fixture that reproduces the original failure — an ON burst followed
   by the OFF burst that should mirror it — and confirm it fails with the
   rounding removed but float64 retained.
2. If it does, the rounding matters and the comment should say *why* in terms
   of pre-cast grid alignment, not post-cast precision.
3. If it does not, test the full float32 path against the regression suite and
   retire the `TODO (TBC)`.

Whichever way it resolves, replace the current comment with the measured
conclusion, and update the corresponding claim in
[`../repo_capabilities_status.md`](../repo_capabilities_status.md) and
`CLAUDE.md`.

---

## CORE-005

**HDR preprocessing has no dark-response floor**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `generate_events`, anchor
  `KEY[E-INTEN-SCALE]`; `v2ecore/emulator_utils.py`, `rescale_intensity_frame`

### Issue

The non-HDR path derives the intensity scale through
`rescale_intensity_frame()`, i.e. `(frame + 20) / 275`, which is bounded below
by `20/275 ≈ 0.073` at black. The HDR path uses `clamp(frame, 0, 1)` directly,
so `inten01` reaches exactly `0`.

### Why it is an issue

`inten01` multiplies the low-pass update weight
(`eps = inten01 * delta_time / tau`). At `inten01 = 0` the weight is zero, so
the photoreceptor state stops evolving entirely: a pixel that transitions from
bright to true black freezes its filtered value and can never recover, because
recovery would require the very update that the zero weight suppresses. The
non-HDR path avoids this by construction; the HDR path reintroduces it. The
`+20` offset in the non-HDR path is explicitly commented "make sure we get no
zero time constants", which is the same hazard, so the two paths disagree about
a constraint one of them names outright.

### Why the offset exists at all is itself undocumented

`rescale_intensity_frame` carries a `# DEVNOTE why +20?` comment. The
constant coincides with the lin-log transition threshold (`threshold=20` in
`Map_linear_to_log_luminance`), which is a plausible but unverified rationale.

### Suggested fix

Give the HDR path an explicit, documented floor with the same physical meaning
as the non-HDR one, rather than copying the magic constant:

```python
inten01 = torch.clamp(frame, min=DARK_FLOOR, max=1.0)   # DARK_FLOOR > 0
```

Choose `DARK_FLOOR` from the intended minimum photoreceptor bandwidth and state
that derivation in `core_model_mapping.md` next to the `KEY[E-INTEN-SCALE]`
anchor. Add a bright-to-black HDR regression asserting the low-pass state keeps
evolving across the transition. While doing so, resolve the `+20` DEVNOTE so
both paths cite the same justification.
