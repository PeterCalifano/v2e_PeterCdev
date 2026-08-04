# Findings: State Lifecycle and Parameter Ownership

Scope: emulator object lifetime — `reset()`, `set_dvs_params()`, and which
attributes are created, replaced, or invalidated by each. Defects here affect
callers that reuse one `EventEmulator` across sequences, which is the pattern
`EventDataGenerationLib` uses for persistent emulation contexts.

Conventions and severity definitions: [`README.md`](README.md).

---

## LIFE-001

**`reset()` then reuse raises `TypeError`**

- **Severity:** Critical
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `reset()` and `_init()`, anchor
  `KEY[C-THRESH-MISMATCH]`

### Issue

`__init__` stores `self.pos_thres` as a scalar float. On the first frame,
`_init()` replaces it with a per-pixel tensor:

```python
self.pos_thres = torch.normal(self.pos_thres, self.sigma_thres,
                              size=first_frame_linear.shape, ...)
```

`reset()` clears the frame states but leaves `self.pos_thres` as that tensor.
The next frame re-enters `_init()`, which calls `torch.normal` again — now with
a **tensor** as `mean` while still passing `size=`. That overload does not
exist.

### Why it is an issue

`reset()` is public, is documented as "Reset model state so the next frame
reinitializes the emulator", and is the only offered way to start a new
sequence on an existing emulator. As written it makes the emulator unusable
rather than reusable, for every configuration with `sigma_thres > 0` — which is
the default (`0.03`) and every realistic sensor model. The failure is a hard
crash on the *next* frame, so it surfaces far from the `reset()` call that
caused it.

The same re-entry problem applies to any `_init()` state derived from a value
that `_init()` itself overwrites.

### Evidence

```text
after 2 frames: type(pos_thres) = Tensor  shape = torch.Size([8, 8])
after reset():  type(pos_thres) = Tensor
generate_events(...) -> RAISED TypeError:
    normal(): argument 'mean' (position 1) must be float, not Tensor
```

### Suggested fix

`reset()` must restore every attribute `_init()` consumes back to its
constructor value. Concretely, keep the nominal scalars as the single source of
truth and rebuild from them:

```python
def reset(self):
    ...
    self.pos_thres = self.pos_thres_nominal
    self.neg_thres = self.neg_thres_nominal
    self.timestamp_mem = None
    self.noise_rate_array = None
    self.pos_thres_pre_prob = None
    self.neg_thres_pre_prob = None
```

The durable version of this fix is to stop letting `_init()` read and write the
same attribute: have it derive `self.pos_thres` from `self.pos_thres_nominal`
explicitly, so re-entry is idempotent regardless of what `reset()` clears. Add
a regression that runs a sequence, calls `reset()`, and runs a second sequence.

---

## LIFE-002

**`reset()` does not rewind `t_previous`**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `reset()`

### Issue

`reset()` clears `frame_counter` and the frame state tensors but leaves
`self.t_previous` at the last processed timestamp. `generate_events()` rejects
any frame not strictly later than `t_previous`.

### Why it is an issue

The natural use of `reset()` is to begin a new recording, and a new recording
usually starts at or near `t = 0`. That is precisely the case the stale
`t_previous` rejects. A caller that resets after a 5-second sequence must then
feed timestamps above 5 s, which silently couples the time base of two logically
independent sequences and corrupts every absolute-time mechanism downstream
(histogram noise scheduling, refractory release times, writer timestamp ranges).

Note this interacts with [LIFE-001](#life-001): callers who work around the
`ValueError` by continuing forward in time then hit the `TypeError` instead.

### Evidence

```text
t_previous before reset: 5.0
t_previous after  reset: 5.0
generate_events(frame, 0.0) -> RAISED ValueError:
    this frame time=0.0 must be later than previous frame time=0.01
```

### Suggested fix

Reset the time origin together with the rest of the state:

```python
def reset(self):
    ...
    self.t_previous = 0.0
```

If some caller genuinely needs a continuous clock across a reset, make that an
explicit argument (`reset(keep_time=False)`) rather than an accident of which
attributes the method forgot. Whichever is chosen, state the time-base contract
in the `reset()` docstring, because [IEBCS-001](iebcs_extensions.md#iebcs-001)
shows the repository already has one unresolved time-origin assumption.

---

## LIFE-003

**`set_dvs_params()` mid-stream raises `AttributeError`**

- **Severity:** Critical
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `set_dvs_params()` and `_init()`, anchor
  `KEY[F-LEAK-FPN]`

### Issue

`_init()` creates `self.noise_rate_array` only when `leak_rate_hz > 0`.
`set_dvs_params('noisy')` sets `leak_rate_hz = 0.1` after `_init()` has already
run. If the emulator was constructed with `leak_rate_hz = 0`, the array was
never created, and the next frame calls `subtract_leak_current(...,
noise_rate_array=self.noise_rate_array, ...)` on a missing attribute.

### Why it is an issue

`set_dvs_params()` is a public convenience for switching between named sensor
presets, and switching *to* the noisier preset is its main use. The method
mutates parameters that `_init()` treats as construction-time-only, without
re-running or invalidating the derived state. The result is a crash whose
message (`no attribute 'noise_rate_array'`) points at leak modelling rather
than at the preset switch that caused it.

### Evidence

```text
emulator constructed with leak_rate_hz=0
set_dvs_params("noisy")            # sets leak_rate_hz = 0.1
generate_events(...) -> RAISED AttributeError:
    'EventEmulator' object has no attribute 'noise_rate_array'
```

### Suggested fix

Make `set_dvs_params()` invalidate everything it invalidates, by routing
through the same reinitialisation path rather than assigning attributes
piecemeal:

```python
def set_dvs_params(self, model):
    ...                      # assign the preset scalars
    self.low_pass_filter = LowPassFilter(cutoff_hz=self.cutoff_hz)
    self.reset()             # forces _init() on the next frame
```

That requires [LIFE-001](#life-001) and [LIFE-002](#life-002) to be fixed first,
otherwise it trades one crash for another — which is the argument for treating
these three as a single change. Also declare in the docstring whether calling
`set_dvs_params()` after streaming has begun is supported at all; rejecting it
outright is a defensible alternative and is simpler to guarantee.

---

## LIFE-004

**`set_dvs_params()` discards per-pixel threshold mismatch**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `set_dvs_params()`

### Issue

The preset branches assign `self.pos_thres = 0.2` and `self.neg_thres = 0.2` as
scalars, overwriting the per-pixel tensors produced by `_init()`. They also
leave `self.pos_thres_nominal` / `neg_thres_nominal` untouched, and never call
`_refresh_threshold_probability_scales()`.

### Why it is an issue

Three separate consequences, all silent:

1. **Threshold mismatch is lost.** Per-pixel threshold dispersion is a core v2e
   model feature (`KEY[C-THRESH-MISMATCH]`). After a preset switch every pixel
   shares one threshold, so fixed-pattern noise disappears from the model
   without any log line saying so.
2. **Nominal values go stale.** `pos_thres_nominal` drives IEBCS threshold
   resampling and shot-noise probability scaling. If the preset changes the
   active threshold away from the nominal, those two mechanisms are calibrated
   against a value that is no longer in use.
3. **`pos_thres_pre_prob` is stale.** It is computed as `nominal / actual` and
   refreshed only in `_init()` and after IEBCS resampling — not here. Shot-noise
   rates therefore keep using the pre-switch ratio.

### Evidence

```text
before: pos_thres is Tensor  std=0.04746
before: pos_thres_nominal = 0.2
after : pos_thres is float   value = 0.2
after : pos_thres_nominal = 0.2   (never updated by the preset)
per-pixel mismatch lost? True
```

`_refresh_threshold_probability_scales()` is confirmed absent from
`set_dvs_params()` by inspection.

### Suggested fix

Treat the preset as setting *nominal* parameters, then rebuild the derived
per-pixel state from them:

```python
self.pos_thres_nominal = 0.2
self.neg_thres_nominal = 0.2
self.sigma_thres = 0.05
...
self.reset()          # per-pixel thresholds and pre_prob rebuilt in _init()
```

This subsumes [LIFE-003](#life-003) and removes the nominal-drift half of the
problem in the same change. Add a test asserting that after a preset switch the
thresholds are still a tensor with the preset's `sigma_thres`, and that
`pos_thres_pre_prob` is consistent with `nominal / actual`.

---

## LIFE-005

**`reset()` and active HDF5 storage have no defined contract**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, `reset()`, `prepare_storage()`,
  `_append_h5_events()`

### Issue

`reset()` rewinds `frame_counter` to zero but deliberately leaves the HDF5
datasets and the logical/written event counters in place — its own docstring
records this as intentional and unsupported.

### Why it is an issue

The combination is representable but incoherent. After a reset, new frames are
written at `frame_idx` values that restart from zero while previously written
events remain appended, so the frame index sequence stored in the file is no
longer monotonic and the frame-to-event-index mapping no longer identifies a
unique frame. Consumers that use `frame_ev_idx` to slice events per frame will
silently read the wrong slice. Documenting the hazard in a docstring does not
prevent it, because nothing in the code path refuses the combination.

### Suggested fix

Pick one contract and enforce it in code:

- **Reject** — raise if `reset()` is called while `dvs_h5_dataset is not None`.
  Simplest, and matches the fact that no current caller needs the other
  behaviour.
- **Recreate** — close and reopen the datasets, resetting
  `h5_event_logical_count`, `h5_event_written_count`, `h5_event_flush_count`
  and the buffer atomically with the model state.

Rejecting is recommended. Whichever is chosen, add a test that constructs an
emulator with `dvs_h5`, writes some frames, calls `reset()`, and asserts the
declared behaviour — the current gap is that neither behaviour is asserted
anywhere.
