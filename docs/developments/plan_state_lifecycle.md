# Plan: State Lifecycle

Scope: emulator object lifetime — construction, `_init()`, `reset()`,
`set_dvs_params()`, and the derived per-pixel state each one owns. Owns
[LIFE-001](../findings/state_lifecycle.md#life-001) through
[LIFE-005](../findings/state_lifecycle.md#life-005).

Not in scope: what the model computes once initialised.

Status: active. LIFE-001 through LIFE-004 are resolved; LIFE-005 and strict
constructor-domain validation remain open.

## Design

The root cause of LIFE-001/003/004 is shared: `_init()` reads attributes it also
writes, and `set_dvs_params()` mutates inputs that `_init()` already consumed.
Fix the ownership rule once rather than patching each symptom.

- [x] Establish that `*_nominal` attributes are the single source of truth for
      construction-time parameters, and that every derived per-pixel tensor is
      rebuilt from them.
- [x] Make `_init()` idempotent: running it twice on the same emulator must
      produce the same state as running it once on a fresh one.
- [ ] Define constructor-domain validation for direct API users. Under
      `strict_model_validity`, reject non-finite values and physically invalid
      signs for thresholds, cutoff, noise/leak rates, refractory duration, and
      extension time constants before allocating derived state.
- [x] Define the lifecycle contract: `reset()` starts a new sequence and
      rewinds model, time, and private RNG state; `set_dvs_params()` is legal
      only before initialization or after a writer-free reset.

## Tests first

- [x] Run a sequence, `reset()`, run a second sequence starting at `t = 0`
      (LIFE-001, LIFE-002).
- [x] `set_dvs_params()` between frames is rejected with an actionable error;
      the same preset succeeds before initialization and after a writer-free
      reset (LIFE-003).
- [x] After a preset switch, thresholds are still per-pixel with the preset's
      `sigma_thres`, and `pos_thres_pre_prob` equals `nominal / actual`
      (LIFE-004).
- [ ] `reset()` with an active HDF5 writer raises before changing model or
      writer state (LIFE-005).
- [x] Fixed-seed reproducibility survives a reset: two emulators, one reset and
      one fresh, produce identical streams for the same input.
- [ ] Strict constructor validation fails without partially initialized model
      or writer state; non-strict behavior remains unchanged unless the
      configuration is already rejected unconditionally.

## Implementation

- [x] `reset()` restores `pos_thres`/`neg_thres` from nominal and clears
      `timestamp_mem`, `noise_rate_array`, the `*_pre_prob` scales, and the
      `scidvs_*` state (LIFE-001).
- [x] `reset()` rewinds `t_previous` and restores the private generators to
      their construction-time states (LIFE-002).
- [x] `set_dvs_params()` rejects initialized state, assigns nominal values, and
      rebuilds the low-pass filter without assigning derived tensors piecemeal
      (LIFE-003, LIFE-004).
- [ ] Enforce the chosen `reset()`-during-recording contract in code, not only
      in the docstring (LIFE-005).

## Decisions required

- [x] `reset()` rewinds the clock unconditionally; no `keep_time` compatibility
      option is added.
- [x] `set_dvs_params()` after streaming begins is rejected.
- [x] `reset()` during active HDF5 recording is rejected. The unnecessary
      reset-before-cleanup call in `EventDataGenerationLib` must be removed in
      that repository before this contract lands.

## Gate

- [ ] All tests above pass.
- [x] `conda run -n v2e python -m pytest -q` (`102 passed` in the
      isolated lifecycle candidate).
- [ ] `EventDataGenerationLib`'s persistent emulation context is checked against
      the chosen contract, since it is the caller that reuses emulators. Its
      five applicable direct/context tests pass; the fixed-seed pool test still
      depends on the unreviewed buffered-HDF5 statistics API.
- [x] Mark resolved findings in
      [`../findings/state_lifecycle.md`](../findings/state_lifecycle.md).
