# Plan: State Lifecycle

Scope: emulator object lifetime — construction, `_init()`, `reset()`,
`set_dvs_params()`, and the derived per-pixel state each one owns. Owns
[LIFE-001](../findings/state_lifecycle.md#life-001) through
[LIFE-005](../findings/state_lifecycle.md#life-005).

Not in scope: what the model computes once initialised.

Status: open. No prerequisite. High priority — LIFE-001 and LIFE-003 are
crashes on public API.

## Design

The root cause of LIFE-001/003/004 is shared: `_init()` reads attributes it also
writes, and `set_dvs_params()` mutates inputs that `_init()` already consumed.
Fix the ownership rule once rather than patching each symptom.

- [ ] Establish that `*_nominal` attributes are the single source of truth for
      construction-time parameters, and that every derived per-pixel tensor is
      rebuilt from them.
- [ ] Make `_init()` idempotent: running it twice on the same emulator must
      produce the same state as running it once on a fresh one.
- [ ] Define whether `reset()` rewinds the time origin, and whether
      `set_dvs_params()` is legal after streaming has begun.

## Tests first

- [ ] Run a sequence, `reset()`, run a second sequence starting at `t = 0`
      (LIFE-001, LIFE-002).
- [ ] `set_dvs_params()` between frames, with the emulator constructed at
      `leak_rate_hz = 0` and the preset enabling leak (LIFE-003).
- [ ] After a preset switch, thresholds are still per-pixel with the preset's
      `sigma_thres`, and `pos_thres_pre_prob` equals `nominal / actual`
      (LIFE-004).
- [ ] `reset()` with an active HDF5 writer behaves as declared — raises, or
      produces a coherent file (LIFE-005).
- [ ] Fixed-seed reproducibility survives a reset: two emulators, one reset and
      one fresh, produce identical streams for the same input.

## Implementation

- [ ] `reset()` restores `pos_thres`/`neg_thres` from nominal and clears
      `timestamp_mem`, `noise_rate_array`, the `*_pre_prob` scales, and the
      `scidvs_*` state (LIFE-001).
- [ ] `reset()` rewinds `t_previous` (LIFE-002).
- [ ] `set_dvs_params()` assigns nominal values, rebuilds the low-pass filter,
      then routes through the same reinitialisation path instead of assigning
      derived state piecemeal (LIFE-003, LIFE-004).
- [ ] Enforce the chosen `reset()`-during-recording contract in code, not only
      in the docstring (LIFE-005).

## Decisions required

- [ ] Does `reset()` rewind the clock unconditionally, or is a
      `keep_time` argument needed by a real caller?
- [ ] Is `set_dvs_params()` after streaming supported, or rejected? Rejecting is
      simpler to guarantee.
- [ ] Is `reset()` during active HDF5 recording rejected or made atomic?
      Rejecting is recommended.

## Gate

- [ ] All tests above pass.
- [ ] `conda run -n v2e python -m pytest -q`.
- [ ] `EventDataGenerationLib`'s persistent emulation context is checked against
      the chosen contract, since it is the caller that reuses emulators.
- [ ] Mark resolved findings in
      [`../findings/state_lifecycle.md`](../findings/state_lifecycle.md).
