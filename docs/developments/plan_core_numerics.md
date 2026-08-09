# Plan: Core Numerics

Scope: core sensor numerics — low-pass filtering and its opt-in strict-validity
policy, lin-log encoding, HDR preprocessing, and preservation of HDR frames
along the no-SloMo CLI path. Owns [CORE-001](../findings/core_model.md#core-001)
through [CORE-006](../findings/core_model.md#core-006).

Not in scope: opt-in IEBCS/V2CE sensor effects, and anything about object
lifetime (see [`plan_state_lifecycle.md`](plan_state_lifecycle.md)).

Status: active. CORE-001 through CORE-003 and CORE-006 are complete; CORE-004
and CORE-005 remain open.

## Tests first

- [x] Scalar low-pass with `delta_time / tau >> 1` must land within
      `[old, target]` (CORE-001).
- [x] Scalar and tensor low-pass paths must agree on whether the caller's tensor
      is mutated (CORE-002).
- [x] `apply_low_pass_filter` must produce identical results whether the filter
      is constructed internally or supplied with the same `filter_tau_const`
      (CORE-003).
- [x] With `strict_model_validity=True`, scalar and per-pixel low-pass updates
      with `eps > 1` must raise before warning or clamping; `0.3 < eps <= 1`
      remains warning-only.
- [x] Without strict validity, the same `eps > 1` updates must retain the
      bounded legacy-compatible clamp.
- [x] A float32 TIFF sequence processed by `--disable_slomo --hdr` must retain
      sub-8-bit frame contrast in HDF5 and produce the same events as direct
      `EventEmulator` input (CORE-006).
- [ ] Bright-to-black HDR transition must leave the low-pass state still
      evolving on the following frames (CORE-005).
- [ ] Lin-log ON-then-OFF mirror fixture that fails when the `1e-8` rounding is
      removed, to establish whether the float64 requirement is real (CORE-004).

## Implementation

- [x] Hoist the `eps` clamp and the `max_eps > 0.3` warning above the
      scalar/tensor branch so both share them (CORE-001).
- [x] Make both `LowPassFilter.__call__` branches update `lp_log_frame` in place
      and return that same tensor (CORE-002).
- [x] Forward `filter_tau_const` to the supplied filter instance (CORE-003).
- [x] Add `strict_model_validity` to the CLI and direct `EventEmulator` API,
      propagate it to `LowPassFilter`, and reject an update before its
      `eps > 1` value would be clamped.
- [x] Keep no-SloMo frames in their existing `.npy` representation through the
      emulation loop instead of serializing them through PNG (CORE-006).
- [ ] Give the HDR path an explicit non-zero dark floor derived from the
      intended minimum photoreceptor bandwidth (CORE-005).
- [ ] Resolve the `# DEVNOTE why +20?` in `rescale_intensity_frame` so both
      intensity paths cite the same justification (CORE-005).
- [ ] Under strict validity, reject non-floating-point frames supplied with
      `hdr=True` instead of only warning before applying HDR preprocessing.

## Merge-artifact repair

- [x] Add functional regressions for one-step SCIDVS decay, one-threshold
      comparator-memory advancement, and one warning-budget update per empty
      frame.
- [x] Restore one explicit Euler SCIDVS decay term and one ON comparator-memory
      update after merge `f026560` duplicated both state transitions.
- [x] Remove the neighboring duplicate assignments, model-state errors,
      high-event warnings, event-map comments, and shot-noise conversion from
      the same merge insertion.

## Decisions required

- [x] CORE-002: use one in-place update contract for both branches. This
      preserves the asserted scalar behavior and avoids an extra allocation.
- [ ] CORE-004: does the lin-log rounding survive its own experiment? Record the
      measured answer in the code comment and retire the `TODO (TBC)` either way.
- [ ] CORE-005: what dark floor value, and on what physical grounds?

## Gate

- [x] Focused RED/GREEN tests for CORE-001 through CORE-003 and CORE-006.
- [x] `conda run -n v2e python -m pytest -q` (`95 passed` in the isolated
      staged tree; `97 passed` in the complete development tree).
- [x] Update the `KEY[E-LPF-EPS]` prose in
      [`../core_model_mapping.md`](../core_model_mapping.md) to match the fixed
      behavior. `KEY[E-INTEN-SCALE]` remains open with CORE-005.
- [x] Mark the resolved findings `Resolved` in
      [`../findings/core_model.md`](../findings/core_model.md).
- [x] Verify direct-filter, direct-emulator, and CLI flag propagation for
      `strict_model_validity` without asserting tunable configuration values.
- [x] Verify the isolated merge-artifact candidate with the full conda suite
      (`110 passed`).
