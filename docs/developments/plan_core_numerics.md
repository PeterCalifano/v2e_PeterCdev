# Plan: Core Numerics

Scope: the default sensor path only — low-pass filtering, lin-log encoding, HDR
preprocessing. Owns [CORE-001](../findings/core_model.md#core-001) through
[CORE-005](../findings/core_model.md#core-005).

Not in scope: anything requiring an opt-in flag, and anything about object
lifetime (see [`plan_state_lifecycle.md`](plan_state_lifecycle.md)).

Status: open. No prerequisite — this plan is independent and can start first.

## Tests first

- [ ] Scalar low-pass with `delta_time / tau >> 1` must land within
      `[old, target]` (CORE-001).
- [ ] Scalar and tensor low-pass paths must agree on whether the caller's tensor
      is mutated (CORE-002).
- [ ] `apply_low_pass_filter` must produce identical results whether the filter
      is constructed internally or supplied with the same `filter_tau_const`
      (CORE-003).
- [ ] Bright-to-black HDR transition must leave the low-pass state still
      evolving on the following frames (CORE-005).
- [ ] Lin-log ON-then-OFF mirror fixture that fails when the `1e-8` rounding is
      removed, to establish whether the float64 requirement is real (CORE-004).

## Implementation

- [ ] Hoist the `eps` clamp and the `max_eps > 0.3` warning above the
      scalar/tensor branch so both share them (CORE-001).
- [ ] Choose and document one aliasing contract for `LowPassFilter.__call__`;
      make both branches obey it (CORE-002).
- [ ] Forward `filter_tau_const` to the supplied filter instance, or reject the
      ambiguous argument combination explicitly (CORE-003).
- [ ] Give the HDR path an explicit non-zero dark floor derived from the
      intended minimum photoreceptor bandwidth (CORE-005).
- [ ] Resolve the `# DEVNOTE why +20?` in `rescale_intensity_frame` so both
      intensity paths cite the same justification (CORE-005).

## Decisions required

- [ ] CORE-002: in-place or pure? In-place needs a renamed method; pure costs an
      allocation on the scalar path.
- [ ] CORE-004: does the lin-log rounding survive its own experiment? Record the
      measured answer in the code comment and retire the `TODO (TBC)` either way.
- [ ] CORE-005: what dark floor value, and on what physical grounds?

## Gate

- [ ] Focused RED/GREEN test for each finding above.
- [ ] `conda run -n v2e python -m pytest -q`.
- [ ] Update the `KEY[E-LPF-EPS]` and `KEY[E-INTEN-SCALE]` prose in
      [`../core_model_mapping.md`](../core_model_mapping.md) to match the fixed
      behaviour; that document currently documents the missing clamp as an open
      defect.
- [ ] Mark the resolved findings `Resolved` in
      [`../findings/core_model.md`](../findings/core_model.md).
