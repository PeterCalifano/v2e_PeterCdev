# Plan: IEBCS Pixel-State Lifecycle

Scope: the opt-in IEBCS mechanisms — refractory-state coupling, threshold reset,
histogram background noise, and their interaction with comparator memory. Owns
[IEBCS-001](../findings/iebcs_extensions.md#iebcs-001) through
[IEBCS-007](../findings/iebcs_extensions.md#iebcs-007).

Not in scope: where a delayed event lands in the output stream — that is
[`plan_event_ordering.md`](plan_event_ordering.md).

Status: open. Prerequisite: [`plan_state_lifecycle.md`](plan_state_lifecycle.md)
(the time-origin decision in LIFE-002 determines what IEBCS-001 anchors to).

## Design

- [ ] Move release-state evolution **before** event eligibility and
      quantisation, so a renewed threshold check is possible at all
      (IEBCS-005).
- [ ] Decide whether release evolution is linear (current) or exponential
      (reference), and record the choice with its justification.
- [ ] Decide whether threshold reset becomes per-event iterative behaviour or
      stays a documented frame-level approximation.
- [ ] Define one cap contract covering scheduling, event-state transition, and
      polarity fairness, rather than three independently patched behaviours
      (IEBCS-006).

## Tests first

- [ ] Zero-duration coupled refractory reproduces the uncoupled output
      **bitwise** for a fixed seed (IEBCS-004).
- [ ] Release-state change followed by a renewed crossing emits the extra event
      (IEBCS-005).
- [ ] Comparator memory after a coupled frame equals the model's defined value,
      with no double reset (IEBCS-005).
- [ ] Stream starting at a large non-zero time emits noise only within the frame
      interval that produced it (IEBCS-001).
- [ ] Non-cumulative histogram input is rejected with an actionable message
      (IEBCS-002).
- [ ] A saturated noise frame preserves ON/OFF balance and shows no row-major
      spatial bias (IEBCS-006).
- [ ] Histogram noise events respect refractory availability and update release
      timestamps.
- [ ] Interaction: histogram noise + refractory coupling + threshold reset
      enabled together.

## Implementation

- [ ] Anchor the initial noise schedule to the first frame timestamp; pass that
      timestamp into `_init()` (IEBCS-001).
- [ ] Validate CDF rows as non-decreasing and reject otherwise, instead of
      coercing with `maximum.accumulate` (IEBCS-002).
- [ ] Match the reference normalisation divisor once the file layout is known
      (IEBCS-003) — blocked on IEBCS-007.
- [ ] Restructure `generate_events` to state-first ordering: evolve, then
      quantise, then reset once (IEBCS-004, IEBCS-005).
- [ ] Merge ON/OFF due candidates by timestamp before applying a single cap;
      select randomly rather than row-major when truncating; expose capped
      counts as a diagnostic counter (IEBCS-006).
- [ ] Apply refractory availability to noise events and update release
      timestamps and comparator memory at emitting pixels.
- [ ] Update `pos_thres_nominal` / `neg_thres_nominal` atomically with the
      active thresholds and dependent probability scales.

## Preset assets

- [ ] Resolve IEBCS-007: bundle the six `input/iebcs_noise/*.npy` files, add a
      documented generation script, or remove `preset` from
      `--iebcs_noise_source` until assets exist. Removal is the recommended
      immediate step.
- [ ] Once resolved, delete the "presets unavailable" caveat from every document
      that currently repeats it.

## Reference validation

- [ ] Build small derived fixtures from `../IEBCS` without introducing a runtime
      dependency on the sibling checkout.
- [ ] Compare event counts, ON/OFF balance, per-pixel inter-event intervals,
      timestamp distributions, and selected internal-state trajectories.
- [ ] Document intentional differences, tolerances, and unsupported reference
      behaviour (arbiter/readout modelling).
- [ ] Use *output-equivalent* only for mechanisms that pass these gates; until
      then keep the vocabulary defined in
      [`../README_error_models_extensions.md`](../README_error_models_extensions.md).

## Gate

- [ ] All tests above, plus every pairwise interaction test.
- [ ] `conda run -n v2e python -m pytest -q`.
- [ ] Comparative plots regenerated without post-hoc stream sanitisation.
- [ ] Mark resolved findings in
      [`../findings/iebcs_extensions.md`](../findings/iebcs_extensions.md), and
      update the classification rows in
      [`../repo_capabilities_status.md`](../repo_capabilities_status.md).
