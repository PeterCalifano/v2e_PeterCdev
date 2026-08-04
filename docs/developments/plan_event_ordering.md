# Plan: Event Ordering and Writer Contracts

Scope: the contract between event generation and its consumers — global
timestamp monotonicity, the delayed-event release policy, writer timestamp
ranges, and the invariant checks that guard them. Owns
[STREAM-001](../findings/stream_and_io.md#stream-001) through
[STREAM-003](../findings/stream_and_io.md#stream-003).

Not in scope: why any individual event is delayed. Latency *models* belong to
[`plan_iebcs_lifecycle.md`](plan_iebcs_lifecycle.md); this plan owns only what
happens to an event's position in the stream once its timestamp exists.

Status: open. Prerequisite: none technically, but this changes the
`generate_events()` return contract, so it should land before
[`v2ce_timing_staged_plan.md`](v2ce_timing_staged_plan.md) Stage 2 and before
any accelerated backend work.

## Design

- [ ] Define a cross-packet delayed-event owner and its watermark rule: an event
      may be released once no future packet can produce an earlier timestamp.
- [ ] Prefer one emulator-owned pending queue over per-writer reorder buffers,
      so every output path inherits the same ordering.
- [ ] Define an explicit finalisation operation that flushes the delayed tail
      through the same conversion and writer path, with no hidden event loss.
- [ ] Keep signal/noise labels in the same queue entry as their event so the
      permutation cannot drift.
- [ ] Decide what `generate_events()` returns for a frame whose events are still
      pending, and specify the change for direct API consumers including
      `EventDataGenerationLib`.

## Tests first

- [ ] Deterministic simple-latency packets whose delayed ranges overlap the next
      packet (STREAM-001).
- [ ] The same for contrast-latency (STREAM-001).
- [ ] Concatenated direct-API output is globally monotonic.
- [ ] HDF5, AEDAT-2, AEDAT-4 and text round-trips preserve that same order.
- [ ] Labels, counts, coordinates and polarities survive reordering.
- [ ] Finalisation flushes every pending event exactly once — asserted by count,
      not by inspection.
- [ ] Timestamps immediately below and above each writer limit: `2147.48 s`
      (AEDAT-2 int32) and `4294.97 s` (HDF5 uint32) (STREAM-002).
- [ ] A cross-packet backward step is detected and raised under strict mode
      (STREAM-003).

## Implementation

- [ ] Queue delayed events at emulator scope; emit only what the current
      watermark makes safe (STREAM-001).
- [ ] Add the finalisation call and wire it into `cleanup()` and the `v2e.py`
      pipeline end (STREAM-001).
- [ ] Keep timestamps in float64 internally; narrow only at the serialisation
      boundary where the target dtype is known (STREAM-002).
- [ ] Raise a named, format-specific error when a timestamp exceeds a writer's
      range instead of allowing the wrap (STREAM-002).
- [ ] Track the last emitted timestamp on the emulator so the monotonicity check
      spans packets; escalate from warning to error under strict mode
      (STREAM-003).
- [ ] Remove benchmark-side sorting as a substitute for producer correctness —
      coordinated with [TOOL-002](../findings/tooling_and_benchmarks.md#tool-002)
      in [`plan_tooling_hygiene.md`](plan_tooling_hygiene.md).

## Decisions required

- [ ] Pending queue, or latency clamped to the packet interval? The queue
      preserves the model; the clamp preserves the current API. If the clamp is
      chosen, `--iebcs_latency_mean_us` must reject values exceeding the frame
      interval rather than silently truncating the distribution.
- [ ] Is a documented maximum recording duration acceptable, or is a per-file
      time origin offset required? The latter is a format change and would be
      planned separately.

## Gate

- [ ] All tests above pass, including every writer round-trip.
- [ ] Combined latency + V2CE + refractory interaction test passes.
- [ ] `conda run -n v2e python -m pytest -q`.
- [ ] Comparative benchmark reports raw stream validity before any sorted
      comparison, and shows zero monotonicity violations.
- [ ] The stream contract table in
      [`../README_error_models_extensions.md`](../README_error_models_extensions.md)
      moves those rows from broken to honoured.
- [ ] Mark resolved findings in
      [`../findings/stream_and_io.md`](../findings/stream_and_io.md).
