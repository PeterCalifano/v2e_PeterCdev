# Correctness Consolidation: Sequencing and Merge Packaging

Scope: **ordering between plans, and the merge packaging that closes them out.**
This document owns no technical work items of its own except the packaging stage
below — each area is planned in its own file, and the detail lives there.

Read this first to know what to do next. Read the area plan to know how.

Status: active.

## Plan map

| Plan | Owns findings | Prerequisite |
|---|---|---|
| [`plan_core_numerics.md`](plan_core_numerics.md) | `CORE-001..005` | none |
| [`plan_state_lifecycle.md`](plan_state_lifecycle.md) | `LIFE-001..005` | none |
| [`plan_tooling_hygiene.md`](plan_tooling_hygiene.md) | `TOOL-001..007` | none |
| [`plan_event_ordering.md`](plan_event_ordering.md) | `STREAM-001..003` | none technically; changes the public return contract |
| [`plan_iebcs_lifecycle.md`](plan_iebcs_lifecycle.md) | `IEBCS-001..007` | `plan_state_lifecycle` (time-origin decision) |
| [`v2ce_timing_staged_plan.md`](v2ce_timing_staged_plan.md) | `V2CE-001..002` | `plan_event_ordering` |
| [`performance_optimization_opportunities.md`](performance_optimization_opportunities.md) | — | all of the above |

Defects are described once in [`../findings/`](../findings/). Plans cite finding
IDs; they do not restate the analysis.

## Ordering rule

Complete each stage and its gate before starting the next.

```text
Stage A  (parallel)   core numerics | state lifecycle | tooling hygiene
Stage B               event ordering and writer contracts
Stage C               IEBCS pixel-state lifecycle
Stage D               V2CE per-pixel analytic timing
Stage E               merge packaging          <- owned by this document
Stage F               accelerated backend
```

Rationale for the order:

- Stage A items are independent and include the two crash-level defects
  (`LIFE-001`, `LIFE-003`), so they are cheapest to land and unblock testing of
  everything else.
- Stage B changes what `generate_events()` returns, so it must precede any work
  that depends on stream shape.
- Stage C depends on Stage A's time-origin decision and on Stage B's ordering
  guarantee before its noise events can be placed correctly.
- Stage D is blocked on Stage B because per-pixel timestamps are only meaningful
  once cross-packet ordering is owned somewhere.
- Stage F is blocked on all of them: porting an unfixed model to CUDA multiplies
  the cost of every remaining correctness change.

## Stage 0: audit — complete

- [x] Re-review committed history and the complete open working tree.
- [x] Reproduce latency ordering, refractory coupling, histogram time-origin,
      scalar low-pass, HDR CLI, and cross-feature defects.
- [x] Compare extension semantics with local IEBCS and V2CE reference code.
- [x] Re-run the full `v2e` conda suite: `103 passed`.
- [x] Separate findings from plans, and give every defect a stable identifier.
- [x] Split the single consolidation checklist into per-area plans.
- [ ] Review and approve this staged order before changing sensor behaviour.

## Stage E: merge packaging

This is the only technical stage this document owns, because it is the only one
that is inherently cross-cutting.

- [ ] Include `v2ecore/model_options.py` atomically with every module that
      imports it — the tracked modules currently import an untracked file, so no
      partial commit is valid.
- [ ] Keep V2CE public flag names and values unchanged.
- [ ] Expose only validated options through the `EventDataGenerationLib`
      `V2EConfig` adapter.
- [ ] Declare the v2e dependency at the parent package boundary.
- [ ] Validate one complete
      `event-based-centroiding -> EventDataGenerationLib -> v2e -> HDF5`
      sequence, including finalisation and fixed-seed reproducibility.
- [ ] Validate the direct `SuperEvent` smoke path.
- [ ] Refresh all current docs and links from the final code.

### Commit boundaries

- [ ] Commit A: core numerics fixes and their tests.
- [ ] Commit B: state lifecycle fixes and their tests.
- [ ] Commit C: tooling, benchmark and packaging hygiene.
- [ ] Commit D: event ordering, finalisation, and writer tests.
- [ ] Commit E: IEBCS lifecycle corrections and reference fixtures.
- [ ] Commit F: enum module and any remaining open-tree cleanup not already
      included above.
- [ ] Commit G: documentation, plans, and archived reports.
- [ ] Inspect every boundary with `git diff --cached --stat` and
      `git diff --cached`.
- [ ] Make no commit until the staged content is approved.

Constraints that apply to all boundaries:

- Do not separate a behaviour fix from the test that proves it.
- Do not mix unresolved physics fixes into the enum-only cleanup.
- Do not present temporary CUDA prototypes as a supported capability.
- Do not add compatibility wrappers around private helpers; fix the owning state
  or output contract directly.

## Completion gate

- [ ] Full conda suite passes from a clean checkout.
- [ ] No required source file is untracked.
- [ ] No broken local Markdown links.
- [ ] No open Critical or High finding in [`../findings/`](../findings/).
- [ ] Current capability claims in
      [`../repo_capabilities_status.md`](../repo_capabilities_status.md) match
      executable tests and packaged integrations.
