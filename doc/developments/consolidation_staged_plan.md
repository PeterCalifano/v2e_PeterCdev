# v2e Correctness Consolidation Plan

Status: active
Ordering rule: complete each stage and its validation gate before starting the
next stage.

This plan owns correctness and merge consolidation. V2CE model enhancement and
the shared accelerated backend have separate plans and are blocked on the
relevant stages here.

## Stage 0: Audit and Sources of Truth

- [x] Re-review committed history and the complete open working tree.
- [x] Reproduce latency ordering, refractory coupling, histogram time-origin,
      scalar low-pass, HDR CLI, and cross-feature defects.
- [x] Compare extension semantics with local IEBCS and V2CE reference code.
- [x] Re-run the full `v2e` conda suite: `103 passed`.
- [x] Consolidate current status, detailed review, extension semantics, archived
      reports, and active plans.
- [x] Mark every open action with `- [ ]` and completed plan work with `- [x]`.
- [ ] Review and approve this staged order before changing sensor behavior.

## Stage 1: Deterministic Core and I/O Invariants

Goal: fix isolated correctness defects without redesigning IEBCS/V2CE state.

### Tests first

- [ ] Add scalar low-pass regression requiring
      \(0\le\epsilon\le1\) for large \(\Delta t/\tau\).
- [ ] Add bright-to-black HDR regression requiring finite low-pass evolution.
- [ ] Add top-level `v2e.main()` float-input/no-SloMo regression that preserves
      sub-8-bit differences into the emulator and HDF5.
- [ ] Add active-HDF5 `reset()` contract test.
- [ ] Add preset test proving active and nominal thresholds remain coherent.
- [ ] Add benchmark-runner test proving repository-local imports/output root.
- [ ] Add raw, pre-sort benchmark monotonicity metric.
- [ ] Add timestamp conversion tests near float precision and writer range
      boundaries.

### Implementation

- [ ] Clamp scalar low-pass epsilon before `lerp_`.
- [ ] Use the same finite dark-response floor for HDR and non-HDR bandwidth
      scaling, with the selected physical convention documented.
- [ ] Keep float frames as `.npy`/float arrays through the disabled-SloMo path;
      do not round-trip them through PNG.
- [ ] Reject `reset()` after output storage starts unless all writer state is
      explicitly recreated.
- [ ] Update preset nominal thresholds and dependent probability scales
      atomically.
- [ ] Fix benchmark `REPO_ROOT` and preserve raw stream order for validity
      metrics.
- [ ] Keep frame timestamps in float64 until checked integer serialization.
- [ ] Document and enforce duration/range errors instead of allowing silent
      writer wrap.

### Gate

- [ ] Run focused RED/GREEN tests for each invariant.
- [ ] Run `conda run -n v2e python -m pytest -q`.
- [ ] Run `conda run -n v2e python -m compileall -q v2e.py v2ecore test`.
- [ ] Run `git diff --check`.
- [ ] Update current status with exact test count and any intentional schema
      decision.

## Stage 2: Globally Ordered Event Lifecycle

Goal: make direct API packets and every writer represent one chronologically
valid stream.

### Design

- [ ] Define a cross-packet delayed-event owner and watermark rule.
- [ ] Prefer one emulator-owned pending queue over writer-specific reorder
      buffers.
- [ ] Define an explicit finalization operation that releases the delayed tail
      without compatibility wrappers or hidden event loss.
- [ ] Preserve signal/noise labels in the same queue item as each event.
- [ ] Specify behavior for direct API consumers and update
      EventDataGenerationLib sequence finalization accordingly.

### Tests first

- [ ] Add deterministic simple-latency packets whose time ranges overlap.
- [ ] Add deterministic contrast-latency packets whose time ranges overlap.
- [ ] Assert concatenated direct API output is globally monotonic.
- [ ] Assert HDF5, AEDAT-2, AEDAT-4, and text output preserve the same order.
- [ ] Assert labels, counts, coordinates, and polarities survive reordering.
- [ ] Assert finalization flushes every pending event exactly once.

### Implementation

- [ ] Queue delayed events at emulator scope and emit only events safe under the
      current input-time watermark.
- [ ] Flush the tail through the same conversion/writer path at finalization.
- [ ] Remove benchmark-side sorting as a substitute for producer correctness;
      retain optional sorted copies only for explicitly order-invariant metrics.
- [ ] Use actual candidate timestamps for refractory decisions, including
      non-uniform V2CE timestamps.

### Gate

- [ ] Pass direct API and every writer round-trip test.
- [ ] Pass combined latency + V2CE + refractory tests.
- [ ] Run the full conda suite, compileall, and `git diff --check`.

## Stage 3: IEBCS Pixel-State Lifecycle

Goal: make each enabled mechanism internally coherent before claiming reference
parity.

### Refractory and threshold reset

- [ ] Move release-state evolution before event eligibility/quantization.
- [ ] Recompute crossings after a release-state change.
- [ ] Require zero-duration coupling to match the uncoupled output exactly.
- [ ] Decide whether threshold reset becomes per-event iterative behavior or
      remains a documented frame-level approximation.
- [ ] If per-event behavior is selected, add multi-crossing fixtures against
      local IEBCS.

### Histogram noise

- [ ] Anchor the initial schedule to the first frame timestamp.
- [ ] Merge ON/OFF due candidates by timestamp before applying one cap.
- [ ] Remove row-major truncation bias with deterministic seeded selection.
- [ ] Apply refractory availability and update release timestamps.
- [ ] Update comparator memory/state at emitted noise pixels.
- [ ] Define cap semantics and expose capped counts as diagnostics.
- [ ] Decide whether to bundle licensed preset assets or remove `preset` as the
      default until assets are available.

### Reference validation

- [ ] Build small derived fixtures from `../IEBCS` without a runtime sibling
      dependency.
- [ ] Compare event counts, ON/OFF balance, per-pixel inter-event intervals,
      timestamp distributions, and selected internal-state trajectories.
- [ ] Document intentional differences, tolerances, and unsupported arbiter
      behavior.
- [ ] Use “output-equivalent” only for mechanisms that pass these gates.

### Gate

- [ ] Pass focused mechanism tests and all pairwise interaction tests.
- [ ] Run the full conda suite, compileall, and `git diff --check`.
- [ ] Re-run comparative plots without post-hoc stream sanitization.

## Stage 4: Integration and Merge Packaging

- [ ] Include `v2ecore/model_options.py` atomically with every importing module.
- [ ] Keep V2CE public flag names and values unchanged.
- [ ] Expose only validated options through EventDataGenerationLib `V2EConfig`.
- [ ] Declare/document the v2e dependency at the parent package boundary.
- [ ] Validate one complete
      `event-based-centroiding -> EventDataGenerationLib -> v2e -> HDF5`
      sequence, including finalization and reproducibility.
- [ ] Validate the direct SuperEvent smoke path.
- [ ] Refresh all current docs and links from final code.

Recommended commit boundaries:

- [ ] Commit A: deterministic core/I/O fixes and tests.
- [ ] Commit B: ordered event lifecycle and writer tests.
- [ ] Commit C: IEBCS lifecycle corrections and reference fixtures.
- [ ] Commit D: enum/private-RNG/open-tree cleanup not already included above.
- [ ] Commit E: documentation, plans, and archived reports.
- [ ] Inspect every boundary with `git diff --cached --stat` and
      `git diff --cached`.
- [ ] Make no commit until the user approves the staged content.

## Completion Gate

- [ ] Full conda suite passes from a clean checkout.
- [ ] No required source file is untracked.
- [ ] No broken local Markdown links.
- [ ] No open high-severity finding in
      `docs/implementation_review_report.md`.
- [ ] Current capability claims match executable tests and packaged
      integrations.
