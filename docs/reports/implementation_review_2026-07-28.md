# Archived v2e Implementation Review and Re-Evaluation

> Frozen snapshot from 2026-07-28, retained for provenance. Its findings were
> subsequently re-verified, extended, and moved into the living register at
> [`../findings/`](../findings/); its action items moved into
> [`../developments/`](../developments/). Test counts, file paths, and the
> `Documentation Consolidation` section below describe the tree as it was on
> that date and are intentionally not updated.
>
> For current defects read [`../findings/README.md`](../findings/README.md).
> For current status read
> [`../repo_capabilities_status.md`](../repo_capabilities_status.md).

Status date: 2026-07-28
Committed baseline: `969dcc5`
Reviewed state: committed history plus the complete open working tree

## Executive Verdict

The open tree contains valuable consolidation work and passes the current suite
(`103 passed in 20.00s`), but it is **not merge-ready**. The original v2e model
remains
the repository's strongest capability. The IEBCS and V2CE options are
experimental mechanisms whose output equivalence is neither implemented in
full nor validated against the local references.

This audit corrects three earlier conclusions:

- IEBCS options cannot be described collectively as behaviorally or
  output-equivalent. Some formulas and sampling components are aligned, but
  state evolution and ordering differ materially.
- V2CE modes do not currently de-layer events. They move the same number of
  frame-global timestamp layers to irregular positions.
- `103 passed` demonstrates regression consistency, not reference-model
  correctness; confirmed defects lie outside current assertions.

## Review Basis

- Current v2e implementation, tests, open diff, and recent commits.
- Local clean references:
  `../IEBCS` and `../V2CE-Toolbox`.
- Current papers:
  [v2e](https://arxiv.org/abs/2006.07722),
  [IEBCS](https://doi.org/10.3389/fnins.2021.702765), and
  [V2CE arXiv v2 / ICRA 2024](https://arxiv.org/abs/2309.08891).
- Current workspace consumers:
  `../EventDataGenerationLib`, `../event-based-centroiding`, and
  `../SuperEvent`.
- Every active development plan found on disk.

## Findings

### Critical history: merge duplicated stateful side effects

Commit `f026560` duplicated shot-noise generation, ON-memory updates, and
AEDAT/text writer calls during conflict resolution. That can advance stochastic
state twice and serialize duplicate packets.

Current state:

- [x] The open `v2ecore/emulator.py` removes the duplicate operations.
- [x] Regression tests cover duplicate output and memory effects.
- [ ] These fixes remain uncommitted and must not be separated from their tests.

### High: latency breaks global stream ordering

`EventEmulator._apply_latency_jitter_and_sort()` sorts one frame packet. Each
packet is then returned and written immediately. A delayed event from packet
\(k\) can therefore be later than an event emitted from packet \(k+1\).

Fresh multi-frame probes produced backward timestamp steps for both simple
latency/jitter and contrast-latency modes.

Impact:

- HDF5, AEDAT, text, and direct API concatenation can be globally non-monotonic.
- HDF5 frame/event-index attribution no longer corresponds cleanly to timestamp
  order.
- The comparative benchmark hides the defect by sorting concatenated packets
  before metrics.

Required design decision:

- Hold delayed events in a cross-packet priority/release queue, or
- explicitly bound/clamp latency to a packet interval and document the changed
  model.

The first choice is closer to a physical delayed-output contract.

### High: refractory-state coupling changes state after quantization

`compute_event_map()` quantizes candidates once. The optional release
interpolation later modifies `base_log_frame`, but event counts are not
recomputed.

Consequences:

- The claimed “release interpolation followed by renewed threshold checks” is
  not implemented.
- Zero-duration coupled refractory is not inert; deterministic probes changed
  event counts substantially.
- The lifecycle differs from IEBCS, which evolves release state and checks
  crossings again.

This needs a state-first event loop, not a documentation workaround.

### High: histogram noise has time, state, and cap defects

The CDF-row and inverse-frequency sampling are recognizably IEBCS-inspired, but
the complete behavior is not aligned.

Confirmed issues:

- Initial schedules are `delay * phase` relative to zero rather than the first
  frame. A stream beginning at `100 s` emitted events timestamped far before
  its first interval.
- Histogram events bypass refractory availability and do not update release
  timestamps.
- Histogram events do not reset/update comparator memory as IEBCS noise events
  do.
- The safety cap processes ON candidates first, so a saturated frame can emit
  only ON events.
- Partial selection keeps row-major coordinates, adding spatial bias.
- The six named preset files are absent.

The scheduling, event-state transition, and cap policy should be redesigned as
one contract rather than patched independently.

### High: top-level CLI destroys HDR precision

The direct emulator and HDF5 frame dataset can preserve float input. The
no-SloMo top-level pipeline cannot: source arrays are saved as `.npy`, converted
to PNG, then loaded with `cv2.IMREAD_GRAYSCALE`.

A float32 frame spanning `0..1/512` became all zero in a direct pipeline probe.
Existing HDR tests bypass this path.

The disabled-SloMo path should keep `.npy`/float frames end to end and receive a
`v2e.main()` regression.

### High: V2CE timing remains global and can bypass refractory

The current sampler produces one sorted vector for the whole frame transition:

- `random`: `sort(U)`
- `slope`: `sort(sqrt(U))`

Every active pixel in layer \(i\) receives the same timestamp. The layer count
is unchanged. This is unlike V2CE's per-voxel/pixel/polarity sampling and local
slope inference.

Additionally, legacy refractory filtering is skipped when nominal linear
`ts_step` is not smaller than the refractory period. Random actual gaps can be
much smaller. A probe configured for `3.5 ms` refractory produced same-pixel
gaps around `72.5 us`.

Stage 2 must use per-pixel actual candidate timestamps and actual-time
refractory checks.

### High integration blocker: required enum module is untracked

Modified tracked modules import `v2ecore/model_options.py`, but that file is
untracked. Integrating only tracked changes would make imports fail.

- [ ] Include the enum module atomically with all importing modules and tests.
- [ ] Preserve public V2CE flag strings; enum conversion remains internal.

### Medium: scalar low-pass update can overshoot

The intensity-dependent low-pass path clamps \(\epsilon\le1\). The scalar path
passes `delta_time / tau` directly to `lerp_`.

With `cutoff_hz=1`, `dt=1`, old value `0`, and target `1`, the current scalar
path returned approximately `6.283`, not `1`.

The existing large-epsilon regression covers only the tensor path.

### Medium: HDR black has zero bandwidth

Non-HDR intensity scaling applies a dark offset through
`rescale_intensity_frame()`. HDR preprocessing uses `clamp(frame, 0, 1)`
directly. At exact black, \(\epsilon=0\), so low-pass state can freeze after a
bright-to-black transition.

The HDR path needs a physically consistent finite dark floor and a transition
test.

### Medium: threshold reset is frame-level and preset nominals drift

The threshold-reset option resamples once per emitting pixel after the complete
packet. IEBCS resamples per event and can generate another crossing in the same
update.

Separately, `set_dvs_params()` changes active thresholds but not
`pos_thres_nominal`/`neg_thres_nominal`, even though those values drive later
resampling and shot-noise probability scaling.

### Medium: `reset()` and active HDF5 storage have incompatible state

`reset()` rewinds `frame_counter` but not HDF5 logical/written counters or
existing datasets. A probe that reset an active writer produced a non-monotonic
`frame_idx` sequence while old events remained appended.

Choose one explicit contract:

- reject `reset()` after storage starts, or
- atomically recreate/reset every writer and dataset state.

Rejecting it is the simpler and safer interface.

### Medium: timestamp precision and duration limits are implicit

- Frame timestamps are converted to float32 before microsecond integer
  conversion.
- Event packets use float32 seconds, reducing sub-millisecond resolution at
  large absolute times.
- HDF5 stores uint32 microseconds and wraps after about `4294.97 s`
  (`71.6 min`).
- AEDAT-2 paths using signed int32 microseconds have a roughly `2147.48 s`
  (`35.8 min`) limit.

The repository needs an explicit recording-duration contract and tests near
each boundary.

### Medium: benchmark provenance and metrics are misleading

- `benchmark_error_models_eventstream.py` resolves `REPO_ROOT` with
  `parents[1]`, which points to `v2ecore`, not the repository root.
- `Run_profile()` sorts concatenated output before computing stream metrics, so
  monotonicity after that step is tautological.
- Fifteen timing functions are collected by pytest but only print values; they
  are not stable performance regressions.
- Historical fixed speedup/TBB claims no longer describe the current
  environment. Shot noise is PyTorch, not Numba.

Raw stream validity must be measured before any normalized/sorted comparison.

### Low: estimator cleanup and version drift

- `PhotoreceptorNoiseVoltageEstimator` contains a dead polynomial helper and a
  duplicated fit.
- A warning checks `rate_per_bw > 0.5` but says “larger than 0.1.”
- Threshold calibration uses a Python loop and should be profiled before
  optimization.
- `pyproject.toml` reports `1.5.1`, while `CHANGELOG.md` starts at `v1.6.2`.

## Reference-Equivalence Matrix

| Option | What current v2e does | Reference difference | Classification |
|---|---|---|---|
| Simple IEBCS latency | Post-generation non-negative random offsets | Not part of crossing/refractory lifecycle; no arbiter | Same effect class only |
| Contrast latency | Logarithmic amplitude-dependent mean plus jitter | Different tau, jitter, state, and ordering | Component-aligned |
| Threshold reset | Once per emitting pixel after packet | IEBCS resets after each event and rechecks | Cross-frame approximation |
| Histogram noise | CDF row, inverse frequency, scheduled events | Wrong initial origin; missing voltage/refractory reset | Incomplete approximation |
| Refractory coupling | Linear release interpolation after quantization | IEBCS exponential evolution and renewed checks | Incomplete approximation |
| V2CE `random` | Shared random global layers | Independent voxel/pixel timing absent | Timing heuristic |
| V2CE `slope` | Shared fixed `sqrt(U)` end bias | No local slope estimation/inverse CDF | Timing heuristic |

No row currently supports a full output-equivalence claim.

## Test Audit

Current strengths:

- [x] `103` tests pass in the `v2e` conda environment.
- [x] Default/disabled option paths, enum parsing, private RNG ownership, writer
      de-duplication, HDF5 buffering, and direct HDR storage have useful
      regressions.
- [x] Fixed-seed streams are isolated from ambient Python, NumPy, Torch CPU,
      and Torch CUDA RNG state.

Missing gates:

- [ ] Cross-packet latency ordering.
- [ ] Refractory coupling with zero duration and renewed crossings.
- [ ] Nonzero input time for histogram initialization.
- [ ] Histogram/refractory/reset interaction and unbiased cap behavior.
- [ ] V2CE actual-gap refractory enforcement.
- [ ] Complete CLI HDR preservation.
- [ ] Scalar low-pass large-epsilon clamp and HDR dark transition.
- [ ] Active-writer reset behavior.
- [ ] Reference-derived IEBCS/V2CE fixtures and distribution metrics.
- [ ] Actual activation of the `>1,000,000`-track histogram branch.

## Open-Tree Assessment

Useful changes that should be retained:

- Enum-backed finite options with unchanged public CLI names.
- Removal of merge-duplicated state/writer side effects.
- Private per-emulator random generators.
- Buffered/chunked HDF5 output with logical/written counters.
- Idempotent cleanup.
- Float frame storage and float TIFF reader support.
- Associated regression tests and corrected comments.

Constraints:

- Do not mix the unresolved physics fixes into an enum-only cleanup commit.
- Do not expose temporary CUDA prototypes as supported capability.
- Do not add compatibility wrappers around private helpers; fix the owning
  state/output contract directly.

## Workspace Integration Assessment

`EventDataGenerationLib` directly owns the reusable v2e adapter and persistent
emulation contexts. `event-based-centroiding` is a transitive consumer of that
API and owns rendering/campaign policy. `SuperEvent` directly exercises v2e in
a simulated-inference test.

The current parent `V2EConfig` exposes baseline v2e settings only, and the
package metadata does not declare v2e as a dependency. Therefore:

- [x] Direct workspace integration exists.
- [ ] The integration is not yet a clean packaged contract.
- [ ] Downstream callers cannot select the audited IEBCS/V2CE options through
      the parent adapter.
- [ ] EventDataGenerationLib and event-based-centroiding changes remain
      dirty/untracked and must not be described as released interfaces.

No direct v2e path was found in `senpi_ebi`, `rpg_vid2e`, `MC-EBCS`, or `ETAP`.
IEBCS and V2CE-Toolbox are reference repositories, not dependencies.

## Documentation Consolidation

- [x] Keep this file as the detailed current review.
- [x] Keep `repo_capabilities_status.md` as the concise current status.
- [x] Keep `README_error_models_extensions.md` as the extension-semantics
      source.
- [x] Merge anchor navigation into `core_model_mapping.md`; remove the duplicate
      line-number index.
- [x] Archive dated July reports under `docs/reports/`.
- [x] Move active plans to `docs/developments/`.
- [ ] Re-run the link checker after every move.

## Recommended Direction

Correctness consolidation must precede V2CE Stage 2 and accelerated backend
work. Follow
[`consolidation_staged_plan.md`](../developments/consolidation_staged_plan.md),
then
[`v2ce_timing_staged_plan.md`](../developments/v2ce_timing_staged_plan.md),
then the parity-gated portions of
[`performance_optimization_opportunities.md`](../developments/performance_optimization_opportunities.md).
