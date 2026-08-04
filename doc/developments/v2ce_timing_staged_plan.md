# V2CE-Inspired Timing Development Plan

Status: Stage 1 implemented; Stages 2-3 blocked on correctness consolidation

Public nomenclature remains:

- `--v2ce_nonuniform_burst_timestamps`
- `--v2ce_burst_timestamps_mode {random,slope}`

This plan improves timestamp output effects without porting V2CE's learned
voxel-generation network.

## Stage 1: Existing Global-Layer Heuristic

- [x] Keep the public V2CE flags unchanged.
- [x] Centralize global burst timestamp generation.
- [x] Preserve default linear timing when disabled.
- [x] Add enum-backed internal mode selection.
- [x] Add fixed-seed tests for `random` and `slope`.
- [x] Preserve event count, polarity, coordinates, and global layer count.
- [x] Remove duplicated hot-path side effects discovered during cleanup.
- [x] Document that `random` uses `sort(U)` and `slope` uses
      `sort(sqrt(U))`.
- [x] Correct the capability claim: these modes relocate global layers; they do
      not yet eliminate layering.

## Prerequisite Gate

- [ ] Complete Stages 1-3 of
      [`consolidation_staged_plan.md`](consolidation_staged_plan.md).
- [ ] Guarantee global stream ordering and actual-timestamp refractory checks.
- [ ] Freeze a reference fixture format and comparison metrics.

## Stage 2: Per-Pixel Analytic Timestamp Inference

Goal: preserve v2e's event-count physics while producing continuous,
locally conditioned timestamps rather than shared frame-global layers.

### Design

- [ ] Flatten per-pixel candidate counts into explicit
      `(x, y, polarity, ordinal)` candidates.
- [ ] Preserve candidate count, coordinates, and polarity when refractory is
      disabled.
- [ ] Generate and sort timestamps independently per pixel/polarity.
- [ ] Keep `random` mode as independent per-event uniform sampling within the
      frame interval or local threshold-crossing strata.
- [ ] Replace the fixed `sqrt(U)` proxy in `slope` mode with inverse-CDF sampling
      from a nonnegative linear temporal density
      \(p(u)=a u+b\), \(u\in[0,1]\).
- [ ] Derive \(a\) from local v2e photoreceptor/change-detector evolution across
      adjacent frame intervals; use the uniform limit when \(|a|\) is small.
- [ ] Constrain the density analytically rather than adding tuning clamps that
      have no model meaning.
- [ ] Apply refractory filtering to each actual candidate timestamp before
      updating emitted-count memory.
- [ ] Avoid alternate APIs or compatibility wrappers; replace the existing
      private global-layer implementation directly.

### Tests first

- [ ] Candidate timestamps stay in \((t_{k-1},t_k]\).
- [ ] Each pixel/polarity sequence is monotonic.
- [ ] Global assembled output is monotonic after the shared lifecycle owner.
- [ ] Event count, polarity, and coordinates match the baseline candidate set
      when refractory is disabled.
- [ ] Independent pixels no longer share only \(N_{\max}\) timestamps.
- [ ] Fixed seeds reproduce `random` and `slope` output exactly.
- [ ] Positive, zero, and negative local slopes produce the expected timestamp
      distribution ordering.
- [ ] Uniform-density limit matches independent uniform sampling statistically.
- [ ] Actual refractory gaps satisfy the configured period.
- [ ] Default/disabled V2CE output remains bitwise unchanged.

### Efficiency gate

- [ ] Use vectorized flattened candidates; do not add a Python loop per event.
- [ ] Benchmark low, typical, and burst-heavy event counts on CPU and CUDA.
- [ ] Record allocation volume and host/device transfers.
- [ ] Keep the default path free of Stage-2 overhead when the flag is disabled.

## Stage 3: Reference Cross-Check

Goal: demonstrate the intended continuous/local timing effect without claiming
the learned V2CE first stage.

- [ ] Build small derived fixtures from local `../V2CE-Toolbox`.
- [ ] Compare against `random_even_sample.py`, `pure_slope_sample.py`, and
      `LDATI.py` at a common normalized voxel/time convention.
- [ ] Compare timestamp ECDF/Wasserstein distance, layer ratio, per-pixel
      inter-event intervals, and count/polarity preservation.
- [ ] Include constant, ramp, accelerating, decelerating, and mixed local-motion
      stimuli.
- [ ] Require Stage 2 to improve continuous/local timing metrics over both
      legacy linear layers and Stage-1 global heuristics.
- [ ] Store only derived fixtures/summary metadata in v2e; keep the sibling
      toolbox optional.
- [ ] Document that learned voxel counts and full V2CE output remain outside
      scope.

## Completion Criteria

- [ ] The modes produce independent, locally conditioned event times.
- [ ] No global-layer or refractory regression remains.
- [ ] Reference metrics and tolerances are reproducible in the `v2e` conda
      environment.
- [ ] Documentation uses “V2CE-inspired analytic timing,” not “V2CE-equivalent
      simulator.”
