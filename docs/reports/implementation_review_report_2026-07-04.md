# Archived v2e Implementation Review Report

> Historical snapshot from 2026-07-04. It is retained for provenance and
> contains claims superseded by the current
> [`../findings/`](../findings/) register.
> In particular, `97 passed`, blanket IEBCS behavioral alignment, and V2CE
> de-layering claims are no longer current.

Status date: 2026-07-04

Validated command:
`conda run -n v2e python -m pytest -q`

Fresh full-suite result after this consolidation pass:
`97 passed`

## Scope Audited

- Current branch:
  `feature/extend_error_models_IEBCS_V2CE`
- Recent committed baseline:
  `969dcc5` plus recent benchmark, optimization, HDR, and IEBCS commits listed
  in [`docs/repo_capabilities_status.md`](../repo_capabilities_status.md).
- Current open implementation changes:
  enum-backed finite options, HDF5 event buffering/chunking, output append
  de-duplication, HDR frame storage, V2CE timestamp cleanup, and related tests.
- Current porting/performance evidence:
  July 4 temporary CUDA/Julia/Python/ETAP benchmark harnesses under
  `/tmp/v2e_bench_copy/temp_bench/`, with artifacts under
  `/tmp/v2e_bench_outputs/`. These harnesses did not edit the repo
  implementation and cover a validated clean/contrast-latency subset, not the
  full stochastic IEBCS/V2CE option space.
- Local comparison repos:
  `../IEBCS`, `../V2CE-Toolbox`, `../EventDataGenerationLib`,
  `../SuperEvent`, `../trajectory-to-events`, and `../rpg_vid2e`.
- Primary papers checked:
  [v2e](https://arxiv.org/abs/2006.07722),
  [IEBCS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.702765/full),
  and [V2CE](https://arxiv.org/abs/2309.08891).

## Capability Summary

Current repo capability is:

```text
video / image folder / synthetic source / HDR frame input
  -> optional SuperSloMo or source-rate timing
  -> v2e pixel model:
       lin-log front end
       finite photoreceptor bandwidth
       threshold mismatch
       leak
       shot or photoreceptor temporal noise
       refractory filtering
       optional CSDVS / SCIDVS variants
       optional IEBCS-inspired latency/noise/refractory mechanisms
       optional V2CE-inspired burst timestamp placement
  -> DVS AVI, original/slomo AVI, HDF5, AEDAT-2.0, AEDAT-4.0, text,
     optional labels, frame/state side outputs
```

This makes the branch a broadened `v2e` emulator, not a replacement for V2CE
or IEBCS. The implementation aim should stay: produce high-quality,
representative DVS-style event streams while preserving the v2e physical
pixel-model backbone.

## Model Correctness Review

### Core v2e Model

Status: sound and still the branch's strongest foundation.

- Lin-log conversion, finite bandwidth, threshold mismatch, event quantization,
  memory reset, leak, and temporal noise are mapped in
  [`docs/core_model_mapping.md`](../core_model_mapping.md).
- Source now has lightweight `# KEY[...]` anchors near the relevant code.
- Regression coverage checks core event generation, determinism, I/O, output
  append behavior, HDF5 state, HDR preservation, and benchmark smoke paths.

Important caveats:

- `Map_linear_to_log_luminance` intentionally uses float64 internally for
  numerical stability. This is correct-first but remains a performance cost.
- HDF5 event timestamps are stored as uint32 microseconds, matching the
  existing v2e-compatible schema but limiting long recordings to about
  4294 seconds before timestamp wrap would become a concern.

### IEBCS-Derived Mechanisms

Status: behaviorally aligned at modeled-effect level, not source-identical.

- Latency/jitter:
  adds non-negative timestamp offsets and sorts output. This gives the same
  output-effect class as IEBCS delayed/jittered event timing.
- Contrast latency:
  uses the same first-order shape as the IEBCS tau-based latency path:
  `mu - tau * log(1 - amp)`, with jitter and clamping.
- Threshold reset:
  resamples thresholds after emitted signal events, matching the intended
  comparator/reset variability effect.
- Histogram background noise:
  schedules per-pixel ON/OFF noise events from CDF rows and resamples after
  emission, matching the IEBCS output mechanism if the same histograms are used.
- Refractory-state coupling:
  interpolates memory state at release and gates eligibility, matching the
  IEBCS intent.

Limitations:

- Preset histogram `.npy` files are not bundled in this repo snapshot. Preset
  mode is therefore a configured interface, not a ready-to-run feature unless
  users provide `input/iebcs_noise/*.npy`.
- Histogram noise has an internal per-frame cap to prevent runaway loops. This
  is a reasonable safety guard, but it intentionally deviates from uncapped
  reference output under extreme noise settings.
- Tests prove mechanism behavior and CLI wiring; they do not yet prove
  distribution-level equivalence against the local `../IEBCS` repo.

### V2CE-Derived Timestamp Placement

Status: useful narrow heuristic, not V2CE-equivalent output.

- The implementation changes only intra-frame burst timestamp placement:
  `random` uses sorted uniform fractions and `slope` uses sorted `sqrt(U)`.
- This reduces visible temporal layering compared with strictly linear
  subdivision and preserves event counts.
- The current mode does not implement V2CE's learned event voxel prediction,
  voxel-conditioned timestamp sampling, or local dynamic-aware timestamp
  inference.

Conclusion:

- It is meaningful as a `v2e` option because it attacks one known artifact of
  frame-to-event conversion: timestamp layering.
- It should not be presented as behaviorally equivalent to full V2CE output.
- Stage 2 should replace the `sqrt(U)` slope proxy with an analytic local-drive
  timing model using `diff_frame`, thresholds, and photoreceptor timing.

### CSDVS / SCIDVS

Status: present and documented as sensor variants, but not the main focus of
this branch.

- They remain part of the capability map.
- No new cross-validation was added in this review pass.
- Future work should avoid entangling these variants with IEBCS/V2CE options
  unless specific interactions are tested.

## Findings

### Medium: V2CE Is Still Only a Partial Output-Effect Match

Evidence:
`v2ecore/emulator.py` samples sorted random or `sqrt(U)` fractions, while local
`../V2CE-Toolbox` uses event voxel counts and slope-conditioned timestamp
inference.

Impact:
The option can improve temporal layering, but it cannot reproduce full V2CE
event statistics or local dynamics.

Plan:
Keep docs explicit. Implement
[`doc/developments/v2ce_timing_staged_plan.md`](../developments/v2ce_timing_staged_plan.md)
Stage 2 before claiming stronger equivalence.

### Medium: IEBCS Histogram Presets Are Exposed But Assets Are Missing

Evidence:
No files exist under `input/iebcs_noise/`; only `input/SuperSloMo39.ckpt` is
present.

Impact:
`--iebcs_hist_noise_model=true --iebcs_noise_source=preset` is not runnable
without local assets.

Plan:
Either bundle validated preset histograms or document an acquisition/generation
workflow. Until then, keep explicit `files` mode as the reliable path.

### Medium: Cross-Repo Equivalence Is Not Yet Quantified

Evidence:
Tests validate local mechanisms, not distributional comparison against IEBCS or
V2CE reference outputs.

Impact:
The docs can claim behaviorally aligned mechanisms, not distribution-identical
or benchmark-proven equivalence.

Plan:
Add small comparison fixtures:
IEBCS for latency/noise distributions, V2CE for normalized burst timestamp
distributions.

### Medium: Shared CUDA Backend Is Benchmarked, Not Productized

Evidence:
Temporary July 4 harnesses validated clean count maps and clean event sets
against v2e for 640x480, 1024x768, and 1920x1080 synthetic cases. They also
validated deterministic contrast-latency output to exact/allclose levels, but
they live outside the repo and skip stochastic RNG equivalence, histogram noise,
refractory coupling, writers, and full timestamp-ordering interactions.

Impact:
The benchmark evidence is strong enough to guide the porting direction, but it
is not yet a supported repo capability. Docs should describe it as a development
result and not as an installed backend.

Plan:
Create a production shared C++/CUDA backend callable from Python and Julia,
with Python fallback and staged parity tests before wiring it into the main CLI.

### Low: HDF5 Timestamp Schema Has Long-Recording Limits

Evidence:
HDF5 output stores `events` as uint32 `[t_us, x, y, p01]`.

Impact:
The schema is compatible with local readers, but uint32 microsecond timestamps
wrap after about 71.6 minutes.

Plan:
Do not change the schema casually. If long recordings matter, add a documented
uint64 or seconds-float variant with reader support in downstream repos.

### Low: Environment Documentation Differs From Repo Instruction

Evidence:
`AGENTS.md` prefers Python >=3.12. The validated `v2e` conda environment is
Python 3.11.11.

Impact:
Tests pass in the actual maintained env, but the repo instruction and runtime
environment are not aligned.

Plan:
Either update the environment to Python >=3.12 and validate, or document that
the current supported env is Python 3.11.11 until migration is complete.

## Performance Review

Current improvements:

- HDF5 event writes are buffered and chunked instead of resizing on every
  packet.
- Duplicated output appends were removed.
- CPU/GPU transfer cleanup and low-pass scalar optimization remain from recent
  committed optimization work.
- Finite option enums do not affect the hot path.

July 4 porting benchmark evidence:

- A fused CUDA count/emit/update subset matched clean v2e count maps and event
  sets exactly for 640x480, 1024x768, and 1920x1080 extended-target cases.
- Deterministic contrast-latency output matched v2e exactly/allclose; the
  Python C++/CUDA extension saw only sub-nanosecond timestamp differences in
  larger cases due near-tied sorting.
- GPU-resident medians for the Julia CUDA prototype were about 0.061 ms,
  0.072 ms, and 0.165 ms for the three resolutions; the Python C++/CUDA
  extension with preallocated CUB scan was about 0.026 ms, 0.044 ms, and
  0.102 ms.
- The observed Python-vs-Julia difference is a backend implementation detail:
  Julia used high-level `CUDA.accumulate` with allocation in the timed loop,
  while the Python extension used a preallocated CUB scan workspace. It is not
  evidence that Python is intrinsically faster than Julia.
- Host event download and disk formats dominate when every event is
  materialized on CPU. For full pipelines, preserving GPU-resident event
  tensors matters as much as kernel speed.
- ETAP-side measurements show representation loading/normalization is a major
  bottleneck: current HDF5 representation load+normalize+upload can exceed a
  model forward pass, while a corrected GPU event-to-stack prototype is around
  5.22 ms for 200k events.

Remaining hotspots:

- Event extraction still loops over `max_num_events_any_pixel`, uses `nonzero`
  per iteration, and shuffles each iteration with `torch.randperm`.
- V2CE non-uniform mode adds a small per-frame random sort over burst depth.
- Histogram noise can loop until due events are exhausted, with a safety cap.
- Renderer frame accumulation and histogram paths still depend on Numba/TBB
  availability and event volume.
- The core emulator remains Python/PyTorch driven, so custom CUDA/C++ kernels
  remain the likely path for large speedups.
- Full shared-backend parity is still open for stochastic RNG paths,
  histogram-noise scheduling, refractory coupling, photoreceptor/shot/leak
  paths, output writers, and final cross-frame ordering policy.

Performance recommendation:

- Treat the July 4 benchmark as the porting direction: build one shared
  C++/CUDA backend callable from both Python and Julia, not separate language
  rewrites.
- Prioritize GPU-resident clean/contrast-latency event generation and ETAP
  event-stack tensors before optimizing disk materialization paths.
- Keep all claims scoped to validated effects until parity tests cover the
  skipped stochastic and writer paths.
- Keep Python fallbacks for portability.

## Design Review

Positive changes:

- Internal enum classes now fit the finite option surfaces without changing CLI
  nomenclature.
- V2CE public flag names were preserved.
- Development plans moved to `doc/developments/`.
- Docs now separate completed work from open work with checkboxes.
- `CONTEXT.md` captures current state before future compaction.

Design risks:

- `EventEmulator` is accumulating many optional feature flags. This is still
  acceptable for this branch, but future model work should keep logic in small
  private methods and avoid compatibility wrappers.
- Some output schemas are shared with sibling repos. HDF5/AEDAT changes should
  be treated as integration changes, not local-only refactors.
- Model names from external papers should remain scoped as “inspired” or
  “behaviorally aligned” unless reference-level validation is added.

## Workspace Integration Review

- `EventDataGenerationLib`:
  supports v2e-compatible HDF5/text/AEDAT-style event formats and is the
  clearest format-level integration.
- `SuperEvent`:
  has tests that generate v2e synthetic events through `EventEmulator`, write
  AEDAT4 through `EventDataGenerationLib`, and run inference.
- `trajectory-to-events`:
  contains a separate video-to-events wrapper using Metavision's GPU simulator
  with v2e-like parameters; it is adjacent, not a direct dependency.
- `IEBCS`:
  local reference for characterized latency/noise/refractory mechanisms and
  histogram assets.
- `V2CE-Toolbox`:
  local reference for learned voxel generation and timestamp inference.
- `rpg_vid2e`:
  comparison pipeline for video-to-event generation.

## Development Plan

- [x] Consolidate current branch status and capabilities.
- [x] Move development plans under `doc/developments/`.
- [x] Mark completed/open work with Markdown checkboxes.
- [x] Preserve V2CE CLI nomenclature.
- [x] Replace appropriate raw finite options with internal enum classes.
- [x] Document recent committed context.
- [x] Validate full test suite in the maintained conda env.
- [x] Capture July 4 temporary CUDA/Julia/Python/ETAP benchmark evidence in
      status docs.
- [ ] Add IEBCS fixture-level distribution tests for latency, threshold reset,
      histogram noise, and refractory release.
- [ ] Implement V2CE Stage 2 analytic burst timing.
- [ ] Add V2CE Stage 3 local toolbox cross-check.
- [ ] Productize a shared C++/CUDA backend callable from Python and Julia.
- [ ] Add parity tests for skipped stochastic/noise/refractory/writer paths
      before exposing a production accelerated backend.
- [ ] Add ETAP-oriented GPU event-stack output path or integration plan.
- [ ] Decide how to handle missing IEBCS histogram preset assets.
- [ ] Resolve Python version support statement.
- [ ] Resolve package version drift between `pyproject.toml` and
      `CHANGELOG.md`.
- [ ] Re-run repo-native comparative benchmarks after final consolidation and
      keep temporary porting benchmark artifacts referenced separately.
- [ ] Review downstream HDF5/AEDAT compatibility before changing event schemas.

## Commit Split Recommendation

The superseding merge sequence is in
[`doc/developments/consolidation_staged_plan.md`](../developments/consolidation_staged_plan.md).

Recommended split:

- [ ] Commit 1:
      enum-backed options and error-model wiring.
- [ ] Commit 2:
      HDF5/output correctness and regression tests.
- [ ] Commit 3:
      documentation/report consolidation.

Do not mix schema/performance changes with documentation-only cleanup unless
the staged diff is intentionally reviewed.
