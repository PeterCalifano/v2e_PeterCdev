# v2e Capabilities and Current Status

Status date: 2026-07-28
Committed baseline: `969dcc5`
Branch: `feature/extend_error_models_IEBCS_V2CE`

This is the single current capability/status summary. Detailed findings and
evidence are in
[`implementation_review_report.md`](implementation_review_report.md); active
work is staged under [`developments/`](developments/).

## Validation Snapshot

- Environment: `conda run -n v2e python --version` -> `Python 3.11.11`
- Full suite: `conda run -n v2e python -m pytest -q`
- Fresh result after this documentation/comment pass:
  `103 passed in 20.00s`
- Working tree: modified and untracked; no review changes are committed
- Readiness: **not merge-ready**

The green suite validates existing local contracts. It does not cover several
confirmed cross-packet, cross-feature, HDR, or reference-parity defects listed
below.

## Input -> Models -> Output

```text
INPUT
  Direct API:
    NumPy grayscale frames + explicit timestamps
  CLI:
    video files | ordered image folders | synthetic generators
    nominal uint8 input | HDR/preprocessed-log modes

  -> TIMING / PREPROCESSING
     source-rate processing (--disable_slomo)
     or optional SuperSloMo interpolation
     crop / resize / grayscale conversion
     lin-log encoding or preprocessed-log bypass

  -> CORE v2e SENSOR MODEL
     finite intensity-dependent photoreceptor bandwidth
     per-pixel ON/OFF threshold mismatch
     contrast quantization and comparator-memory reset
     leak
     simple shot noise or filtered photoreceptor noise
     practical refractory filtering
     optional CSDVS / SCIDVS variants

  -> OPT-IN EXPERIMENTAL EXTENSIONS
     IEBCS-inspired latency, threshold reset, histogram noise,
       contrast latency, and refractory-state coupling
     V2CE-inspired frame-global random/sqrt(U) timestamp layers

  -> OUTPUT
     Direct API: NumPy [t, x, y, p] packets
     Files: HDF5 | AEDAT-2.0 | AEDAT-4.0 | whitespace text
     Visuals: DVS AVI/preview | source/slomo AVI | state videos
     Diagnostics: frame timestamps | optional signal/noise labels |
       per-pixel state recording
```

## Capability Classification

| Area | Capability | Status |
|---|---|---|
| Direct frame API | Stateful `EventEmulator.generate_events(frame, t)` | Implemented and broadly regression-tested |
| Standard CLI conversion | Video/image/synthetic input through v2e pipeline | Implemented |
| HDR direct API and HDF5 frame storage | Float frames preserved at those boundaries | Tested |
| HDR complete CLI path | No-SloMo path writes/reads PNG intermediates | **Incorrect: precision can be quantized away** |
| Core default model | v2e lin-log, bandwidth, thresholds, reset, leak, noise, refractory | Implemented; scalar low-pass and HDR-dark edge defects remain |
| CSDVS / SCIDVS | Optional sensor variants | Implemented; not revalidated against references in this audit |
| IEBCS options | Same broad effect classes exposed as opt-in flags | Experimental; not IEBCS-output-equivalent |
| V2CE options | Irregular placement of existing global event layers | Experimental; not V2CE local timing inference |
| Histogram presets | Named `3klux`, `161lux`, `0.1lux` interface | **Unavailable without six external `.npy` files** |
| Writers | HDF5, AEDAT-2, AEDAT-4, text, AVI/diagnostics | Implemented; long-time precision and reset boundaries remain open |
| Comparative visualization | JSON/CSV/NPZ metrics and plots | Implemented; benchmark currently sorts away a raw ordering defect |
| Shared C++/CUDA backend | Development prototypes outside this repo | **Not a repository capability** |

## Paper Alignment

### Core v2e

The repository's strongest and best-supported model remains the original v2e
pipeline from Hu, Liu, and Delbruck,
[*v2e: From Video Frames to Realistic DVS Events*](https://arxiv.org/abs/2006.07722).
The math-to-code map is
[`core_model_mapping.md`](core_model_mapping.md).

### IEBCS

The optional flags are inspired by Joubert et al.,
[*Event Camera Simulator Improvements via Characterized
Parameters*](https://doi.org/10.3389/fnins.2021.702765).

Current classification:

- Simple and contrast-dependent latency: same effect class/logarithmic shape,
  but different ordering and pixel-state lifecycle.
- Threshold reset: once after a frame packet, not iterative after every event.
- Histogram noise: related CDF scheduling, but incorrect nonzero time origin and
  missing refractory/comparator-state coupling.
- Refractory coupling: linear release interpolation over already quantized
  counts, not the reference exponential evolution plus renewed checks.
- Missing reference behavior includes arbiter/readout modeling and the complete
  iterative pixel lifecycle.

These are component-level approximations, not validated IEBCS output
equivalents.

### V2CE

The current official paper revision is Zhang et al.,
[*V2CE: Video to Continuous Events Simulator*](https://arxiv.org/abs/2309.08891),
arXiv v2 / ICRA 2024.

v2e's `random` and `slope` options sample one sorted vector per frame transition
and share each timestamp layer across all active pixels. V2CE instead predicts
event voxels and infers local, voxel/pixel-conditioned timestamps. The current
options preserve the number of global layers and only relocate them; they are
not a V2CE port or demonstrated de-layering equivalent.

## Confirmed Open Blockers

- [ ] Guarantee timestamp monotonicity across successive packets and every
      writer when latency is enabled.
- [ ] Rework refractory-state coupling so state evolution precedes renewed event
      eligibility checks; zero-duration coupling must be inert.
- [ ] Anchor histogram schedules to the first frame, merge ON/OFF candidates
      before capping, and integrate noise with refractory/reset state.
- [ ] Apply refractory filtering from actual candidate timestamps when V2CE
      non-uniform timing is enabled.
- [ ] Preserve HDR precision through the complete CLI no-SloMo path.
- [ ] Clamp the scalar low-pass update and retain a finite HDR dark-response
      floor.
- [ ] Define or reject `reset()` while HDF5 recording is active.
- [ ] Resolve timestamp precision/wrap contracts for long recordings.
- [ ] Include untracked `v2ecore/model_options.py` atomically with all modules
      that import it.
- [ ] Add IEBCS/V2CE derived reference fixtures and distribution-level parity
      gates before using “output-equivalent.”

The staged order and tests are in
[`consolidation_staged_plan.md`](developments/consolidation_staged_plan.md).

## Current Open-Tree Improvements

- [x] Remove duplicate shot-noise generation, memory updates, and writer calls
      introduced by merge commit `f026560`.
- [x] Add enum-backed finite option values without changing public CLI strings.
- [x] Give each emulator private CPU/device random generators and avoid ambient
      Python/NumPy/PyTorch RNG mutation.
- [x] Buffer and chunk HDF5 event writes while tracking logical and physical
      counts separately.
- [x] Make cleanup idempotent and preserve float HDF5 frame storage.
- [x] Preserve float TIFF values in `ImageFolderReader`.
- [x] Add regression coverage for RNG isolation, writer de-duplication, HDF5
      bookkeeping, enum parsing, and direct HDR boundaries.
- [ ] Commit or split these changes only after the unresolved behavior fixes are
      reviewed and the required untracked module is included.

## Recent Commit Reassessment

- `11f77de`: introduced the first latency, threshold-reset, and V2CE timing
  extensions in the emulator.
- `01c0155`: added the main later IEBCS contrast-latency, histogram-noise, and
  refractory-coupling paths; current lifecycle defects originate here.
- `fb40595`: added optimization work; the scalar low-pass clamp regression
  remains.
- `f026560`: merged `dev_main`; conflict resolution duplicated stateful side
  effects and writer calls. The open tree removes them.
- `6b66007`, `a7a4071`, `379db53`: reorganized and implemented comparative
  benchmarks; raw global ordering is currently hidden by post-sort.
- `50cbe8b`, `969dcc5`: expanded status/docs, but linked untracked files and
  overstated IEBCS/V2CE behavior. This documentation pass corrects those claims.

## Workspace Integrations

### Direct and transitive consumers

- `EventDataGenerationLib` directly imports and drives
  `v2ecore.emulator.EventEmulator` through its streaming conversion/context
  modules and an older frames-to-events pipeline.
- `event-based-centroiding` consumes the EventDataGenerationLib streaming/pool
  API and has direct owner-level v2e tests. Its production path is:

```text
renderer/campaign
  -> EventDataGenerationLib persistent pool and v2e adapter
  -> v2e EventEmulator
  -> canonical EventStream/HDF5
  -> centroiding dataset and models
```

- `SuperEvent` has a direct v2e simulated-inference test.

Important boundary:

- v2e owns sensor emulation and its native writers.
- EventDataGenerationLib owns generic streaming, persistent pooling, adaptation,
  and canonicalization.
- event-based-centroiding owns scene/campaign generation and downstream dataset
  consumption.

The EventDataGenerationLib and event-based-centroiding integrations are present
in dirty/untracked workspace trees, and the parent `V2EConfig` currently exposes
only baseline controls. The IEBCS/V2CE options are not yet selectable through
that adapter and v2e is not declared as a packaged dependency there.

### Reference or comparison repositories

- `IEBCS` and `V2CE-Toolbox`: clean local reference implementations used for
  this audit, not runtime dependencies.
- `senpi_ebi`, `rpg_vid2e`, `MC-EBCS`, and `ETAP`: no direct v2e call path found.
  ETAP uses `rpg_vid2e`, not this repository.
- `trajectory-to-events`: separate generation pipeline/comparison point.

## Active Documents

- Current detailed review:
  [`implementation_review_report.md`](implementation_review_report.md)
- Extension semantics:
  [`README_error_models_extensions.md`](README_error_models_extensions.md)
- Core model map:
  [`core_model_mapping.md`](core_model_mapping.md)
- Comparative benchmark:
  [`../README_comparative_benchmark.md`](../README_comparative_benchmark.md)
- Consolidation plan:
  [`developments/consolidation_staged_plan.md`](developments/consolidation_staged_plan.md)
- V2CE timing plan:
  [`developments/v2ce_timing_staged_plan.md`](developments/v2ce_timing_staged_plan.md)
- Performance plan:
  [`developments/performance_optimization_opportunities.md`](developments/performance_optimization_opportunities.md)
