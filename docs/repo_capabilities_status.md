# v2e Capabilities and Current Status

Status date: 2026-08-09
Baseline reviewed before CORE-006: `bc8f17d`
Branch: `feature/extend_error_models_IEBCS_V2CE`

This is the single current capability/status summary: what the repository can do
today, and how far that has been validated. It deliberately contains no defect
explanations and no action items.

- Defects, with evidence and suggested fixes: [`findings/`](findings/)
- Ordered work: [`developments/`](developments/)
- Full document ownership map: [`README.md`](README.md)

## Validation Snapshot

- Environment: `conda run -n v2e python --version` -> `Python 3.11.11`
- Full suite: `conda run -n v2e python -m pytest -q`
- Isolated staged-tree result: `95 passed`
- Additional development changes remain modified and untracked
- Readiness: **not merge-ready**

The green suite validates existing local contracts. It does not cover several
confirmed cross-packet, cross-feature, SuperSloMo HDR, or reference-parity
defects listed below.

## Input -> Models -> Output

```text
INPUT
  Direct API:
    NumPy grayscale frames + explicit timestamps
  CLI:
    video files | ordered image folders | synthetic generators
    nominal uint8 input | HDR/preprocessed-log modes

  -> TIMING / PREPROCESSING
     source-rate `.npy` processing (--disable_slomo)
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
| HDR no-SloMo CLI path | Float TIFF -> `.npy` -> emulator/HDF5 without PNG narrowing | Tested against direct API output |
| HDR through SuperSloMo | Interpolator output still uses PNG intermediates | Float HDR preservation unsupported and unvalidated |
| Core default model | v2e lin-log, bandwidth, thresholds, reset, leak, noise, refractory | Implemented; HDR-dark edge defect remains |
| Strict model validity | `--strict_model_validity` and direct API policy | Low-pass `eps > 1` fails before clamping; additional invariants remain planned |
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

## Blocking Defects

These are the open defects that currently prevent a merge-ready claim. Each is
described once in [`findings/`](findings/); this table only records that the
blocker exists and how severe it is.

| Finding | Severity | Effect on capability |
|---|---|---|
| [LIFE-001](findings/state_lifecycle.md#life-001) | Critical | `reset()` makes an emulator unusable; blocks sequence reuse |
| [LIFE-003](findings/state_lifecycle.md#life-003) | Critical | `set_dvs_params()` mid-stream crashes |
| [STREAM-001](findings/stream_and_io.md#stream-001) | High | Latency produces globally non-monotonic streams in every writer |
| [IEBCS-001](findings/iebcs_extensions.md#iebcs-001) | High | Histogram noise emits events before the stream start |
| [IEBCS-002](findings/iebcs_extensions.md#iebcs-002) | High | Non-cumulative noise input is silently misread |
| [IEBCS-004](findings/iebcs_extensions.md#iebcs-004) | High | Zero-duration refractory coupling is not inert |
| [IEBCS-005](findings/iebcs_extensions.md#iebcs-005) | High | Refractory interpolation corrupts comparator memory |
| [V2CE-002](findings/v2ce_timing.md#v2ce-002) | High | Configured refractory period is not enforced under V2CE timing |
| [LIFE-002](findings/state_lifecycle.md#life-002) | High | `reset()` leaves a stale time origin |
| [LIFE-004](findings/state_lifecycle.md#life-004) | High | Preset switching discards per-pixel threshold mismatch |

Medium and low severity findings, and the full evidence for each entry above,
are in [`findings/README.md`](findings/README.md).

The ordered work to resolve them is in
[`developments/consolidation_staged_plan.md`](developments/consolidation_staged_plan.md).

## Open-Tree Changes Awaiting Commit

The working tree carries reviewed improvements that are not yet committed:
removal of the merge-duplicated shot-noise, memory and writer side effects from
`f026560`; enum-backed finite options with unchanged public CLI strings; private
per-emulator random generators; buffered and chunked HDF5 writes with separate
logical and physical counters; idempotent cleanup; and regressions covering
those paths.

`v2ecore/model_options.py` is untracked but imported by tracked modules, so
these changes cannot be committed piecemeal. Sequencing and commit boundaries
are owned by
[`developments/consolidation_staged_plan.md`](developments/consolidation_staged_plan.md).

## Recent Commit Reassessment

- `11f77de`: introduced the first latency, threshold-reset, and V2CE timing
  extensions in the emulator.
- `01c0155`: added the main later IEBCS contrast-latency, histogram-noise, and
  refractory-coupling paths; current lifecycle defects originate here.
- `fb40595`: added optimization work; its low-pass regressions were resolved by
  `bc8f17d`.
- `f026560`: merged `dev_main`; conflict resolution duplicated stateful side
  effects and writer calls. The open tree removes them.
- `6b66007`, `a7a4071`, `379db53`: reorganized and implemented comparative
  benchmarks; raw global ordering is currently hidden by post-sort.
- `50cbe8b`, `969dcc5`: expanded status/docs, but linked untracked files and
  overstated IEBCS/V2CE behavior. This documentation pass corrects those claims.
- `66cdfb4`: fixed the comparative benchmark entrypoint and empty-data plots.
- `bc8f17d`: consolidated low-pass stability, state aliasing, explicit tau, and
  opt-in strict validity.

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

## Related Documents

The full ownership map is in [`README.md`](README.md). The documents this one
defers to:

- Defect register: [`findings/README.md`](findings/README.md)
- Extension semantics:
  [`README_error_models_extensions.md`](README_error_models_extensions.md)
- Core model map: [`core_model_mapping.md`](core_model_mapping.md)
- Plans: [`developments/`](developments/)
