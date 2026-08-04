# v2e Error-Model Extensions

This document is the authoritative description of the opt-in IEBCS- and
V2CE-inspired options on
`feature/extend_error_models_IEBCS_V2CE`.

These options are implemented inside the v2e pixel-model pipeline. They are not
source-identical ports, and the current audit does **not** establish
reference-output equivalence.

## References and Scope

- Joubert et al., *Event Camera Simulator Improvements via Characterized
  Parameters*, Frontiers in Neuroscience, 2021:
  <https://doi.org/10.3389/fnins.2021.702765>
- Zhang et al., *V2CE: Video to Continuous Events Simulator*, ICRA 2024,
  arXiv v2: <https://arxiv.org/abs/2309.08891>

The V2CE paper's current arXiv revision is v2 from 26 April 2024. Its pipeline
has two major stages: learned event-voxel prediction and local dynamic-aware
timestamp inference. The options in this repository replace neither stage.

Terminology used below:

- **Implemented**: the code path is callable and locally tested.
- **Component-aligned**: it implements a mechanism from the same effect class.
- **Output-equivalent**: event distributions and state evolution match the
  reference under controlled inputs. No IEBCS/V2CE extension currently meets
  this stronger standard.

## Compatibility Boundary

- [x] Every extension is opt-in and disabled by default.
- [x] The public event schema remains `[t, x, y, p]`.
- [x] Existing V2CE flag names remain unchanged.
- [x] Finite string options are represented internally by enums while retaining
      their public CLI strings.
- [ ] Exact default-path equivalence still needs a committed golden-stream
      fixture across all writers.

## CLI Surface

| Mechanism | Flags | Current audit status |
|---|---|---|
| Simple latency/jitter | `--iebcs_latency_jitter_model`, `--iebcs_latency_mean_us`, `--iebcs_latency_jitter_us` | Implemented; packet-local ordering only |
| Threshold reset | `--iebcs_resample_thresholds_on_event` | Implemented after each frame packet, not after each event |
| Contrast latency | `--iebcs_contrast_latency_model`, `--iebcs_latency_tau_us`, `--iebcs_latency_clamp_us`, `--iebcs_latency_slope_jitter` | Component-aligned logarithmic latency shape |
| Histogram noise | `--iebcs_hist_noise_model`, `--iebcs_noise_source`, `--iebcs_noise_preset`, explicit path flags | Implemented with external files; state lifecycle incomplete |
| Refractory coupling | `--iebcs_refractory_state_coupling`, `--iebcs_refractory_us` | Implemented but behaviorally incomplete |
| Non-uniform V2CE timing | `--v2ce_nonuniform_burst_timestamps`, `--v2ce_burst_timestamps_mode {random,slope}` | Frame-global layer-relocation heuristic |

## IEBCS-Inspired Mechanisms

### Simple latency and jitter

For event \(i\), the post-generation model samples

$$
\delta_i \sim \mathcal N(\mu_{\mathrm{lat}},\sigma_{\mathrm{lat}}),
\qquad
t_i' = t_i+\max(\delta_i,0).
$$

The current frame packet is then sorted by \(t_i'\), and an optional
signal/noise label receives the same permutation.

Current limitations:

- Sorting is only within one `generate_events` result.
- A delayed event from packet \(k\) can be later than an event already emitted
  from packet \(k+1\), so the complete stream can be non-monotonic.
- The simple offset is applied after event formation; IEBCS uses latency in the
  signal-crossing and refractory lifecycle.
- The model therefore produces delayed/jittered timestamps but is not
  IEBCS-output-equivalent.

### Threshold reset

For a pixel that emitted at least one signal event in the current frame packet,
the corresponding threshold is resampled once:

$$
\theta_+' \sim \mathcal N(\bar\theta_+,\sigma_\theta),
\qquad
\theta_-' \sim \mathcal N(\bar\theta_-,\sigma_\theta),
$$

followed by a lower clamp at \(0.01\).

Current limitations:

- v2e computes all event counts for the frame first and resamples after the
  packet. IEBCS resamples after each event and can immediately produce further
  crossings.
- This gives cross-frame threshold variability, not IEBCS's iterative
  same-frame reset dynamics.
- `set_dvs_params()` changes the active preset thresholds but not the stored
  nominal thresholds used by this resampling and shot-noise scaling.

### Contrast-dependent latency

For an emitted event, the implementation computes

$$
d=\left|L_{\mathrm{photo}}+\eta_{\mathrm{photo}}-L_{\mathrm{mem}}\right|,
\qquad
a=\operatorname{clamp}\left(\frac{\theta}{d},\epsilon,1-\epsilon\right),
$$

$$
\ell
=
\operatorname{clamp}
\left(
\mu_{\mathrm{lat}}
-\tau_{\mathrm{lat}}\log(1-a)
+\xi,\,
0,\,
\ell_{\max}
\right),
\qquad
t'=t+\ell.
$$

This is component-aligned with IEBCS's logarithmic contrast/latency shape.
It differs in intensity-dependent time-constant handling, jitter
parameterization, state lifecycle, and packet ownership. It also inherits the
cross-packet ordering defect.

### Histogram-driven background noise

Each pixel selects an ON and OFF CDF row. A uniform sample \(u\) selects a
frequency bin

$$
f=F_{\mathrm{row}}^{-1}(u),
\qquad
\Delta t_{\mathrm{noise}}=\frac{1}{\max(f,10^{-6})}.
$$

The initial schedule currently uses a random phase

$$
t_{\mathrm{next}}=\phi\,\Delta t_{\mathrm{noise}},
\qquad
\phi\sim\mathcal U(0,1),
$$

and subsequent due events add another sampled delay.

Current limitations:

- The initial timestamp is relative to absolute zero, not the first input-frame
  timestamp. Streams beginning at nonzero time can emit stale events.
- Histogram events bypass signal refractory state and do not update the
  comparator memory/reset state used by later signal events.
- The safety cap processes ON candidates before OFF candidates, causing ON bias
  when saturated; partial selection is also row-major.
- IEBCS updates pixel voltage state when noise events occur.
- v2e can emit multiple due events per polarity during one frame update, whereas
  the reference update lifecycle differs.

The CLI defines presets `3klux`, `161lux`, and `0.1lux`, but this repository
does not contain the six expected `input/iebcs_noise/*.npy` files. Explicit
`files` mode is usable when valid CDF arrays are supplied.

### Refractory-state coupling

For a pixel whose release time lies in the current interval, v2e linearly
interpolates memory:

$$
\alpha
=
\operatorname{clamp}
\left(
\frac{t_{\mathrm{release}}-t_{k-1}}{\Delta t_k},
0,
1
\right),
$$

$$
L_{\mathrm{mem}}'
=
L_{\mathrm{mem}}
+\alpha
\left(
L_{\mathrm{photo}}+\eta_{\mathrm{photo}}-L_{\mathrm{mem}}
\right).
$$

The event maps were already quantized before this interpolation. The current
code does not recompute crossings after the state change. IEBCS instead evolves
the release state exponentially and performs renewed threshold checks.

A zero-duration coupled refractory option is therefore not currently an inert
configuration. This mechanism must be classified as behaviorally incomplete.

## V2CE-Inspired Timestamp Modes

Assume a frame interval \((t_{k-1},t_k]\) contains \(N\) global event layers.
The legacy v2e path uses

$$
t_i=t_{k-1}+i\frac{t_k-t_{k-1}}{N},
\qquad i=1,\ldots,N.
$$

The optional modes draw one vector shared by every active pixel:

$$
u_i\sim\mathcal U(0,1),
$$

$$
t_i^{\mathrm{random}}
=
t_{k-1}
+(t_k-t_{k-1})\operatorname{sort}(u_i),
$$

$$
t_i^{\mathrm{slope}}
=
t_{k-1}
+(t_k-t_{k-1})\operatorname{sort}(\sqrt{u_i}).
$$

Both modes preserve event count, polarity, coordinates, and the number of
global timestamp layers. They irregularly relocate those layers; they do not
remove them.

Differences from V2CE:

- No learned event-voxel prediction.
- No per-voxel, per-pixel, per-polarity timestamp sampling.
- No slope inferred from neighboring temporal bins.
- No local dynamic-aware timestamp inference.
- The fixed `sqrt(U)` transform is an end-biased proxy, not the V2CE slope
  density.
- Legacy refractory gating decides whether to filter using nominal linear
  spacing. Randomly adjacent layers can consequently violate the configured
  refractory interval.

The current modes are useful experimental v2e timing heuristics, but neither is
V2CE output-equivalent or a demonstrated de-layering implementation.

## Interaction and Stream Contracts

- [x] Per-instance random generators isolate emulator streams from ambient
      Python, NumPy, and PyTorch RNG state.
- [x] Packet-local label permutations follow latency sorting.
- [x] HDF5 buffering tracks logical and physically written event counts
      separately.
- [ ] Enforce global timestamp monotonicity across packets and writers.
- [ ] Apply actual-timestamp refractory checks to non-uniform V2CE layers.
- [ ] Integrate histogram events with refractory and comparator-memory state.
- [ ] Define safe `reset()` behavior while HDF5 output is active.
- [ ] Preserve HDR precision through the complete `v2e.py` no-SloMo path.

## Validation Standard

Current unit tests validate local mechanics and CLI wiring. They do not validate
reference equivalence. A parity claim requires:

- [ ] Derived deterministic fixtures from the local IEBCS and V2CE reference
      repositories.
- [ ] Event-count, polarity, coordinate, timestamp-distribution, and state
      comparisons at a shared input/time convention.
- [ ] Cross-packet ordering and writer round-trip checks.
- [ ] Interaction tests for latency, refractory, histogram noise, and V2CE
      timing.
- [ ] Explicit tolerances and a documented list of intentionally different
      behavior.

See:

- [`repo_capabilities_status.md`](repo_capabilities_status.md)
- [`implementation_review_report.md`](implementation_review_report.md)
- [`developments/consolidation_staged_plan.md`](developments/consolidation_staged_plan.md)
- [`developments/v2ce_timing_staged_plan.md`](developments/v2ce_timing_staged_plan.md)
