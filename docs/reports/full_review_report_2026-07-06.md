# Archived Full Implementation Review - `feature/extend_error_models_IEBCS_V2CE`

> Historical audit snapshot from 2026-07-06. Its confirmed defects have been
> rechecked and merged into the current
> [`../findings/`](../findings/) register.
> Test counts and file locations in this archive intentionally remain tied to
> that working tree.

- **Date:** 2026-07-06
- **Scope:** fork vs upstream divergence, development-plan status, model correctness of
  IEBCS/V2CE error-model extensions (verified against local `../IEBCS` source),
  confirmed bugs, validation gaps, optimization review, test-suite audit.
- **State reviewed:** working tree on `feature/extend_error_models_IEBCS_V2CE`
  (uncommitted enum/HDF5/docs changes included). Test suite: **97 passed** (~10 s).
- **Method:** full read of `v2ecore/emulator.py`, `v2ecore/emulator_utils.py`,
  `v2ecore/model_options.py`, output backends, all test files, all development plans;
  cross-check against `../IEBCS/src/dvs_sensor.py`; bugs confirmed empirically by
  running a verification script against the emulator (results quoted below).

---

## TL;DR

The branch is in good shape overall — the development plans accurately describe what is
implemented, the IEBCS histogram-noise mechanism is genuinely faithful to the reference,
the docs are unusually honest about scope, and all 97 tests pass. However, **three real
bugs were confirmed empirically**:

1. **B1** — the global event stream becomes non-monotonic *across frame packets* when any
   latency model is enabled;
2. **B2** — refractory-state coupling is not a no-op at zero refractory and never
   re-checks thresholds after release interpolation, despite the docs claiming it does;
3. **B3 (hist-noise anchor)** — the histogram-noise schedule is anchored at t=0 rather
   than the first frame time, producing a capped garbage burst if the video does not
   start at t=0.

Plus several smaller precision/robustness issues (B4–B6) and one fully duplicated test.

---

## 1. Scope: original v2e vs this fork

Against `upstream/master` (SensorsINI/v2e), the branch carries **93 commits,
~9,700 insertions across 52 files**. Functional deltas:

- IEBCS Stage-1/2 and V2CE error-model extensions (all opt-in, disabled by default).
- HDR input pipeline (`hdr` / `hdr_disable_prepro`, float TIFF/EXR via `IMREAD_UNCHANGED`).
- Deterministic seeding (`--dvs_emulator_seed`).
- Performance work: in-place `lerp_` low-pass, CPU/GPU transfer cleanup, HDF5
  buffered/chunked writes with logical-vs-written count separation.
- Enum-backed finite options (`v2ecore/model_options.py`).
- Comparative benchmark suite and 3D event viewer.
- A large test suite (upstream has essentially none).

Uncommitted working-tree changes match `doc/developments/commit_split_todo.md`
(enum wiring / HDF5+output regressions / docs), including a removed duplicated
`resolve_dvs_emulator_seed` call in `v2e.py` (it was called twice — good catch).

**Development-plan status verified:** `doc/developments/v2ce_timing_staged_plan.md`
Stage 1 items are actually done; Stages 2 (analytic burst timing) and 3
(V2CE-Toolbox cross-check) are genuinely open. The 3-commit split is still pending.
`docs/implementation_review_report.md` (2026-07-04) is mostly accurate; its
"renewed threshold checking" claim for refractory coupling is overstated (see B2).

---

## 2. Confirmed bugs (verified by running the code)

### B1 — Cross-packet non-monotonic timestamps under latency models — **highest priority**

`_apply_latency_jitter_and_sort` (`v2ecore/emulator.py:1268`) and the contrast-latency
path sort **within a packet only**. Latency offsets (mean 100–400 µs, clamp up to 10 ms)
can push event timestamps past `t_frame`, so packet *N* can end later than packet *N+1*
begins.

**Reproduced** with 0.5 ms frame intervals and 400±150 µs latency: 3 backward time steps
across 4 packets, for both `iebcs_latency_jitter_model` and
`iebcs_contrast_latency_model`. This is exactly the regime the latency models are meant
for (frame interval comparable to latency).

Downstream impact: `v2ecore/output/aedat2_output.py:155` casts to int32 µs with no
reordering; jAER and most event consumers assume monotonic streams; the HDF5 `frame_idx`
mapping keeps events attributed to the wrong frame window.

**Fix options:** buffer events whose shifted timestamp exceeds `t_frame` and release them
with the next packet (IEBCS-equivalent — it operates on a global timeline), or clamp
latencies to `t_frame` (cheaper, biased). The existing per-packet monotonicity tests
cannot catch this (they check `np.diff` inside a single packet only).

### B2 — Refractory-state coupling: no renewed threshold check; not a no-op at 0 µs

`docs/README_error_models_extensions.md` §6 claims "release-time interpolation followed
by renewed threshold checking inside the same frame interval." The code does not do
that: `pos_evts_frame`/`neg_evts_frame` are computed **once** from `diff_frame` before
the iteration loop (`emulator.py:1540`), while `_apply_refractory_release_interpolation`
(`emulator.py:1097`) mutates `base_log_frame` mid-loop. Event counts are never
recomputed against the interpolated reference, and the end-of-frame memory update
`base_log_frame += final_pos * pos_thres` (`emulator.py:1763`) is applied *on top of*
the interpolated base — memory and emitted events go inconsistent.

**Confirmed consequence:** with `iebcs_refractory_us=0` (should be behaviorally inert),
coupling changes output from **10,240 to 11,520 events** on a simple flicker sequence.

Also note the interpolation itself deviates from IEBCS: the fork interpolates
**linearly** between memory and current photoreceptor value; IEBCS
(`dvs_sensor.py`, `update()`) computes the photoreceptor's first-order exponential
response `cur_v + (img_l − cur_v)(1 − exp(−Δt/τ_p))` at release time. Worth documenting
even after the count-consistency fix.

### B3 — Histogram-noise schedule anchored at t=0; no lower bound on due events

`_init_iebcs_noise_schedule` (`emulator.py:908`) sets `next = delay * phase` relative to
0, but init runs at the **first frame's timestamp**, whatever it is. The due mask
`next_pos <= t_frame` (`emulator.py:982`) has no `> t_previous` lower bound.

**Confirmed:** first frame at t=100 s → second frame emits a **capped burst of 2,048
events with timestamps starting at ~0.017 s** — 100 seconds before the frame interval —
plus the cap warning.

**Fix:** anchor the schedule at the first frame time (`t_first + delay*phase`), which
also matches IEBCS (whose clock genuinely starts at 0).

### B4 — Scalar-ε low-pass path lacks the stability clamp

`LowPassFilter.__call__` clamps `eps ≤ 1.0` on the tensor path but not the scalar path
(`emulator_utils.py:130–132`): `lp_log_frame.lerp_(log_new_frame, eps)` with
`eps = delta_time/tau > 1` overshoots and can oscillate/diverge. This path filters the
photoreceptor noise array, so a low `cutoff_hz` with a large frame interval silently
amplifies injected noise. `test_large_eps_clamped` covers only the tensor path.
**One-line fix:** `eps = min(delta_over_tau, 1.0)`.

### B5 — HDR `inten01` has no dark floor

Non-HDR input uses `rescale_intensity_frame = (x+20)/275` (`emulator_utils.py:53`),
floor ≈ 0.073, specifically to avoid zero photoreceptor bandwidth. The HDR branches
(`emulator.py:1416–1426`) use `clamp(new_frame, 0, 1)` and `clamp(exp(x)/255, 0, 1)`
with **floor 0**: black HDR pixels get `eps = 0` and their photoreceptor state *never
updates* when `cutoff_hz > 0` — frozen at the first frame forever. For mostly-black
EXR/space imagery this will bite. **Fix:** apply the same +20/275-style offset in the
HDR branches.

### B6 — Timestamp precision (partly inherited, worth documenting)

- `prepare_storage` (`emulator.py:615`): `np.array(frame_ts, dtype=np.float32) * 1e6` —
  float32 µs loses integer precision past ~16.7 s (2²⁴ µs). Use float64 before scaling.
- Event timestamps are float32 tensors throughout; at t = 1000 s the representable step
  is ~61 µs — *larger than the latency/jitter effects being modeled*. Inherited from
  upstream, but the new µs-scale models make it matter now.
- `aedat2_output.py:155` uses int32 µs → overflow at ~35.8 min; HDF5 uint32 µs wraps at
  ~71.6 min (already documented).

### Minor issues

- `PhotoreceptorNoiseVoltageEstimator`: `_compute_vn_from_log_rate_per_hz` is dead code
  whose polynomial is duplicated inline at `emulator_utils.py:391` (the
  `# DOUBT what's this?` comment — it is the Graca–Delbruck fit); the warning at
  line 368 checks `> 0.5` but the message says "larger than 0.1"; the calibration IIR
  loop (`emulator_utils.py:417–419`) is pure Python over up to ~100k samples per cache
  miss — vectorize with `scipy.signal.lfilter` or use the analytic
  noise-equivalent-bandwidth ratio.
- `set_dvs_params` (`emulator.py:762`) overwrites `pos_thres`/`neg_thres` but not
  `*_nominal`, so `--pos_thres 0.15 --dvs_params clean` leaves the shot-noise
  probability scale and resample-nominal inconsistent (inherited from upstream; now also
  affects `iebcs_resample_thresholds_on_event`). It correctly rebuilds
  `low_pass_filter`, which upstream got wrong.
- Hist-noise cap truncation keeps the first `remaining` pixels in row-major order
  (`emulator.py:1001`) — spatially biased toward the top of the frame when the cap
  engages. Random subsampling would be fairer (only matters in pathological configs).

---

## 3. Model fidelity verification (vs papers + local `../IEBCS`)

- **Core v2e model:** unchanged and sound; `# KEY[...]` anchors match
  `docs/core_model_mapping.md`. Lin-log float64 rounding rationale preserved.
- **Histogram noise:** verified against `../IEBCS/src/dvs_sensor.py` — the fork's
  mechanism is a **faithful port**, stronger than the docs claim: per-pixel random
  CDF-row assignment, per-event inverse-CDF frequency draw with `delay = 1/f`, random
  initial phase — identical to `init_bgn_hist`/`get_next_noise`. Two genuine
  differences: (a) IEBCS noise events update `time_px`, coupling noise into
  refractory/latency state; fork noise events touch no signal-path state; (b) IEBCS
  normalizes rows by the second-to-last bin, the fork by the max — negligible. The
  per-frame cap (`8×num_pixels`) is a fork-only safety deviation (documented).
- **Contrast latency:** mean structure `µ − τ·log(1−amp)` with clamp [0, 10 ms] matches
  IEBCS `get_latency_tau`. Deviations: IEBCS τ_p is **per-pixel, intensity-dependent**
  (`τ·1e3/(img+1)`); the fork uses fixed `iebcs_latency_tau_us`. IEBCS jitter grows as
  `√(jit² + (σ_th·τ_p/drive)²)` (unbounded for weak drive); the fork's `√(1+amp²)`
  scaling is bounded by √2. All k events of a same-pixel burst share one `drive`, so
  they get near-identical latencies — an approximation IEBCS does not make. The docs'
  "behaviorally aligned, not identical" claim is accurate; deriving τ from `inten01`
  would close the biggest gap.
- **Latency+jitter and threshold resample:** match stated intent; implementations clean.
- **V2CE:** correctly scoped in docs as a de-layering heuristic
  (`sorted U` / `sorted √U`), not V2CE-equivalent. The Stage-2 plan (analytic per-pixel
  timing from `diff_frame` and thresholds) is the right direction and would subsume the
  `slope` proxy.

---

## 4. Validation gaps

Already flagged in docs (agreed): no distribution-level tests vs IEBCS/V2CE references;
preset `.npy` noise assets absent; shared C++/CUDA backend is a roadmap, not a
capability.

Additional gaps found in this review — each maps to a confirmed bug the current suite
could not catch:

- No test exercises **multi-frame** streams with latency models (hence B1 survived).
- No zero-parameter no-op equivalence test for refractory coupling (hence B2).
- No nonzero-start-time input test for hist noise (hence B3).
- No HDR + `cutoff_hz > 0` test (hence B5).
- Scalar-ε clamp path untested (hence B4).

---

## 5. Optimization review

Applied optimizations are correct and covered by `test_optimizations.py` (in-place
`lerp_`, fused tensor path, HDF5 buffering — the `frame_idx` logical-count fix at
`emulator.py:1817` is subtle and done right). Remaining hot-path items, in impact order:

1. Per-iteration loop over `max_num_events_any_pixel` with `nonzero()` + `randperm` per
   iteration (`emulator.py:1583–1697`). The `randperm` shuffle exists only to mix ON/OFF
   within an iteration; it could be replaced with a single post-hoc shuffle — and with
   latency models enabled the subsequent timestamp sort makes it entirely redundant.
2. Float64 lin-log map (already documented; correctness-first, keep).
3. `PhotoreceptorNoiseVoltageEstimator` Python calibration loop (see minor issues).

The shared C++/CUDA backend direction
(`doc/developments/performance_optimization_opportunities.md`) is sensible; the docs
correctly refuse to call it a capability yet.

---

## 6. Test-suite audit

**Verdict: meaningful overall, small redundancy, targeted gaps.** 97 passed in ~10 s;
tests assert behavior (event equality, threshold mutation, cap+reschedule semantics,
CLI→emulator kwarg forwarding) rather than just "doesn't crash". The strongest tests are
the disabled-path equivalence tests (`test_v2ce_disabled_matches_default_linear_path`,
`test_new_models_disabled_matches_baseline_behavior`) — they protect the compatibility
guarantee.

### Redundancies to remove

- `test_optimizations.py::TestEventGenerationSanity::test_basic_event_generation`
  (line ~245) is a near byte-for-byte duplicate of
  `test_emulator_regression.py::test_generate_events_shape_bounds_and_monotonic_timestamps`
  (line 33) — only the seed differs. `_assert_event_packet_valid` is copy-pasted in both
  files: move it to a `conftest.py` helper and delete the duplicate test.
- `test_moving_edge_generates_events_with_valid_packet` and
  `test_moving_blob_generates_on_and_off_events` assert the same properties. Marginal —
  keep if the scenario variety is valued.
- `test_contrast_latency_model_increases_delay_for_low_slope_cases` compares mean packet
  timestamps between low/high-contrast steps, but the mean is dominated by
  burst-subdivision count, not latency — it would pass even with the latency term
  deleted. Rewrite to compare against a latency-disabled run of the *same* frames.

### Coverage additions (priority order; first three would have caught the confirmed bugs)

1. **Global monotonicity over ≥4 frames** with `iebcs_latency_jitter_model` and
   `iebcs_contrast_latency_model` at frame intervals below the latency scale —
   currently fails; pin after fixing B1.
2. **No-op equivalence:** coupling with 0 µs refractory ≡ baseline; contrast latency
   with `mean=tau=jitter=0` ≡ baseline.
3. **Nonzero start time** for hist noise: first frame at t≫0 must not emit pre-frame
   events.
4. Scalar-ε clamp test (`inten01=None`, `delta_time > tau`).
5. HDR black-frame + `cutoff_hz>0`: photoreceptor state must still converge.
6. Distribution-level fixtures (already planned): KS-test hist-noise inter-arrival
   distribution against the input CDF; check latency mean/std against
   `µ − τ·log(1−amp)`.
7. Long-time precision guard: events at t > 100 s survive the µs round-trip within
   tolerance (currently they do not, cleanly).

---

## 7. Doc corrections needed

The docs are largely accurate. Three claims need adjustment:

1. `README_error_models_extensions.md` §6: "renewed threshold checking inside the same
   frame interval" — not implemented (B2).
2. Monotonicity claims under latency models ("Events are sorted by timestamp afterwards
   to preserve monotonicity") are true **per-packet only** (B1) — add the cross-packet
   caveat until fixed.
3. The histogram-noise section can be *upgraded*: the sampling mechanism is
   mechanism-identical to IEBCS (verified against reference source), with the two
   deviations listed in §3.

---

## Suggested order of attack

B1 (breaks the headline feature's output contract) → B3 + B2 (small, well-localized
fixes) → B4/B5 (one-liners protecting the HDR use case) → coverage tests 1–3 → then the
commit split per `commit_split_todo.md`, folding the fixes into commit 1.
