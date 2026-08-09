# Plan: Tooling and Repository Hygiene

Scope: benchmark harnesses, estimator numerics, dead code, packaging metadata,
and test-suite hygiene. Owns
[TOOL-001](../findings/tooling_and_benchmarks.md#tool-001) through
[TOOL-007](../findings/tooling_and_benchmarks.md#tool-007).

Not in scope: the accelerated backend roadmap, which is
[`performance_optimization_opportunities.md`](performance_optimization_opportunities.md).

Status: active. TOOL-001 and TOOL-007 are complete; TOOL-002 through TOOL-006
remain open. Items here are independent of each other, except TOOL-002 which is
coordinated with
[`plan_event_ordering.md`](plan_event_ordering.md).

## Benchmarks

- [x] Fix `REPO_ROOT` to `parents[2]` in
      `benchmark_error_models_eventstream.py` (TOOL-001).
- [x] Add a functional test for the comparative benchmark's repository-relative
      default and a real run from outside the checkout (TOOL-001).
- [x] Remove the empty generated `v2ecore/output/benchmarks_comparative/`
      directory. Repository-root `output/*` is already ignored; the tracked
      `v2ecore/output/` writer package must not be ignored (TOOL-001).
- [ ] Measure and report raw-stream monotonicity violations **before** any
      sorting: count, largest backward step, and the packet boundary where each
      occurs (TOOL-002).
- [ ] Restrict the sorted copy to metrics that are genuinely order-invariant,
      and label sorted-derived metrics in the JSON/CSV output (TOOL-002).
- [x] Remove the pytest-collected print-only timing functions so they are not
      mistaken for regressions.

## Estimator numerics

- [ ] Delete the inline duplicate of the Graca–Delbruck polynomial and call
      `_compute_vn_from_log_rate_per_hz`; keep the `KEY[G-PHOTO-VRMS-FIT]`
      anchor on the surviving copy (TOOL-003).
- [ ] Resolve the `# DOUBT what's this?` comment by pointing at that helper's
      docstring and the spreadsheet derivation (TOOL-003).
- [ ] Fix the warning text to quote its own threshold constant (TOOL-004).
- [ ] Replace the scalar Python IIR loop with a vectorised filter, or with the
      closed-form variance ratio if it agrees with the simulated value
      (TOOL-005).
- [ ] Bound the calibration sample count and warn when the bound binds
      (TOOL-005).
- [ ] Document the `sample_rate_rel_tolerance = 0.1` cache tolerance as a
      deliberate choice, or drop it once the computation is cheap (TOOL-005).

## Packaging

- [ ] Set `pyproject.toml` to the newest released version in `CHANGELOG.md`
      (TOOL-006).
- [ ] Add a test asserting the installed version matches the changelog's top
      entry (TOOL-006).
- [ ] Reconcile `AGENTS.md` (Python >= 3.12) with the validated environment
      (Python 3.11.11), or state explicitly that `AGENTS.md` describes a
      cross-project preference rather than this repository's requirement.

## Test suite

- [x] Add a correctness test crossing the `1_000_000`-track threshold,
      asserting the public histogram equals the sequential result exactly
      (TOOL-007).
- [x] Re-measure the warmed kernels on the current runtime. Retain the parallel
      branch because it is about 3.1 times faster at `1,000,001` tracks while
      producing the exact sequential result (TOOL-007).
- [x] Remove the duplicated emulator test shared between
      `test_optimizations.py` and `test_emulator_regression.py`.

## Gate

- [x] `conda run -n v2e python -m pytest -q` (`100 passed` in the isolated
      histogram/test-hygiene candidate).
- [x] `conda run -n v2e python -m compileall -q v2e.py v2ecore test`.
- [x] `python3.12 -m compileall -q v2e.py v2ecore test`.
- [x] `git diff --check`.
- [x] Comparative benchmark runs from a directory other than the repository
      root and writes smoke-test artifacts to a temporary output directory.
- [x] Mark resolved findings in
      [`../findings/tooling_and_benchmarks.md`](../findings/tooling_and_benchmarks.md).
