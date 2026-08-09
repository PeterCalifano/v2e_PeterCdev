# v2e Findings Register

Scope: **one durable description per defect.** This directory is the only place
that explains *what is wrong, why it matters, and how to fix it*. Nothing here
is a task list, a status report, or a plan.

Ownership contract for the whole `docs/` tree is in
[`../README.md`](../README.md).

## What belongs here

- Confirmed defects, gaps, incorrect calls, and reference discrepancies.
- The reasoning that makes each one a defect rather than a design choice.
- A concrete suggested fix.

## What does not belong here

- Checkboxes, stage ordering, or scheduling. Those live in
  [`../developments/`](../developments/).
- Capability or validation claims. Those live in
  [`../repo_capabilities_status.md`](../repo_capabilities_status.md).
- Narrative point-in-time audits. Those are frozen in
  [`../reports/`](../reports/).

## Files

| File | Area |
|---|---|
| [`core_model.md`](core_model.md) | Lin-log, photoreceptor low-pass, HDR preprocessing |
| [`state_lifecycle.md`](state_lifecycle.md) | `reset()`, `set_dvs_params()`, per-pixel state ownership |
| [`iebcs_extensions.md`](iebcs_extensions.md) | Latency, threshold reset, histogram noise, refractory coupling |
| [`v2ce_timing.md`](v2ce_timing.md) | Non-uniform burst timestamp modes |
| [`stream_and_io.md`](stream_and_io.md) | Event ordering, writers, timestamp precision |
| [`tooling_and_benchmarks.md`](tooling_and_benchmarks.md) | Benchmarks, estimators, packaging, dead code |

## Identifier convention

`AREA-NNN`, allocated once and never reused: `CORE`, `LIFE`, `IEBCS`, `V2CE`,
`STREAM`, `TOOL`. Plans and status docs cite these IDs instead of restating a
defect. When a finding is fixed, mark it `Resolved` in place and keep the entry
— the ID must keep resolving for older references.

## Severity

| Level | Meaning |
|---|---|
| **Critical** | Crashes, or silently produces an invalid event stream on a documented path |
| **High** | Materially wrong model behavior on an advertised opt-in path |
| **Medium** | Wrong or misleading in bounded conditions; correct default path |
| **Low** | Cosmetic, dead code, or documentation/claim accuracy |

## Verification status

| Marker | Meaning |
|---|---|
| **Confirmed (probe)** | Reproduced by an executable probe; the observed value is quoted in the entry |
| **Confirmed (inspection)** | Established by reading the implementation and, where relevant, the reference source |
| **Open question** | A stated rationale that the evidence does not support; needs a decision or an experiment |

Probes used for this pass were run against the working tree at commit
`2d109f4` in the `v2e` conda environment (Python 3.11.11) on CPU. They are
investigation scripts, not committed tests; the corresponding regression tests
are listed as work items in
[`../developments/consolidation_staged_plan.md`](../developments/consolidation_staged_plan.md).

## Summary

| ID | Severity | Status | Title |
|---|---|---|---|
| [CORE-001](core_model.md#core-001) | Medium | Resolved (test) | Scalar low-pass path lacked an `eps <= 1` clamp |
| [CORE-002](core_model.md#core-002) | Medium | Resolved (test) | `LowPassFilter` mutated caller state on one path only |
| [CORE-003](core_model.md#core-003) | High | Resolved (test) | `apply_low_pass_filter` silently ignored `filter_tau_const` |
| [CORE-004](core_model.md#core-004) | Open question | Confirmed (probe) | float64 lin-log rationale is not supported by its own rounding grid |
| [CORE-005](core_model.md#core-005) | Medium | Confirmed (inspection) | HDR preprocessing has no dark-response floor |
| [CORE-006](core_model.md#core-006) | High | Resolved (CLI test) | No-SloMo CLI processing destroyed floating-point HDR contrast |
| [LIFE-001](state_lifecycle.md#life-001) | Critical | Confirmed (probe) | `reset()` then reuse raises `TypeError` |
| [LIFE-002](state_lifecycle.md#life-002) | High | Confirmed (probe) | `reset()` does not rewind `t_previous` |
| [LIFE-003](state_lifecycle.md#life-003) | Critical | Confirmed (probe) | `set_dvs_params()` mid-stream raises `AttributeError` |
| [LIFE-004](state_lifecycle.md#life-004) | High | Confirmed (probe) | `set_dvs_params()` discards per-pixel threshold mismatch |
| [LIFE-005](state_lifecycle.md#life-005) | Medium | Confirmed (inspection) | `reset()` and active HDF5 storage have no defined contract |
| [IEBCS-001](iebcs_extensions.md#iebcs-001) | High | Confirmed (probe) | Histogram noise schedule is anchored to absolute zero |
| [IEBCS-002](iebcs_extensions.md#iebcs-002) | High | Confirmed (probe) | CDF loader uses a running maximum, not a cumulative sum |
| [IEBCS-003](iebcs_extensions.md#iebcs-003) | Medium | Confirmed (inspection) | CDF normalisation divisor differs from the IEBCS reference |
| [IEBCS-004](iebcs_extensions.md#iebcs-004) | High | Confirmed (probe) | Zero-duration refractory coupling is not inert |
| [IEBCS-005](iebcs_extensions.md#iebcs-005) | High | Confirmed (probe) | Refractory interpolation edits memory after quantisation |
| [IEBCS-006](iebcs_extensions.md#iebcs-006) | Medium | Confirmed (inspection) | Noise cap is polarity-ordered and spatially biased |
| [IEBCS-007](iebcs_extensions.md#iebcs-007) | Medium | Confirmed (inspection) | Preset noise assets are not present in the repository |
| [STREAM-001](stream_and_io.md#stream-001) | High | Confirmed (probe) | Latency breaks global timestamp monotonicity |
| [STREAM-002](stream_and_io.md#stream-002) | Medium | Confirmed (inspection) | Writer timestamp ranges wrap without an explicit contract |
| [STREAM-003](stream_and_io.md#stream-003) | Low | Confirmed (inspection) | Monotonicity check warns per packet and cannot see the real defect |
| [V2CE-001](v2ce_timing.md#v2ce-001) | High | Confirmed (inspection) | Timestamp layers remain frame-global |
| [V2CE-002](v2ce_timing.md#v2ce-002) | High | Confirmed (inspection) | Refractory gating uses nominal spacing, not actual gaps |
| [TOOL-001](tooling_and_benchmarks.md#tool-001) | Medium | Confirmed (probe) | Comparative benchmark resolves the wrong repository root |
| [TOOL-002](tooling_and_benchmarks.md#tool-002) | Medium | Confirmed (inspection) | Benchmark sorts the stream before measuring stream validity |
| [TOOL-003](tooling_and_benchmarks.md#tool-003) | Low | Confirmed (inspection) | Dead polynomial helper duplicated inline |
| [TOOL-004](tooling_and_benchmarks.md#tool-004) | Low | Confirmed (inspection) | Noise-rate warning text contradicts its own threshold |
| [TOOL-005](tooling_and_benchmarks.md#tool-005) | Medium | Confirmed (inspection) | Noise calibration loop length is unbounded in Python |
| [TOOL-006](tooling_and_benchmarks.md#tool-006) | Low | Confirmed (inspection) | Package version disagrees with the changelog |
| [TOOL-007](tooling_and_benchmarks.md#tool-007) | Low | Confirmed (inspection) | Parallel histogram branch is never exercised |
