# Performance and Shared-Backend Development Plan

Status: open; blocked on correctness contracts for every model being ported

Temporary July 2026 CUDA, Julia, Python-extension, and ETAP harnesses are
development evidence outside the supported repository implementation. They
must not be reported as an installed backend capability.

## Evidence Retained

- [x] Clean count maps and event sets matched the Python reference on selected
      640x480, 1024x768, and 1920x1080 fixtures.
- [x] A deterministic contrast-latency subset matched within floating-point
      tolerance after normalized comparison.
- [x] Preallocated CUB scan avoided the allocation cost seen in the Julia
      prototype; this is an implementation difference, not a Python-versus-Julia
      language conclusion.
- [x] Host download and file formatting often cost more than the tested event
      extraction kernel.
- [x] ETAP experiments showed that saved float64 representations and CPU
      normalization can dominate model-forward time.
- [x] A high-level GPU event-stack prototype demonstrated the value of
      GPU-resident downstream representations.

Evidence limitations:

- [ ] The full stochastic model set was not ported.
- [ ] Current IEBCS/V2CE correctness defects were not solved by the prototypes.
- [ ] Writer parity and end-to-end packaging were not implemented.
- [ ] Temporary benchmark artifacts are not reproducible from a clean checkout.

## Stage P0: Correctness and Reproducible Baseline

- [ ] Complete the applicable stages in
      [`consolidation_staged_plan.md`](consolidation_staged_plan.md).
- [ ] Define reference fixtures for clean, noisy, refractory, latency, HDR, and
      high-burst cases.
- [ ] Separate kernel time, event materialization, representation construction,
      serialization, and end-to-end time.
- [ ] Record hardware, driver, CUDA, PyTorch, Python, dtype, resolution, event
      count, and warmup policy in machine-readable output.
- [ ] Move stable benchmark harnesses into the repository only after their
      contracts are reviewed.
- [ ] Treat projected speedups as hypotheses until reproduced.

## Stage P1: Stable Shared Backend Boundary

Goal: one C++17/20 + CUDA backend callable from Python and Julia, with
Python/PyTorch retained as the reference fallback.

- [ ] Define typed inputs, persistent state, RNG ownership, output ownership,
      and error semantics before writing kernels.
- [ ] Start with the clean deterministic count/emit/update path only.
- [ ] Return GPU-resident event tensors without forced host materialization.
- [ ] Provide a direct GPU event-representation/event-stack path for consumers
      that do not need raw events.
- [ ] Keep writer APIs outside the first kernel boundary.
- [ ] Avoid parallel independent Python and Julia implementations.
- [ ] Add CPU-only fallback and clear build/runtime capability detection.

## Stage P2: Deterministic Parity Expansion

- [ ] Lin-log/HDR preprocessing parity.
- [ ] Intensity-dependent low-pass parity, including dark floor and large
      epsilon behavior.
- [ ] Threshold mismatch and comparator-memory update parity.
- [ ] Leak-disabled and deterministic leak fixtures.
- [ ] CSDVS and SCIDVS only after explicit reference tests exist.
- [ ] Stage-2 V2CE analytic timing only after its own plan is complete.
- [ ] Require count/polarity/coordinate equality and declared timestamp
      tolerances per model.

## Stage P3: Stochastic and Stateful Parity

- [ ] Specify a counter-based or otherwise cross-language RNG contract.
- [ ] Preserve private per-context reproducibility and processing-order
      independence.
- [ ] Add shot noise, photoreceptor noise, threshold reset, and leak mismatch
      one mechanism at a time.
- [ ] Add IEBCS latency, histogram, and refractory paths only after their
      lifecycle fixes and fixtures pass in Python.
- [ ] Compare distributions as well as fixed-seed exact output where exact RNG
      parity is intended.

## Stage P4: Writers and Downstream Integration

- [ ] Validate HDF5, AEDAT-2, AEDAT-4, text, labels, and delayed-tail
      finalization separately from kernel timing.
- [ ] Add EventDataGenerationLib adapter support without exposing backend-owned
      details to event-based-centroiding.
- [ ] Benchmark
      `renderer -> GPU events -> GPU representation -> model`
      against saved-event and saved-representation paths.
- [ ] Define lifetime/stream synchronization for zero-copy tensors.
- [ ] Package and test from a clean environment.

## Secondary Python Opportunities

These are measurement candidates, not the main architecture:

- [ ] Profile lin-log precision/cast cost before considering a float32 variant.
- [ ] Re-evaluate the >1,000,000-track histogram branch on the current TBB/Numba
      environment; do not retain the obsolete “TBB unavailable” claim.
- [ ] Remove or isolate pytest-collected print-only timing functions.
- [ ] Profile the photoreceptor-noise calibration loop and remove dead duplicate
      polynomial code.
- [ ] Reject Cython/TorchScript work unless a current profile shows a dominant
      bottleneck that the shared backend does not address.

## Completion Criteria

- [ ] Every advertised accelerated model has fixture-backed parity.
- [ ] Full output order and writer behavior match the reference contract.
- [ ] GPU-resident and materialized-output benchmarks are reported separately.
- [ ] Build/package tests pass on supported CPU and CUDA environments.
- [ ] Current capability docs distinguish supported code from development
      evidence.
