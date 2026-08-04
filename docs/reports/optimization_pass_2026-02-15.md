# Archived v2e Performance Optimization Pass

> Frozen snapshot from 2026-02-15, retained for provenance. Its open "Future
> Work" items moved into
> [`../developments/performance_optimization_opportunities.md`](../developments/performance_optimization_opportunities.md);
> the correctness defect it flags in the scalar low-pass path is
> [CORE-001](../findings/core_model.md#core-001).
>
> Do **not** treat the speedups, FPS figures, TBB version, or test count below
> as current results. A later rerun produced materially different timing
> ratios, and the current environment has a newer TBB runtime. Current status
> is in [`../repo_capabilities_status.md`](../repo_capabilities_status.md).

Date: 2026-02-15
Branch: `feature/extend_error_models_IEBCS_V2CE`
Baseline commit: `01c0155` ([MAJOR] Implement relevant IEBCS features)

## Optimization Overview

Implemented targeted performance optimizations focusing on:

1. Low-level kernel optimizations (Map_linear_to_log_luminance, LowPassFilter)
2. CPU-GPU transfer patterns
3. Event buffer management
4. Histogram parallelization

## Results

### ✅ Successful Optimizations

| Optimization | Micro-benchmark Speedup | Status |
|--------------|-------------------------|--------|
| **LowPassFilter (scalar ε)** | **2.29x** (24.8µs → 10.8µs) | ✅ Implemented |
| **CPU-GPU `.item()`** | **1.27x** (9.7µs → 7.6µs) | ✅ Implemented |
| **CPU-GPU `.cpu().numpy()`** | **1.51x** (68.4µs → 45.2µs) | ✅ Implemented |

### ⚠️ Mixed Results / Reverted

| Optimization | Micro-benchmark Result | End-to-End Impact | Decision |
|--------------|----------------------|-------------------|----------|
| **Pre-allocated event buffer** | **0.51x slower** (torch.cat highly optimized on CUDA) | No measurable slowdown in real emulator | ❌ Reverted (see note below) |
| **LowPassFilter (tensor ε)** | **0.63x** (reverted to original) | Neutral (same as baseline) | ✅ Reverted to original |

### ❌ Not Beneficial

| Optimization | Result | Reason |
|--------------|--------|--------|
| **Parallel histogram** | **0.00x** (catastrophically slow) | Requires TBB >= 2021.6 (have 12050, need 12060) |
| **Map_linear_to_log_luminance float64→float32** | Not implemented | Would break OFF event generation (numerical precision required) |

## Event Buffer Analysis (Reverted)

The pre-allocated buffer showed **0.51-0.57x in micro-benchmarks** (about 2x slower):

**Micro-benchmark (isolated):**

- Old (list + torch.cat): 766µs for 200 iterations
- New (pre-allocated buffer): 1495µs for 200 iterations
- Speedup: **0.51x** (2x slower)

**Why torch.cat() is faster:**

- torch.cat() is highly optimized on CUDA for concat patterns
- Slice assignment `buf[idx:idx+n] = ev` is less optimized on GPU
- CUDA memory coalescing and kernel fusion favor torch.cat()

**Why reverted despite neutral end-to-end impact:**

- Micro-benchmark consistently shows 2x regression
- No measurable end-to-end improvement to justify complexity
- torch.cat() pattern is simpler and more maintainable
- **Decision**: Reverted, added comment in code with benchmark results for future reference

## Benchmark Details

### Micro-benchmarks

Platform: CUDA (GPU-accelerated)
Resolution: DAVIS346 (346x260)
Tool: `test/test_perf_benchmarks.py`

### End-to-end Benchmarks

Platform: CUDA
Resolution: DAVIS346 (346x260)
Frames: 30-60
Tool: `test/test_end_to_end_perf.py`, `v2ecore/benchmarks/benchmark_emulator.py`

**DAVIS346 Performance:**

- Throughput: 81-97 frames/s
- Event rate: 7-8M events/s
- Per-frame latency: 10-12 ms/frame

**DAVIS240 Performance:**

- Throughput: 118 frames/s
- Event rate: 5.2M events/s
- Per-frame latency: 8.5 ms/frame

**High Event Rate Scenario:**

- Throughput: 26 frames/s
- Event rate: 2.4M events/s
- Per-frame latency: 39 ms/frame

## Code Changes

### Modified Files

- `v2ecore/emulator_utils.py`: LowPassFilter optimization, Map_linear_to_log_luminance TODO comment
- `v2ecore/emulator.py`: CPU-GPU transfer cleanup, event buffer comment with benchmark results
- `v2ecore/v2e_utils.py`: Parallel histogram implementation
- `v2ecore/renderer.py`: Use auto-selecting histogram function

### New Test Files

- `test/test_optimizations.py`: 18 unit tests for optimized code paths
- `test/test_perf_benchmarks.py`: 11 micro-benchmarks for individual optimizations
- `test/test_end_to_end_perf.py`: 3 end-to-end emulator throughput benchmarks

### Test Coverage

- Validation at review time:
  `conda run -n v2e python -m pytest -q`
- Result:
  **83 tests passing**
- Current branch validation is tracked in
  [`docs/repo_capabilities_status.md`](../repo_capabilities_status.md).

## Implementation Details

### 1. LowPassFilter Optimization (`v2ecore/emulator_utils.py`)

**Scalar ε path** (intensity-independent):

```python
# Before: allocating expression
return lp_log_frame + eps * (log_new_frame - lp_log_frame)

# After: in-place lerp_()
lp_log_frame.lerp_(log_new_frame, eps)
return lp_log_frame
```

**Result**: 2.29x speedup (24.8µs → 10.8µs)

**Tensor ε path** (intensity-dependent):

- Kept original non-in-place expression
- PyTorch fuses element-wise ops into fewer CUDA kernels than lerp_() with tensor weight
- lerp_() with tensor weight was 0.48x (2x slower)

### 2. CPU-GPU Transfer Cleanup (`v2ecore/emulator.py`)

```python
# Before
max_num_events = max_num_events_any_pixel.cpu().numpy().item()
events_np = events.cpu().data.numpy()

# After
max_num_events = max_num_events_any_pixel.item()
events_np = events.cpu().numpy()
```

**Result**: 1.27x and 1.51x speedup for scalar and array transfers

### 3. Event Buffer (`v2ecore/emulator.py`) - REVERTED

```python
# Kept original torch.cat pattern
signal_event_chunks: list[torch.Tensor] = []
for i in range(max_num_events_any_pixel):
    # ... create events_curr_iter ...
    signal_event_chunks.append(events_curr_iter)
events = torch.cat(signal_event_chunks, dim=0)
```

**Result**: Pre-allocated buffer was 0.51x (2x slower) in micro-benchmarks. torch.cat() is highly optimized on CUDA. Added comment in code documenting the attempt and benchmark results.

### 4. Parallel Histogram (`v2ecore/v2e_utils.py`)

```python
@njit(nogil=True, parallel=True)
def hist2d_numba_parallel(tracks, bins, ranges):
    # Thread-local partial histograms to avoid race conditions
    partials = np.zeros((n_chunks, bins[0], bins[1]), dtype=np.float64)
    for c in prange(n_chunks):  # parallel loop
        # ... bin events into partials[c] ...
    # Merge partials
    return np.sum(partials, axis=0)

def hist2d_numba(tracks, bins, ranges):
    # Auto-select: only use parallel for >1M tracks
    if tracks.shape[1] > 1_000_000:
        return hist2d_numba_parallel(tracks, bins, ranges)
    return hist2d_numba_seq(tracks, bins, ranges)
```

**Result**: Parallel version requires TBB >= 2021.6 (interface version 12060). Current system has 12050. Auto-selector effectively disables parallel path for all realistic track counts.

## Recommendations

### Kept

- [x] LowPassFilter scalar epsilon optimization (clear 2.29x win)
- [x] CPU-GPU transfer cleanup (1.27x, 1.51x wins)
- [x] Unit tests and benchmarks from the optimization pass
- [x] Event buffer comment documenting attempted optimization and why it was
      reverted

### Future Work

- [x] The current environment is no longer blocked on the historical TBB
      version recorded above.
- [ ] Add a test/benchmark that actually activates the
      `>1,000,000`-track parallel histogram branch.
- [ ] Fix and test the scalar low-pass large-epsilon correctness defect before
      retaining its optimization.
- [ ] Profile end-to-end to identify if event buffer has hidden costs
- [ ] Investigate other PyTorch kernel fusion opportunities
- [ ] Consider a float32 path for `Map_linear_to_log_luminance` only with
      extensive regression testing
- [ ] Use the July 4 benchmark evidence to guide shared C++/CUDA backend work
      rather than treating Cython/TorchScript as the primary porting path
- [ ] Use the active development roadmap for larger work:
      [`docs/developments/performance_optimization_opportunities.md`](../developments/performance_optimization_opportunities.md)

## Validation

Optimizations in this pass were verified through:

- Correctness: 83 tests passing in the validated `v2e` conda environment at
  review time
- Performance: micro-benchmarks showed 1.27-2.29x for successful optimizations
- End-to-end: emulator maintained 81-97 FPS on DAVIS346 at 346x260
- Regression: no existing tests broken, no output format changes
- Current branch validation: see
  [`docs/repo_capabilities_status.md`](../repo_capabilities_status.md)

## Conclusion

Successfully implemented **3 optimizations** with measurable speedups (2.29x, 1.27x, 1.51x) in critical code paths:

- **LowPassFilter (scalar ε)**: 2.29x speedup using in-place lerp_()
- **CPU-GPU transfers**: 1.27x and 1.51x speedups using cleaner patterns

Event buffer pre-allocation was tested but reverted (0.51x regression - torch.cat() is highly optimized on CUDA). Parallel histogram requires TBB upgrade to be beneficial.

Overall emulator performance: **97 FPS** on DAVIS346 (346x260) with photoreceptor noise enabled.

The original optimization pass ended with 83 tests passing in the validated
`v2e` conda environment. Current branch validation is tracked in
[`docs/repo_capabilities_status.md`](../repo_capabilities_status.md).
