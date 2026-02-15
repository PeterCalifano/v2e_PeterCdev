# Advanced Optimization Opportunities (JIT/Cython/CUDA/C++)

This document outlines potential performance improvements using JIT compilation, Cython, CUDA kernels, or C++ extensions.

## Current Performance Baseline

**DAVIS346 (346x260):** 97 FPS, 7-8M events/s, 10-12 ms/frame (CUDA)

## 🔥 High-Impact Opportunities

### 1. **Custom CUDA Kernel for Event Extraction**

**Target**: `v2ecore/emulator.py` lines 1454-1527 (event generation loop)

**Current bottleneck:**

```python
for i in range(max_num_events_any_pixel):
    pos_cord = (pos_evts_frame >= i + 1)
    neg_cord = (neg_evts_frame >= i + 1)
    pos_event_xy = pos_cord.nonzero(as_tuple=True)
    neg_event_xy = neg_cord.nonzero(as_tuple=True)
    # ... build events ...
```

**CUDA optimization:**

- **Fused kernel** combining comparison, nonzero, and event building
- Eliminate multiple kernel launches (currently ~4-6 per iteration)
- Direct event buffer writing in GPU memory
- **Expected speedup**: 3-5x for event generation loop

**Implementation approach:**

```cpp
// CUDA kernel pseudocode
__global__ void extract_events_kernel(
    const int32_t* pos_evts,    // [H, W]
    const int32_t* neg_evts,    // [H, W]
    float* events_out,          // [max_events, 4]
    int* event_count_out,
    int iteration,
    float timestamp,
    int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int y = idx / W;
    int x = idx % W;

    // Atomically write events if threshold crossed
    if (pos_evts[idx] >= iteration + 1) {
        int pos = atomicAdd(event_count_out, 1);
        events_out[pos * 4 + 0] = timestamp;
        events_out[pos * 4 + 1] = x;
        events_out[pos * 4 + 2] = y;
        events_out[pos * 4 + 3] = 1.0f;
    }
    // Similar for neg events
}
```

**Integration**: PyTorch C++ extension or CuPy

---

### 2. **Numba JIT for Histogram (Already Implemented, Needs TBB)**

**Target**: `v2ecore/v2e_utils.py` - `hist2d_numba_parallel()`

**Current status:**

- Implemented with `@njit(parallel=True)`
- **Disabled** due to TBB < 12060 (have 12050)

**Action required:**

```bash
# Upgrade TBB
conda install "tbb>=2021.6" -c conda-forge
# Or
pip install "tbb>=2021.6"
```

**Expected speedup** (for >1M tracks): 4-8x over sequential

---

### 3. **Cython for `lin_log` Transformation**

**Target**: `v2ecore/emulator_utils.py` lines 17-46

**Current implementation:**

- Uses float64 (precision required)
- Element-wise operations in PyTorch

**Cython optimization:**

```cython
# lin_log_cy.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double lin_log_scalar(double x, double threshold) nogil:
    if x < threshold:
        return x * (1.0 / threshold) * log(threshold)
    return log(x)

def lin_log_cy(np.ndarray[np.float64_t, ndim=2] x, double threshold=20.0):
    cdef int H = x.shape[0]
    cdef int W = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty_like(x)
    cdef int i, j

    for i in prange(H, nogil=True):
        for j in range(W):
            out[i, j] = lin_log_scalar(x[i, j], threshold)
    return out
```

**Expected speedup**: 2-3x (currently 34µs → ~12µs for 346x260)

**Note**: Requires maintaining float64 precision (TODO comment documents why)

---

### 4. **Custom CUDA Kernel for IIR Lowpass Filter**

**Target**: `v2ecore/emulator_utils.py` - `LowPassFilter.__call__()`

**Current bottleneck** (tensor ε path):

```python
eps = inten01 * delta_over_tau
eps = torch.clamp(eps, max=1.0)
return lp_log_frame + eps * (log_new_frame - lp_log_frame)
```

**CUDA fused kernel:**

```cpp
__global__ void lowpass_filter_kernel(
    const float* log_new,
    float* lp_log,          // in-place update
    const float* inten01,
    float delta_over_tau,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float eps = fminf(inten01[idx] * delta_over_tau, 1.0f);
    lp_log[idx] += eps * (log_new[idx] - lp_log[idx]);
}
```

**Expected speedup**: 1.5-2x (currently 48µs → ~25µs for tensor ε path)

---

### 5. **C++ Extension for `compute_event_map`**

**Target**: `v2ecore/emulator_utils.py` lines 155-179

**Current implementation:**

```python
pos_evts_frame = torch.floor(
    torch.relu(diff_frame) / pos_thres).to(dtype=torch.int32)
neg_evts_frame = torch.floor(
    torch.relu(-diff_frame) / neg_thres).to(dtype=torch.int32)
```

**CUDA fused kernel:**

```cpp
__global__ void compute_event_map_kernel(
    const float* diff,
    const float* pos_thres,
    const float* neg_thres,
    int32_t* pos_evts,
    int32_t* neg_evts,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float d = diff[idx];
    pos_evts[idx] = (d > 0) ? (int32_t)(d / pos_thres[idx]) : 0;
    neg_evts[idx] = (d < 0) ? (int32_t)(-d / neg_thres[idx]) : 0;
}
```

**Expected speedup**: 1.5-2x (currently 27µs → ~15µs)

---

## 🔧 Medium-Impact Opportunities

### 6. **TorchScript JIT Compilation**

**Target**: Entire `EventEmulator.generate_events()` method

**Approach:**

```python
@torch.jit.script
def generate_events_jit(
    new_frame: torch.Tensor,
    t_frame: float,
    # ... all state as arguments ...
) -> torch.Tensor:
    # Full event generation logic
    pass
```

**Challenges:**

- Requires extensive refactoring (stateful class → pure function)
- Python control flow needs to be TorchScript-compatible
- Debugging is harder

**Expected speedup**: 1.3-1.8x if successfully compiled

---

### 7. **Numba JIT for Shot Noise Generation**

**Target**: `v2ecore/emulator_utils.py` - `generate_shot_noise()`

**Current status**: Already uses Numba `@njit`

**Optimization opportunity:**

```python
@njit(parallel=True, fastmath=True)
def generate_shot_noise(...):
    # Add parallel=True for pixel-parallel sampling
    for i in prange(height):
        for j in range(width):
            # Independent random sampling per pixel
```

**Expected speedup**: 1.5-2x for large resolutions

---

## 🚀 Implementation Priority

### Phase 1: Quick Wins (Low Effort, Medium Gain)

1. **Upgrade TBB** → Enable parallel histogram (4-8x for rendering)
2. **Add `parallel=True` to shot noise** → 1.5-2x for noise generation
3. **Cython for lin_log** → 2-3x for lin_log (if float64 precision validated)

**Estimated total speedup**: 1.3-1.5x end-to-end

---

### Phase 2: Custom CUDA Kernels (High Effort, High Gain)

1. **Fused event extraction kernel** → 3-5x for event loop (**biggest win**)
2. **Fused IIR filter kernel** → 1.5-2x for lowpass
3. **Fused event map kernel** → 1.5-2x for quantization

**Estimated total speedup**: 2-3x end-to-end on top of Phase 1

---

### Phase 3: Advanced (Very High Effort)

1. **TorchScript JIT** → 1.3-1.8x if successful
2. **Full C++ rewrite of emulator core** → 2-4x potential, but high maintenance cost

---

## 📊 Expected Overall Gains

| Phase | Cumulative Speedup | Effort Level | Baseline FPS → New FPS |
|-------|-------------------|--------------|------------------------|
| Baseline | 1.0x | N/A | 97 FPS |
| Phase 1 | **1.3-1.5x** | Low-Medium | 126-146 FPS |
| Phase 1+2 | **2.6-4.5x** | High | 252-437 FPS |
| Phase 1+2+3 | **3.4-8.1x** | Very High | 330-786 FPS |

---

## 🛠️ Implementation Tools

### For CUDA Kernels

1. **PyTorch C++ Extensions** (recommended)
   - Seamless PyTorch integration
   - `torch.utils.cpp_extension.load()`
   - Example: <https://pytorch.org/tutorials/advanced/cpp_extension.html>

2. **CuPy** (alternative)
   - Python-friendly CUDA
   - RawKernels for custom CUDA code
   - Example: <https://docs.cupy.dev/en/stable/user_guide/kernel.html>

### For Cython

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("v2ecore/lin_log_cy.pyx"),
    include_dirs=[np.get_include()]
)
```

### For Numba

- Already integrated (just add `parallel=True` where applicable)

---

## ⚠️ Considerations

### Precision Requirements

- **lin_log** requires float64 (documented in TODO comment)
- Test extensively if attempting float32 conversion

### Maintainability

- CUDA/C++ extensions add build complexity
- Keep Python fallbacks for CPU-only systems
- Extensive testing required for each optimization

### Platform Compatibility

- CUDA kernels require NVIDIA GPU
- Cython requires C compiler
- TBB parallel requires proper threading backend

---

## 🎯 Recommended Next Steps

1. **Immediate** (if you can write C++/CUDA):
   - Implement fused event extraction CUDA kernel (biggest win: 3-5x)
   - Profile with `nvprof` or `nsys` to identify exact kernel bottlenecks

2. **Short-term**:
   - Upgrade TBB for parallel histogram
   - Add Cython version of lin_log (with float64)

3. **Long-term**:
   - Consider PyTorch C++ extension for entire emulator core
   - Explore GPU-native event data structures

---

## 📚 References

- PyTorch CUDA Extensions: <https://pytorch.org/tutorials/advanced/cpp_extension.html>
- CuPy Custom Kernels: <https://docs.cupy.dev/en/stable/user_guide/kernel.html>
- Numba CUDA: <https://numba.readthedocs.io/en/stable/cuda/index.html>
- Cython Parallel: <https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html>
