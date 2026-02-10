# v2e Core Event Model: Math-to-Code Map

This document maps the event-generation model from the v2e paper to the
implementation in `v2ecore/emulator.py` and `v2ecore/emulator_utils.py`.

## Primary references

- Hu, Y., Liu, S.-C., Delbruck, T. (2021). *v2e: From Video Frames to Realistic
  DVS Events* (CVPRW). arXiv: [2006.07722](https://arxiv.org/abs/2006.07722).
- Graca, R., Delbruck, T. (2021). *Unraveling the Paradox of
  Intensity-Dependent DVS Pixel Noise*. arXiv:
  [2109.08640](http://arxiv.org/abs/2109.08640).

## Notation used here

- `I`: input brightness (linear intensity / luma).
- `L`: log-domain brightness.
- `L_lp`: low-pass filtered log brightness (photoreceptor output).
- `L_mem`: memorized brightness at event comparator/reset node.
- `DeltaL = L_lp - L_mem`: contrast error driving event generation.
- `theta+`, `theta-`: ON/OFF thresholds.
- `dt`: frame interval.

## Model equations and where they live

1. Time discretization:
   `dt = t_k - t_{k-1}`.
2. Lin-log encoding:
   `L = f(I)` with piecewise linear/log mapping.
3. Photoreceptor low-pass:
   `tau = 1 / (2*pi*f_c)`,
   `epsilon ~ dt/tau` (scaled by intensity),
   `L_lp <- (1-epsilon)L_lp + epsilon L`.
4. Event quantization from contrast:
   `k+ = floor(max(DeltaL, 0)/theta+)`,
   `k- = floor(max(-DeltaL, 0)/theta-)`.
5. Memory update (reset-by-threshold increments):
   `L_mem <- L_mem + k+*theta+ - k-*theta-`.
6. Leak term (noise-driven drift):
   `L_mem <- L_mem - dt * r_leak * theta+`.
7. Temporal shot noise:
   per-pixel Bernoulli draws with probabilities proportional to
   `shot_rate * dt` and threshold-dependent scaling.

## Stage-by-stage mapping (paper blocks -> code)

### A. Discrete update interval

- Meaning: all dynamics/noise are integrated over `dt`.
- Tags:
  - `#CORE[A-DELTA-T]`:
    [`v2ecore/emulator.py#L705`](../v2ecore/emulator.py#L705)

### C. Threshold mismatch (pixel-to-pixel)

- Meaning: ON/OFF thresholds are random per pixel (Gaussian mismatch), then
  clamped positive for stability.
- Tags:
  - `#CORE[C-THRESH-MISMATCH]`:
    [`v2ecore/emulator.py#L502`](../v2ecore/emulator.py#L502)

### D. Lin-log front-end

- Meaning: convert frame intensities to log-like domain before contrast
  differencing.
- Tags:
  - `#CORE[D-LINLOG-CALL]`:
    [`v2ecore/emulator.py#L716`](../v2ecore/emulator.py#L716)
  - `#CORE[D-LINLOG]`:
    [`v2ecore/emulator_utils.py#L35`](../v2ecore/emulator_utils.py#L35)

### E. Finite photoreceptor bandwidth

- Meaning: intensity-dependent IIR low-pass on log intensity.
- Tags:
  - `#CORE[E-INTEN-SCALE]`:
    [`v2ecore/emulator.py#L726`](../v2ecore/emulator.py#L726)
  - `#CORE[E-LPF-CALL]`:
    [`v2ecore/emulator.py#L741`](../v2ecore/emulator.py#L741)
  - `#CORE[E-LPF-TAU]`:
    [`v2ecore/emulator_utils.py#L82`](../v2ecore/emulator_utils.py#L82)
  - `#CORE[E-LPF-EPS]`:
    [`v2ecore/emulator_utils.py#L89`](../v2ecore/emulator_utils.py#L89)
  - `#CORE[E-LPF-IIR]`:
    [`v2ecore/emulator_utils.py#L106`](../v2ecore/emulator_utils.py#L106)

### F. Event generation core (DeltaL, quantization, reset, leak)

- Meaning: compare filtered signal against memorized state, quantize to event
  counts, emit events, and update/reset memory.
- Tags:
  - `#CORE[F-LEAK-FPN]`:
    [`v2ecore/emulator.py#L548`](../v2ecore/emulator.py#L548)
  - `#CORE[F-LEAK-CALL]`:
    [`v2ecore/emulator.py#L792`](../v2ecore/emulator.py#L792)
  - `#CORE[F-DIFF]`:
    [`v2ecore/emulator.py#L809`](../v2ecore/emulator.py#L809)
  - `#CORE[F-EVENT-MAP-CALL]`:
    [`v2ecore/emulator.py#L831`](../v2ecore/emulator.py#L831)
  - `#CORE[F-TS-SUBDIV]`:
    [`v2ecore/emulator.py#L853`](../v2ecore/emulator.py#L853)
  - `#CORE[F-LMEM-UPDATE]`:
    [`v2ecore/emulator.py#L1012`](../v2ecore/emulator.py#L1012)
  - `#CORE[F-LEAK-RATE]`:
    [`v2ecore/emulator_utils.py#L133`](../v2ecore/emulator_utils.py#L133)
  - `#CORE[F-LEAK-LMEM]`:
    [`v2ecore/emulator_utils.py#L137`](../v2ecore/emulator_utils.py#L137)
  - `#CORE[F-EVENT-QUANT]`:
    [`v2ecore/emulator_utils.py#L164`](../v2ecore/emulator_utils.py#L164)

### G. Temporal noise models

- Meaning: two paths are implemented.
- `photoreceptor_noise` path: add Gaussian noise in photoreceptor branch and
  low-pass it; this yields temporally correlated noise events.
- simple shot-noise path: direct Bernoulli ON/OFF sampling with probability
  terms using `shot_rate`, `dt`, and threshold scaling.
- Tags:
  - `#CORE[G-THRESH-PROB-SCALE]`:
    [`v2ecore/emulator.py#L520`](../v2ecore/emulator.py#L520)
  - `#CORE[G-PHOTO-NOISE-INJECT]`:
    [`v2ecore/emulator.py#L749`](../v2ecore/emulator.py#L749)
  - `#CORE[G-SHOT-CALL]`:
    [`v2ecore/emulator.py#L966`](../v2ecore/emulator.py#L966)
  - `#CORE[G-SHOT-RESET]`:
    [`v2ecore/emulator.py#L1019`](../v2ecore/emulator.py#L1019)
  - `#CORE[G-PHOTO-VRMS-FIT]`:
    [`v2ecore/emulator_utils.py#L223`](../v2ecore/emulator_utils.py#L223)
  - `#CORE[G-SHOT-PROB]`:
    [`v2ecore/emulator_utils.py#L340`](../v2ecore/emulator_utils.py#L340)
  - `#CORE[G-SHOT-THRESH]`:
    [`v2ecore/emulator_utils.py#L350`](../v2ecore/emulator_utils.py#L350)

### H. Practical extension: refractory gate

- Meaning: per-pixel minimum inter-spike interval filter after candidate events
  are formed.
- This is an implementation extension used for realism/control; it is not the
  central analytical block in the paper's core equation set.
- Tag:
  - `#CORE[H-REFRACTORY-EXT]`:
    [`v2ecore/emulator.py#L894`](../v2ecore/emulator.py#L894)

## Important implementation notes vs paper idealization

- Multi-event timestamps in one frame are linearly subdivided
  (`#CORE[F-TS-SUBDIV]`). This is a practical discretization choice.
- Thresholds are clamped to avoid near-zero values that destabilize rates.
- Simple shot-noise mode and photoreceptor-noise mode are two different noise
  realizations; only one is active at a time.
- `L_mem` update is based on emitted events (including refractory filtering),
  which preserves reset consistency with actual output stream.

## Quick navigation

- Find all core tags:
  `rg -n "#CORE\\[" v2ecore/emulator.py v2ecore/emulator_utils.py`
- Primary entry point:
  [`v2ecore/emulator.py`](../v2ecore/emulator.py)
- Math kernels:
  [`v2ecore/emulator_utils.py`](../v2ecore/emulator_utils.py)
