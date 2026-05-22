# v2e Core Event Model: Math-to-Code Map

This document maps the event-generation model from the v2e paper to the
implementation in `v2ecore/emulator.py` and `v2ecore/emulator_utils.py`.

The symbolic anchor IDs below are documentation identifiers used to keep the
mapping readable. In this branch they are tracked in docs, not as literal
`#KEY[...]` comments embedded in source.

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

```text
1. Time discretization
   dt = t_k - t_{k-1}

2. Lin-log encoding
   L = f(I)

3. Photoreceptor low-pass
   tau = 1 / (2 * pi * f_c)
   epsilon = min(inten01 * dt / tau, 1)      when intensity scaling is active
   epsilon = dt / tau                        otherwise
   L_lp,new = L_lp + epsilon * (L - L_lp)

4. Event quantization from contrast
   k_on  = floor(max( DeltaL, 0) / theta_on)
   k_off = floor(max(-DeltaL, 0) / theta_off)

5. Memory update (reset-by-threshold increments)
   L_mem,new = L_mem + k_on * theta_on - k_off * theta_off

6. Leak term
   L_mem,new = L_mem - dt * r_leak * theta_on

7. Temporal shot noise
   Bernoulli ON/OFF draws with probabilities proportional to shot_rate * dt
   and scaled by the nominal-to-actual threshold ratio.
```

## Stage-by-stage mapping (paper blocks -> code)

### A. Discrete update interval

- Meaning: all dynamics/noise are integrated over `dt`.
- Tags:
  - `#KEY[A-DELTA-T]`:
    [`v2ecore/emulator.py#L705`](../v2ecore/emulator.py#L705)

### C. Threshold mismatch (pixel-to-pixel)

- Meaning: ON/OFF thresholds are random per pixel (Gaussian mismatch), then
  clamped positive for stability.
- Tags:
  - `#KEY[C-THRESH-MISMATCH]`:
    [`v2ecore/emulator.py#L502`](../v2ecore/emulator.py#L502)

### D. Lin-log front-end

- Meaning: convert frame intensities to log-like domain before contrast
  differencing.
- Tags:
  - `#KEY[D-LINLOG-CALL]`:
    [`v2ecore/emulator.py#L716`](../v2ecore/emulator.py#L716)
  - `#KEY[D-LINLOG]`:
    [`v2ecore/emulator_utils.py#L35`](../v2ecore/emulator_utils.py#L35)

### E. Finite photoreceptor bandwidth

- Meaning: intensity-dependent IIR low-pass on log intensity.
- Tags:
  - `#KEY[E-INTEN-SCALE]`:
    [`v2ecore/emulator.py#L726`](../v2ecore/emulator.py#L726)
  - `#KEY[E-LPF-CALL]`:
    [`v2ecore/emulator.py#L741`](../v2ecore/emulator.py#L741)
  - `#KEY[E-LPF-TAU]`:
    [`v2ecore/emulator_utils.py#L82`](../v2ecore/emulator_utils.py#L82)
  - `#KEY[E-LPF-EPS]`:
    [`v2ecore/emulator_utils.py#L89`](../v2ecore/emulator_utils.py#L89)
  - `#KEY[E-LPF-IIR]`:
    [`v2ecore/emulator_utils.py#L106`](../v2ecore/emulator_utils.py#L106)

### F. Event generation core (DeltaL, quantization, reset, leak)

- Meaning: compare filtered signal against memorized state, quantize to event
  counts, emit events, and update/reset memory.
- Tags:
  - `#KEY[F-LEAK-FPN]`:
    [`v2ecore/emulator.py#L548`](../v2ecore/emulator.py#L548)
  - `#KEY[F-LEAK-CALL]`:
    [`v2ecore/emulator.py#L792`](../v2ecore/emulator.py#L792)
  - `#KEY[F-DIFF]`:
    [`v2ecore/emulator.py#L809`](../v2ecore/emulator.py#L809)
  - `#KEY[F-EVENT-MAP-CALL]`:
    [`v2ecore/emulator.py#L831`](../v2ecore/emulator.py#L831)
  - `#KEY[F-TS-SUBDIV]`:
    [`v2ecore/emulator.py#L853`](../v2ecore/emulator.py#L853)
  - `#KEY[F-LMEM-UPDATE]`:
    [`v2ecore/emulator.py#L1012`](../v2ecore/emulator.py#L1012)
  - `#KEY[F-LEAK-RATE]`:
    [`v2ecore/emulator_utils.py#L133`](../v2ecore/emulator_utils.py#L133)
  - `#KEY[F-LEAK-LMEM]`:
    [`v2ecore/emulator_utils.py#L137`](../v2ecore/emulator_utils.py#L137)
  - `#KEY[F-EVENT-QUANT]`:
    [`v2ecore/emulator_utils.py#L164`](../v2ecore/emulator_utils.py#L164)

### G. Temporal noise models

- Meaning: two paths are implemented.
- `photoreceptor_noise` path: add Gaussian noise in photoreceptor branch and
  low-pass it; this yields temporally correlated noise events.
- simple shot-noise path: direct Bernoulli ON/OFF sampling with probability
  terms using `shot_rate`, `dt`, and threshold scaling.
- Tags:
  - `#KEY[G-THRESH-PROB-SCALE]`:
    [`v2ecore/emulator.py#L520`](../v2ecore/emulator.py#L520)
  - `#KEY[G-PHOTO-NOISE-INJECT]`:
    [`v2ecore/emulator.py#L749`](../v2ecore/emulator.py#L749)
  - `#KEY[G-SHOT-CALL]`:
    [`v2ecore/emulator.py#L966`](../v2ecore/emulator.py#L966)
  - `#KEY[G-SHOT-RESET]`:
    [`v2ecore/emulator.py#L1019`](../v2ecore/emulator.py#L1019)
  - `#KEY[G-PHOTO-VRMS-FIT]`:
    [`v2ecore/emulator_utils.py#L223`](../v2ecore/emulator_utils.py#L223)
  - `#KEY[G-SHOT-PROB]`:
    [`v2ecore/emulator_utils.py#L340`](../v2ecore/emulator_utils.py#L340)
  - `#KEY[G-SHOT-THRESH]`:
    [`v2ecore/emulator_utils.py#L350`](../v2ecore/emulator_utils.py#L350)

### H. Practical extension: refractory gate

- Meaning: per-pixel minimum inter-spike interval filter after candidate events
  are formed.
- This is an implementation extension used for realism/control; it is not the
  central analytical block in the paper's core equation set.
- Tag:
  - `#KEY[H-REFRACTORY-EXT]`:
    [`v2ecore/emulator.py#L894`](../v2ecore/emulator.py#L894)

## Important implementation notes vs paper idealization

- Multi-event timestamps in one frame are linearly subdivided
  (`#KEY[F-TS-SUBDIV]`). This is a practical discretization choice.
- Thresholds are clamped to avoid near-zero values that destabilize rates.
- Simple shot-noise mode and photoreceptor-noise mode are two different noise
  realizations; only one is active at a time.
- `L_mem` update is based on emitted events (including refractory filtering),
  which preserves reset consistency with actual output stream.

## Quick navigation

- Primary entry point:
  [`v2ecore/emulator.py`](../v2ecore/emulator.py)
- Math kernels:
  [`v2ecore/emulator_utils.py`](../v2ecore/emulator_utils.py)
- Useful symbol search:
  `rg -n "generate_events|Map_linear_to_log_luminance|LowPassFilter|compute_event_map" v2ecore/emulator.py v2ecore/emulator_utils.py`
