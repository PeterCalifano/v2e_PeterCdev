# v2e Core Event Model: Math-to-Code Map

This is the single source of truth that maps the core v2e model to searchable
implementation anchors. It intentionally avoids line-number indexes, which
become stale whenever comments or nearby code move.

## References

- Hu, Liu, and Delbruck, *v2e: From Video Frames to Realistic DVS Events*,
  CVPRW 2021, [arXiv:2006.07722](https://arxiv.org/abs/2006.07722).
- Graca and Delbruck, *Unraveling the Paradox of Intensity-Dependent DVS Pixel
  Noise*, 2021, [arXiv:2109.08640](https://arxiv.org/abs/2109.08640).

The optional IEBCS- and V2CE-inspired extensions are mapped separately in
[`README_error_models_extensions.md`](README_error_models_extensions.md).

## Notation

| Symbol | Meaning |
|---|---|
| \(I_k(x,y)\) | Input luma or normalized HDR intensity at frame \(k\) |
| \(L_k\) | Lin-log encoded brightness |
| \(L_{\mathrm{lp},k}\) | Low-pass photoreceptor output |
| \(L_{\mathrm{mem},k}\) | Memorized change-detector reference |
| \(\Delta L_k\) | \(L_{\mathrm{lp},k}-L_{\mathrm{mem},k}\) |
| \(\theta_+(x,y),\theta_-(x,y)\) | Per-pixel ON and OFF thresholds |
| \(\Delta t_k\) | \(t_k-t_{k-1}\) |

## Core Equations

### Lin-log front end

For transition value \(I_0=20\), the implementation uses

$$
L(I)=
\begin{cases}
I\,\dfrac{\log I_0}{I_0}, & I\le I_0,\\[6pt]
\log I, & I>I_0.
\end{cases}
$$

The result is rounded at \(10^{-8}\) in float64 before conversion to float32.
This avoids numerical drift that can suppress expected OFF events.

Search anchors:

- `KEY[D-LINLOG-CALL]` in `v2ecore/emulator.py`
- `KEY[D-LINLOG]` in `v2ecore/emulator_utils.py`

### Finite photoreceptor bandwidth

With cutoff \(f_c\),

$$
\tau=\frac{1}{2\pi f_c},
\qquad
L_{\mathrm{lp},k}
=
L_{\mathrm{lp},k-1}
+\epsilon_k\left(L_k-L_{\mathrm{lp},k-1}\right).
$$

The intensity-dependent path uses

$$
\epsilon_k(x,y)
=
\min\left(I_{01,k}(x,y)\frac{\Delta t_k}{\tau},1\right).
$$

The scalar path currently uses \(\epsilon_k=\Delta t_k/\tau\) without the same
upper clamp. That is an open correctness defect for large
\(\Delta t_k/\tau\), not an intended model difference.

Search anchors:

- `KEY[E-INTEN-SCALE]`, `KEY[E-LPF-CALL]`
- `KEY[E-LPF-EPS]`, `KEY[E-LPF-IIR]`

### Contrast and event quantization

$$
\Delta L_k=L_{\mathrm{lp},k}-L_{\mathrm{mem},k},
$$

$$
n_+(x,y)
=
\left\lfloor
\frac{\max(\Delta L_k(x,y),0)}{\theta_+(x,y)}
\right\rfloor,
\qquad
n_-(x,y)
=
\left\lfloor
\frac{\max(-\Delta L_k(x,y),0)}{\theta_-(x,y)}
\right\rfloor.
$$

Search anchors:

- `KEY[F-DIFF]`
- `KEY[F-EVENT-MAP-CALL]`
- `KEY[F-EVENT-QUANT]`

### Comparator-memory update

After refractory filtering, the emitted signal-event counts
\(\hat n_+\) and \(\hat n_-\) update memory as

$$
L_{\mathrm{mem},k}
=
L_{\mathrm{mem},k-1}
+\hat n_+\theta_+
-\hat n_-\theta_-.
$$

Search anchor: `KEY[F-LMEM-UPDATE]`.

Histogram-driven IEBCS noise is currently appended after signal generation and
does not update this state. That limitation is tracked in the consolidation
plan.

### Leak

Ignoring mismatch notation, the default leak update is

$$
L_{\mathrm{mem}}
\leftarrow
L_{\mathrm{mem}}
-\Delta t\,r_{\mathrm{leak}}\theta_+.
$$

The implementation applies a per-pixel random rate factor.

Search anchors:

- `KEY[F-LEAK-FPN]`, `KEY[F-LEAK-CALL]`
- `KEY[F-LEAK-RATE]`, `KEY[F-LEAK-LMEM]`

### Temporal noise

The repository has two distinct paths:

- The simple shot-noise path uses per-frame Bernoulli ON/OFF draws whose
  probabilities scale with rate, \(\Delta t\), brightness, and the
  nominal-to-actual threshold ratio.
- The photoreceptor-noise path injects Gaussian voltage noise and filters it
  before thresholding, producing temporally correlated events.

Search anchors:

- `KEY[G-THRESH-PROB-SCALE]`
- `KEY[G-PHOTO-NOISE-INJECT]`, `KEY[G-PHOTO-VRMS-FIT]`
- `KEY[G-SHOT-CALL]`, `KEY[G-SHOT-PROB]`, `KEY[G-SHOT-THRESH]`
- `KEY[G-SHOT-RESET]`

### Refractory filtering

The default path applies a practical per-pixel minimum inter-event interval to
already quantized candidates. It is an implementation extension rather than a
central analytical block in the original v2e equations.

Search anchor: `KEY[H-REFRACTORY-EXT]`.

## Timestamp Discretization

For \(N\) global event layers in \((t_{k-1},t_k]\), the default path uses

$$
t_i=t_{k-1}+i\frac{\Delta t_k}{N},
\qquad i=1,\ldots,N.
$$

Every active pixel in layer \(i\) receives the same \(t_i\). The optional V2CE
modes relocate these same global layers; they do not yet infer independent
per-pixel timestamps.

Search anchors:

- `KEY[A-DELTA-T]`
- `KEY[F-TS-SUBDIV]`

## Navigation

Use symbolic anchors instead of copied line numbers:

```bash
rg -n "KEY\\[[^]]+\\]" v2ecore/emulator.py v2ecore/emulator_utils.py
```

Primary files:

- `v2ecore/emulator.py`
- `v2ecore/emulator_utils.py`
