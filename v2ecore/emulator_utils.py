"""Provide numerical primitives for event-camera emulation.

This module owns lin-log conversion, caller-owned photoreceptor filtering,
event-map quantization, and stochastic sensor-noise helpers. ``LowPassFilter``
updates a supplied state tensor in place so state ownership remains with the
emulator.

Example:
    import torch

    from v2ecore.emulator_utils import LowPassFilter

    target_ = torch.ones((1, 1))
    state_ = torch.zeros_like(target_)
    result_ = LowPassFilter(tau=1.0)(
        target_, state_, inten01=None, delta_time=0.25)
    print(result_)

Output:
    tensor([[0.2500]])

Authors: Pietro Califano <petercalifano.gs@gmail.com>
Originally developed by: Yuhuang Hu <yuhuang.hu@ini.uzh.ch>,
    Tobi Delbruck <tobi@ini.uzh.ch>
"""
import logging
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def Map_linear_to_log_luminance(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # TODO (TBC): Performance optimization — converting to float64 here is
    # expensive (~1.5-2x slower). Using float32 throughout would avoid the
    # double() cast and final .float() cast. However, the rounding logic
    # below relies on float64 precision to prevent numerical drift that
    # suppresses OFF events. Validate against regression tests before changing.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    lin_to_log_scale = (1./threshold) * math.log(threshold)

    # KEY[D-LINLOG]: Fig. 5D in Hu et al. 2021 (v2e paper): piecewise
    # lin-log mapping from luma Y to log-domain brightness L.
    y = torch.where(x <= threshold, x*lin_to_log_scale, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()


def rescale_intensity_frame(new_frame):
    """Rescale intensity frames.

    make sure we get no zero time constants
    limit max time constant to ~1/10 of white intensity level
    """
    return (new_frame + 20) / 275.  # DEVNOTE why +20?


IIR_MAX_WARNINGS = 10
_PHOTO_NOISE_RATE_PER_BW_WARN = 0.5


class LowPassFilter:
    """First-order IIR filter with an explicit in-place state contract.

    The caller owns ``lp_log_frame``. A successful filtered update mutates and
    returns that same tensor; disabled filtering returns ``log_new_frame``
    unchanged. The object retains only filter configuration and warning counts.
    """

    __slots__ = (
        "cutoff_hz",
        "tau",
        "strict_model_validity",
        "iir_warning_count",
        "disabled_warning_count",
    )

    def __init__(self,
                 cutoff_hz: float = 0.0,
                 tau: float | None = None,
                 strict_model_validity: bool = False) -> None:
        """Initialize the default filter time constant.

        Args:
            cutoff_hz: Photoreceptor low-pass cutoff in Hz.
            tau: Explicit time constant in seconds. When present, it takes
                precedence over ``cutoff_hz``.
            strict_model_validity: Raise before an update weight above one
                would be clamped.
        """
        self.cutoff_hz = float(cutoff_hz)
        if tau is None:
            self.tau = (
                1 / (2 * math.pi * self.cutoff_hz)
                if self.cutoff_hz > 0
                else -1.0
            )
        else:
            self.tau = float(tau)
        self.strict_model_validity = bool(strict_model_validity)
        self.iir_warning_count = 0
        self.disabled_warning_count = 0

    def __call__(self,
                 log_new_frame: torch.Tensor,
                 lp_log_frame: torch.Tensor,
                 inten01: torch.Tensor | None,
                 delta_time: float,
                 filter_tau_const: float | None = None,
                 strict_model_validity: bool = False) -> torch.Tensor:
        """Advance the supplied low-pass state by one frame.

        Args:
            log_new_frame: Current lin-log brightness tensor.
            lp_log_frame: Caller-owned filtered state to update in place.
            inten01: Optional per-pixel bandwidth scale.
            delta_time: Time since the previous frame in seconds.
            filter_tau_const: Optional per-call time-constant override.
            strict_model_validity: Tighten this call to strict execution. A
                strictly configured filter cannot be weakened per call.

        Returns:
            The updated ``lp_log_frame`` object, or ``log_new_frame`` when
            filtering is disabled.
        """

        # Resolve the per-call override before the object's configured default.
        if filter_tau_const is not None:
            tau_ = filter_tau_const
        else:
            tau_ = self.tau if self.tau > 0 else (
                1 / (2 * math.pi * self.cutoff_hz) if self.cutoff_hz > 0 else -1.0)

        if tau_ <= 0:
            if self.disabled_warning_count == 0:
                logger.warning(
                    f'cutoff_hz={self.cutoff_hz} is non-positive; '
                    'skipping low-pass filtering')
            self.disabled_warning_count += 1
            return log_new_frame

        delta_over_tau_ = delta_time / tau_

        # KEY[E-LPF-EPS]: brightness scales the update weight when available.
        # Scalar and per-pixel paths share stability and warning behavior.
        eps_: float | torch.Tensor = (
            delta_over_tau_
            if inten01 is None
            else inten01 * delta_over_tau_
        )
        max_eps_ = (
            float(torch.max(eps_).item())
            if isinstance(eps_, torch.Tensor)
            else eps_
        )

        # Strict execution must not silently replace extrapolation with a
        # clamped approximation or mutate the caller-owned state.
        strict_model_validity_ = (
            self.strict_model_validity or strict_model_validity)
        if strict_model_validity_ and max_eps_ > 1.0:
            minimum_sample_rate_hz_ = max_eps_ / delta_time
            raise ValueError(
                'Low-pass update requires clamping with '
                f'strict_model_validity=True: eps={max_eps_:.6g} > 1, '
                f'delta_time={delta_time:.6g}s, tau={tau_:.6g}s, '
                f'cutoff_hz={self.cutoff_hz:.6g}Hz. Increase the input '
                f'sample rate to at least {minimum_sample_rate_hz_:.6g}Hz '
                'or reduce cutoff_hz.')

        if max_eps_ > 0.3 and self.iir_warning_count < IIR_MAX_WARNINGS:
            logger.warning(
                f'IIR lowpass filter update has large maximum update '
                f'eps={max_eps_:.2f} from '
                f'delta_time/tau={delta_time:.3g}/{tau_:.3g}')
            self.iir_warning_count += 1

            if self.iir_warning_count == IIR_MAX_WARNINGS:
                logger.warning(
                    'Suppressing further warnings about inaccurate IIR '
                    'low-pass filtering; check timestamp resolution and DVS '
                    'photoreceptor cutoff frequency')

        # Keep the interpolation stable and make tensor weights compatible
        # with the caller-owned state before the in-place operation.
        if isinstance(eps_, torch.Tensor):
            eps_ = torch.clamp(eps_, max=1.0).to(
                dtype=lp_log_frame.dtype,
                device=lp_log_frame.device,
            )
        else:
            eps_ = min(eps_, 1.0)

        # KEY[E-LPF-IIR]: update the caller-owned photoreceptor state using one
        # explicit in-place contract for scalar and per-pixel weights.
        return lp_log_frame.lerp_(log_new_frame, eps_)


_LOW_PASS_FILTER_CACHE: dict[
    tuple[float, float | None, bool], LowPassFilter] = {}


def apply_low_pass_filter(log_new_frame: torch.Tensor,
                          lp_log_frame: torch.Tensor,
                          inten01: torch.Tensor | None,
                          delta_time: float,
                          cutoff_hz: float = 0.0,
                          filter_tau_const: float | None = None,
                          low_pass_filter: LowPassFilter | None = None,
                          strict_model_validity: bool = False) -> torch.Tensor:
    """Apply a cached or caller-supplied low-pass filter.

    Args:
        log_new_frame: Current lin-log brightness tensor.
        lp_log_frame: Caller-owned filtered state.
        inten01: Optional per-pixel bandwidth scale.
        delta_time: Time since the previous frame in seconds.
        cutoff_hz: Cutoff used when constructing a cached filter.
        filter_tau_const: Optional time-constant override in seconds.
        low_pass_filter: Optional filter instance. The time-constant override
            is forwarded to it rather than silently discarded.
        strict_model_validity: Raise before a low-pass update would be clamped.

    Returns:
        Updated low-pass state according to ``LowPassFilter``'s aliasing
        contract.
    """
    active_filter_ = low_pass_filter
    if active_filter_ is None:
        tau_key_ = (
            None
            if filter_tau_const is None
            else float(filter_tau_const)
        )
        cache_key_ = (
            float(cutoff_hz), tau_key_, bool(strict_model_validity))
        active_filter_ = _LOW_PASS_FILTER_CACHE.get(cache_key_)

        if active_filter_ is None:
            active_filter_ = LowPassFilter(
                cutoff_hz=cutoff_hz,
                tau=filter_tau_const,
                strict_model_validity=strict_model_validity)
            _LOW_PASS_FILTER_CACHE[cache_key_] = active_filter_

    return active_filter_(
        log_new_frame=log_new_frame,
        lp_log_frame=lp_log_frame,
        inten01=inten01,
        delta_time=delta_time,
        filter_tau_const=filter_tau_const,
        strict_model_validity=strict_model_validity)


def subtract_leak_current(base_log_frame: torch.Tensor,
                          leak_rate_hz: float,
                          delta_time: float,
                          pos_thres: float | torch.Tensor,
                          leak_jitter_fraction: float,
                          noise_rate_array: torch.Tensor,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """Subtract stochastic leak current from comparator memory.

    Args:
        base_log_frame: Current per-pixel comparator memory.
        leak_rate_hz: Nominal leak-event rate per pixel.
        delta_time: Elapsed frame interval in seconds.
        pos_thres: Active ON threshold scalar or per-pixel tensor.
        leak_jitter_fraction: Fractional sample-to-sample leak jitter.
        noise_rate_array: Fixed per-pixel leak-rate multipliers.
        generator: Optional caller-owned random generator.

    Returns:
        Updated comparator memory. The input tensor is not modified.
    """

    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32,
        device=noise_rate_array.device, generator=generator)

    # KEY[F-LEAK-RATE]: Sec. 4(F) leak model with per-pixel randomization.
    curr_leak_rate = \
        leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

    # KEY[F-LEAK-LMEM]: Lmem continuously decreases to create spontaneous ON
    # leak events.
    delta_leak = delta_time*curr_leak_rate*pos_thres  # this is a matrix

    # ideal model
    #  delta_leak = delta_time*leak_rate_hz*pos_thres  # this is a matrix

    return base_log_frame-delta_leak


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """
        Compute event maps, i.e. 2d arrays of [width,height] containing quantized number of ON and OFF events.

    Args:
        diff_frame:  the input difference frame between stored log intensity and current frame log intensity [width, height]
        pos_thres:  ON threshold values [width, height]
        neg_thres:  OFF threshold values [width, height]

    Returns:
        pos_evts_frame, neg_evts_frame;  2d Tensors of integer ON and OFF event counts
    """
    # Extract positive and negative differences
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    # Compute quantized number of ON and OFF events for each pixel
    # KEY[F-EVENT-QUANT]: Fig. 5F / Sec. 4 event generation:
    # DeltaL -> integer event count via threshold quantization (elem-wise division with floor rounding).
    pos_evts_frame = torch.div(
        pos_frame, pos_thres, rounding_mode="floor").type(torch.int32)

    neg_evts_frame = torch.div(
        neg_frame, neg_thres, rounding_mode="floor").type(torch.int32)

    #  max_events = max(pos_evts_frame.max(), neg_evts_frame.max())

    #  # boolean array (max_events, height, width)
    #  # positive events and negative
    #  pos_evts_cord = torch.arange(
    #      1, max_events+1, dtype=torch.int32,
    #      device=diff_frame.device).unsqueeze(-1).unsqueeze(-1).repeat(
    #          1, diff_frame.shape[0], diff_frame.shape[1])
    #  neg_evts_cord = pos_evts_cord.clone().detach()
    #
    #  # generate event cords
    #  pos_evts_cord_post = (pos_evts_cord >= pos_evts_frame.unsqueeze(0))
    #  neg_evts_cord_post = (neg_evts_cord >= neg_evts_frame.unsqueeze(0))

    return pos_evts_frame, neg_evts_frame
    #  return pos_evts_cord_post, neg_evts_cord_post, max_events


class PhotoreceptorNoiseVoltageEstimator:
    """Estimate and cache noise RMS for a target photoreceptor event rate.

    The estimator combines the Graca-Delbruck threshold/noise-rate fit with a
    sampled first-order photoreceptor response. Each instance owns its random
    generator and one-entry argument cache.
    """

    __slots__ = (
        "_rng",
        "_last_args",
        "_last_vn",
        "_vrms_computation_printed",
        "_computation_count",
        "sample_rate_rel_tolerance",
        "value_rel_tolerance",
        "value_abs_tolerance",
        "num_threshold_samples",
    )

    def __init__(self,
                 seed: int | None = None,
                 sample_rate_rel_tolerance: float = 0.1,
                 value_rel_tolerance: float = 1e-6,
                 value_abs_tolerance: float = 1e-12,
                 num_threshold_samples: int = 300) -> None:
        """Initialize estimator randomness, cache tolerances, and sample count.

        Args:
            seed: Seed for the estimator-owned NumPy random generator.
            sample_rate_rel_tolerance: Relative sample-rate difference accepted
                by the one-entry cache.
            value_rel_tolerance: Relative tolerance for other cached inputs.
            value_abs_tolerance: Absolute tolerance for all cached inputs.
            num_threshold_samples: Number of ON/OFF threshold pairs sampled
                for the voltage estimate.
        """
        self._rng = np.random.default_rng(seed)
        self._last_args: tuple[float, float, float, float, float, float] | None = None
        self._last_vn: float | None = None
        self._vrms_computation_printed = False
        self._computation_count = 0
        self.sample_rate_rel_tolerance = float(sample_rate_rel_tolerance)
        self.value_rel_tolerance = float(value_rel_tolerance)
        self.value_abs_tolerance = float(value_abs_tolerance)
        self.num_threshold_samples = int(num_threshold_samples)

    @property
    def computation_count(self) -> int:
        """Return the number of non-cached calibration computations."""
        return self._computation_count

    def reset_cache(self) -> None:
        """Discard the cached argument tuple and calibrated voltage."""
        self._last_args = None
        self._last_vn = None

    def _compute_vn_from_log_rate_per_hz(self,
                                         threshold_: float | np.ndarray,
                                         log_rate_per_hz_: float) -> float | np.ndarray:
        """Map event thresholds to required RMS photoreceptor noise.

        The polynomial fits Figure 3 of Graca and Delbruck (2021). Its
        spreadsheet derivation is stored in
        ``media/noise_event_rate_simulation.xlsx``.

        Args:
            threshold_: Scalar or sampled ON/OFF threshold in log units.
            log_rate_per_hz_: Base-10 logarithm of noise rate divided by
                photoreceptor bandwidth.

        Returns:
            Required RMS noise voltage with the same shape as ``threshold_``.
        """
        # y = log10(threshold / Vn), x = log10(Rn / f3db).
        # KEY[G-PHOTO-VRMS-FIT]: fitted relation from Graca & Delbruck 2021
        # used to map target noise-rate-per-bandwidth to RMS noise voltage.
        log_threshold_per_vn_ = (
            -0.0026 * log_rate_per_hz_ ** 3
            - 0.036 * log_rate_per_hz_ ** 2
            - 0.1949 * log_rate_per_hz_
            + 0.321
        )
        threshold_per_vn_ = 10 ** log_threshold_per_vn_
        return threshold_ / threshold_per_vn_

    def _cache_hit(self,
                   shot_noise_rate_hz: float,
                   f3db: float,
                   sample_rate_hz: float,
                   pos_thr: float,
                   neg_thr: float,
                   sigma_thr: float) -> bool:
        
        if self._last_args is None or self._last_vn is None:
            return False
        
        last_shot_noise_rate_hz, last_f3db, last_sample_rate_hz, last_pos_thr, last_neg_thr, last_sigma_thr = self._last_args
        
        if not np.isclose(
                sample_rate_hz, last_sample_rate_hz,
                rtol=self.sample_rate_rel_tolerance,
                atol=self.value_abs_tolerance):
            return False
        
        return (
            np.isclose(shot_noise_rate_hz, last_shot_noise_rate_hz, rtol=self.value_rel_tolerance, atol=self.value_abs_tolerance)
            and np.isclose(f3db, last_f3db, rtol=self.value_rel_tolerance, atol=self.value_abs_tolerance)
            and np.isclose(pos_thr, last_pos_thr, rtol=self.value_rel_tolerance, atol=self.value_abs_tolerance)
            and np.isclose(neg_thr, last_neg_thr, rtol=self.value_rel_tolerance, atol=self.value_abs_tolerance)
            and np.isclose(sigma_thr, last_sigma_thr, rtol=self.value_rel_tolerance, atol=self.value_abs_tolerance)
        )

    def __call__(self,
                 shot_noise_rate_hz: float,
                 f3db: float,
                 sample_rate_hz: float,
                 pos_thr: float,
                 neg_thr: float,
                 sigma_thr: float) -> float:
        """Estimate pre-filter noise RMS for a target event-noise rate.

        Args:
            shot_noise_rate_hz: Desired total pixel shot-noise rate in Hz.
            f3db: First-order photoreceptor cutoff frequency in Hz.
            sample_rate_hz: Upsampled frame rate before low-pass filtering.
            pos_thr: Nominal ON threshold in natural-log units.
            neg_thr: Nominal OFF threshold in natural-log units.
            sigma_thr: Standard deviation of per-pixel thresholds.

        Returns:
            Gaussian RMS in natural-log units to inject before the
            photoreceptor low-pass filter.
        """
        if self._cache_hit(
                shot_noise_rate_hz=shot_noise_rate_hz, f3db=f3db, sample_rate_hz=sample_rate_hz,
                pos_thr=pos_thr, neg_thr=neg_thr, sigma_thr=sigma_thr):
            assert self._last_vn is not None
            return self._last_vn

        # Simulation data are on ON event rates, divide by 2 here to end up with correct total rate
        rate_per_bw = 0.5 * (shot_noise_rate_hz / f3db)
        
        if rate_per_bw > _PHOTO_NOISE_RATE_PER_BW_WARN:
            logger.warning(
                f'Shot noise rate per Hz of bandwidth is larger than '
                f'{_PHOTO_NOISE_RATE_PER_BW_WARN:g} '
                f'(rate_hz={shot_noise_rate_hz} Hz, '
                f'3dB bandwidth={f3db} Hz)')
            
        x = math.log10(rate_per_bw)

        if x < -5.0:
            logger.warning(
                f'Desired noise rate of {shot_noise_rate_hz}Hz is too low to accurately compute a threshold value')
            
        elif x > 0.0:
            logger.warning(
                f'Desired noise rate of {shot_noise_rate_hz}Hz is too large to accurately compute a threshold value')

        # Now we need to numerically estimate the required Vnrms given the thresholds and the sigma thresholds, since the noise rate varies dramatically with threshold
        
        # Sample thresholds from the distribution defined by the nominal threshold and sigma, compute the noise voltage for each sampled threshold, and average to get the final vn
        pos_samps = pos_thr + sigma_thr * self._rng.standard_normal(self.num_threshold_samples)
        neg_samps = neg_thr + sigma_thr * self._rng.standard_normal(self.num_threshold_samples)

        mins = np.minimum(pos_samps, neg_samps)
        
        # Map every sampled threshold through the documented fitted physical
        # relation, then average the required RMS voltage.
        sampled_vn_ = self._compute_vn_from_log_rate_per_hz(mins, x)
        vn = float(np.mean(sampled_vn_))

        # Now we need to find the scaling factor from white noise. To get the correct noise vn after RC lowpass to get this NEB factor, we generate white samples here, lowpass filter them the same exact way as we do in the emulator (i.e. with same IIR time constant and sample rate). Compute the variance, and scale the amplitude to give us vn
        tau = 1/(f3db*2*math.pi)

        dt = 1/sample_rate_hz
        t = np.arange(0, 1000*tau, dt)
        
        # Generated Gaussian random sequence with amplitude vn RMS
        
        rin = vn * self._rng.standard_normal(t.shape)
        rms_in = np.std(rin)  # Check the RMS, should be vn
        rout = np.zeros_like(rin)

        # RC lowpass the noise
        eps = dt/tau
        eps_limit = .1

        if eps > eps_limit:
            logger.warning(f'\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce timestep (currently {dt:.3f}s) (using higher frame rate) or decrease cutuff_hz (currently {f3db:.3f} Hz)'
                           f'\n\tExpect the generated shot noise rate to be significantly lower than the desired rate.'
                           f'\n\tConsider not using --photoreceptor_noise option if you only want simple Poisson shot noise without temporal correlation of lowpass filtering and ON/OFF events.')
        rout[0] = 0  # init value is mean 0
        # lp filter the sequence with same tau and dt as v2e
        for i in range(1, len(rin)):
            rout[i] = rout[i-1]*(1-eps) + rin[i]*eps
        rms_out = np.std(rout)  # compute the amplitude of this noise
        scale = rms_in / rms_out

        # divide the computed vn to get the necessary vn to add before RC lowpass filtering
        vnscaled = float(scale * vn)

        self._last_vn = vnscaled
        self._last_args = (
            float(shot_noise_rate_hz),
            float(f3db),
            float(sample_rate_hz),
            float(pos_thr),
            float(neg_thr),
            float(sigma_thr),
        )
        self._computation_count += 1

        if not self._vrms_computation_printed:
            logger.info(
                f'For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units,'
                f' scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, '
                f'sample interval dt={dt*1000:.3f} ms,'
                f', cutoff_hz={f3db} Hz, tau={tau*1000:.3f} ms,  Rn/f3dB={rate_per_bw:.3g} Hz, '
                f' and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units.'
                # f' The sample lowpass filtered has RMS amplitude {stdout:.3f}.'
            )
            self._vrms_computation_printed = True
        return vnscaled


_DEFAULT_PHOTORECEPTOR_NOISE_ESTIMATOR = PhotoreceptorNoiseVoltageEstimator()


def compute_photoreceptor_noise_voltage(shot_noise_rate_hz: float,
                                        f3db: float,
                                        sample_rate_hz: float,
                                        pos_thr: float,
                                        neg_thr: float,
                                        sigma_thr: float) -> float:
    """Estimate photoreceptor-noise voltage through the shared estimator.

    Args:
        shot_noise_rate_hz: Desired total pixel shot-noise rate in Hz.
        f3db: First-order photoreceptor cutoff frequency in Hz.
        sample_rate_hz: Upsampled frame rate before low-pass filtering.
        pos_thr: Nominal ON threshold in natural-log units.
        neg_thr: Nominal OFF threshold in natural-log units.
        sigma_thr: Standard deviation of per-pixel thresholds.

    Returns:
        Gaussian RMS in natural-log units to inject before filtering.
    """
    
    return _DEFAULT_PHOTORECEPTOR_NOISE_ESTIMATOR(
        shot_noise_rate_hz=shot_noise_rate_hz,
        f3db=f3db,
        sample_rate_hz=sample_rate_hz,
        pos_thr=pos_thr,
        neg_thr=neg_thr,
        sigma_thr=sigma_thr,
    )


def generate_shot_noise(shot_noise_rate_hz: float,
                        delta_time: float,
                        shot_noise_inten_factor: float,
                        inten01: torch.Tensor,
                        pos_thres_pre_prob: torch.Tensor,
                        neg_thres_pre_prob: torch.Tensor,
                        generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample simplified ON/OFF temporal shot-noise masks.

    Args:
        shot_noise_rate_hz: Total per-pixel temporal noise rate in Hz.
        delta_time: Elapsed frame interval in seconds.
        shot_noise_inten_factor: Brightness-dependent noise-rate scale.
        inten01: Per-pixel intensity normalized to [0, 1].
        pos_thres_pre_prob: ON probability scale nominal / actual.
        neg_thres_pre_prob: OFF probability scale nominal / actual.
        generator: Optional caller-owned random generator.

    Returns:
        Boolean ON and OFF event masks with the same shape as inten01.
    """
    # new shot noise generator, generate for the entire batch of iterations over this frame

    if shot_noise_rate_hz*delta_time > 1:
        logger.warning(
            f'shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz*delta_time:.2f} is too large, decrease timestamp resolution or sample rate')

    # shot noise factor is the probability of generating an OFF event in this frame (which is tiny typically)
    # we compute it by taking half the total shot noise rate (OFF only),
    # multiplying by the delta time of this frame,
    # and multiplying by the intensity factor
    # division by num_iter is correct if generate_shot_noise is called outside the iteration loop, unless num_iter=1 for calling outside loop
    # KEY[G-SHOT-PROB]: Sec. 4(G) temporal noise model (Poisson-style per
    # sample probabilities scaled by delta time and brightness).
    shot_noise_factor = (
        (0.5 * shot_noise_rate_hz) * delta_time) * \
        ((shot_noise_inten_factor-1)*inten01+1)  # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1 # TODO check this logic again, the shot noise rate should increase with intensity but factor is negative here

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    # KEY[G-SHOT-THRESH]: compare uniform random samples against ON/OFF
    # thresholds to emit temporal noise events.
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob  # ON shot events are generated when uniform sampled random number from range 0-1 is larger than this; the larger shot_noise_factor, the larger the noise rate
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob  # OFF shot events when 0-1 sample less than this

    # for shot noise generate rands from 0-1 for each pixel
    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32,
        device=inten01.device,
        generator=generator)  # draw_frame samples

    # precompute all the shot noise cords, gets binary array size of chip
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample)
    
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample)

    return shot_on_cord, shot_off_cord

    # old shot noise, generate at every iteration.
    # the right device
    #  device = base_log_frame.device

    # array with True where ON noise event
    #  shot_ON_cord = rand01 > (1-shot_ON_prob_this_sample)
    #
    #  shot_OFF_cord = rand01 < shot_OFF_prob_this_sample

    # get shot noise event ON and OFF cordinates
    #  shot_ON_xy = shot_ON_cord.nonzero(as_tuple=True)
    #  shot_ON_count = shot_ON_xy[0].shape[0]
    #
    #  shot_OFF_xy = shot_OFF_cord.nonzero(as_tuple=True)
    #  shot_OFF_count = shot_OFF_xy[0].shape[0]

    #  self.num_events_on += shotOnCount
    #  self.num_events_off += shotOffCount
    #  self.num_events_total += shotOnCount+shotOffCount

    # update log_frame
    #  base_log_frame += shot_ON_cord*pos_thres
    #  base_log_frame -= shot_OFF_cord*neg_thres

    #  if shot_ON_count > 0:
    #      shot_ON_events = torch.ones(
    #          (shot_ON_count, 4), dtype=torch.float32, device=device)
    #      shot_ON_events[:, 0] *= ts
    #      shot_ON_events[:, 1] = shot_ON_xy[1]
    #      shot_ON_events[:, 2] = shot_ON_xy[0]
    #
    #      base_log_frame += shot_ON_cord*pos_thres
    #  else:
    #      shot_ON_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    #
    #  if shot_OFF_count > 0:
    #      shot_OFF_events = torch.ones(
    #          (shot_OFF_count, 4), dtype=torch.float32, device=device)
    #      shot_OFF_events[:, 0] *= ts
    #      shot_OFF_events[:, 1] = shot_OFF_xy[1]
    #      shot_OFF_events[:, 2] = shot_OFF_xy[0]
    #      shot_OFF_events[:, 3] *= -1
    #
    #      base_log_frame -= shot_OFF_cord*neg_thres
    #  else:
    #      shot_OFF_events = torch.zeros(
    #          (0, 4), dtype=torch.float32, device=device)
    # end temporal noise

    #  return shot_ON_events, shot_OFF_events, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord, base_log_frame
    #  return shot_ON_cord, shot_OFF_cord


if __name__ == "__main__":
    # DEVNOTE what is this?
    temp_input = torch.randint(0, 256, (1280, 720), dtype=torch.float32).cuda()

    for i in range(1000):
        temp_out = Map_linear_to_log_luminance(temp_input, threshold=20)

    pass
