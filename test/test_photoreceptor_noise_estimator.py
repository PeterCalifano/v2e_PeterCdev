"""Validate photoreceptor-noise calibration and cache behavior.

Example:
    conda run -n v2e python -m pytest -q test/test_photoreceptor_noise_estimator.py

Output:
    All estimator contract tests pass.
"""
import logging
import math

import numpy as np
import pytest

from v2ecore.emulator_utils import PhotoreceptorNoiseVoltageEstimator
from v2ecore.emulator_utils import compute_photoreceptor_noise_voltage


def _Estimator_args() -> dict[str, float]:
    return {
        "shot_noise_rate_hz": 0.02,
        "f3db": 5.0,
        "sample_rate_hz": 120.0,
        "pos_thr": 0.2,
        "neg_thr": 0.2,
        "sigma_thr": 0.03,
    }


def test_estimator_returns_finite_positive_value() -> None:
    estimator_ = PhotoreceptorNoiseVoltageEstimator(seed=7)
    vnscaled_ = estimator_(**_Estimator_args())

    assert math.isfinite(vnscaled_)
    assert vnscaled_ > 0


def test_estimator_cache_hit_for_identical_arguments() -> None:
    estimator_ = PhotoreceptorNoiseVoltageEstimator(seed=11)

    first_ = estimator_(**_Estimator_args())
    computation_count_ = estimator_.computation_count
    second_ = estimator_(**_Estimator_args())

    assert computation_count_ == 1
    assert estimator_.computation_count == computation_count_
    assert first_ == second_


def test_estimator_cache_miss_when_key_input_changes() -> None:
    estimator_ = PhotoreceptorNoiseVoltageEstimator(seed=13)

    estimator_(**_Estimator_args())
    computation_count_ = estimator_.computation_count

    changed_args_ = _Estimator_args()
    changed_args_["pos_thr"] = 0.23
    estimator_(**changed_args_)

    assert estimator_.computation_count == computation_count_ + 1


def test_estimator_instances_are_state_isolated() -> None:
    estimator_a_ = PhotoreceptorNoiseVoltageEstimator(seed=17)
    estimator_b_ = PhotoreceptorNoiseVoltageEstimator(seed=17)

    estimator_a_(**_Estimator_args())
    estimator_b_(**_Estimator_args())

    changed_args_ = _Estimator_args()
    changed_args_["f3db"] = 6.0
    estimator_a_(**changed_args_)

    assert estimator_a_.computation_count == 2
    assert estimator_b_.computation_count == 1


def test_legacy_function_wrapper_returns_float() -> None:
    vnscaled_ = compute_photoreceptor_noise_voltage(**_Estimator_args())

    assert isinstance(vnscaled_, float)
    assert math.isfinite(vnscaled_)
    assert vnscaled_ > 0


def test_fit_maps_threshold_array_to_reference_noise_voltage() -> None:
    """The documented Graca-Delbruck fit supports sampled thresholds."""
    estimator_ = PhotoreceptorNoiseVoltageEstimator(seed=19)
    thresholds_ = np.array([0.1, 0.2], dtype=np.float64)
    expected_vn_ = np.array([
        0.0258463963576984,
        0.0516927927153969,
    ])

    actual_vn_ = estimator_._compute_vn_from_log_rate_per_hz(thresholds_, -2.0)

    np.testing.assert_allclose(actual_vn_, expected_vn_, rtol=1e-14, atol=0.0)


def test_warning_reports_enforced_rate_boundary(caplog: pytest.LogCaptureFixture) -> None:
    """The diagnostic reports the same boundary used by its condition."""
    estimator_ = PhotoreceptorNoiseVoltageEstimator(
        seed=23, num_threshold_samples=8)
    args_ = _Estimator_args()
    args_["shot_noise_rate_hz"] = 6.0

    with caplog.at_level(logging.WARNING, logger="v2ecore.emulator_utils"):
        estimator_(**args_)

    assert "larger than 0.5" in caplog.text
    assert "larger than 0.1" not in caplog.text
