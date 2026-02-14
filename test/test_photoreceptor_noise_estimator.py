import math

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


def test_estimator_returns_finite_positive_value():
    estimator = PhotoreceptorNoiseVoltageEstimator(seed=7)
    vnscaled = estimator(**_Estimator_args())

    assert math.isfinite(vnscaled)
    assert vnscaled > 0


def test_estimator_cache_hit_for_identical_arguments():
    estimator = PhotoreceptorNoiseVoltageEstimator(seed=11)

    first = estimator(**_Estimator_args())
    computation_count = estimator.computation_count
    second = estimator(**_Estimator_args())

    assert computation_count == 1
    assert estimator.computation_count == computation_count
    assert first == second


def test_estimator_cache_miss_when_key_input_changes():
    estimator = PhotoreceptorNoiseVoltageEstimator(seed=13)

    estimator(**_Estimator_args())
    computation_count = estimator.computation_count

    changed_args = _Estimator_args()
    changed_args["pos_thr"] = 0.23
    estimator(**changed_args)

    assert estimator.computation_count == computation_count + 1


def test_estimator_instances_are_state_isolated():
    estimator_a = PhotoreceptorNoiseVoltageEstimator(seed=17)
    estimator_b = PhotoreceptorNoiseVoltageEstimator(seed=17)

    estimator_a(**_Estimator_args())
    estimator_b(**_Estimator_args())

    changed_args = _Estimator_args()
    changed_args["f3db"] = 6.0
    estimator_a(**changed_args)

    assert estimator_a.computation_count == 2
    assert estimator_b.computation_count == 1


def test_legacy_function_wrapper_returns_float():
    vnscaled = compute_photoreceptor_noise_voltage(**_Estimator_args())

    assert isinstance(vnscaled, float)
    assert math.isfinite(vnscaled)
    assert vnscaled > 0
