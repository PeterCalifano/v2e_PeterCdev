"""Validate finite model-option contracts across public entry points.

Example:
    conda run -n v2e python -m pytest -q test/test_model_options.py

Output:
    6 passed
"""

import argparse
import importlib
import importlib.util
from enum import StrEnum
import json
from pathlib import Path
from types import ModuleType

import pytest


def _Load_cli_module() -> ModuleType:
    """Load the repository CLI module without executing its entry point."""
    cli_path_ = Path(__file__).resolve().parents[1] / "v2e.py"
    spec_ = importlib.util.spec_from_file_location(
        "v2e_model_options_under_test", cli_path_)
    assert spec_ is not None and spec_.loader is not None
    module_ = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module_)
    return module_


def test_option_values_are_native_string_enums() -> None:
    """Finite options serialize as their unchanged public CLI strings."""
    options_module_ = importlib.import_module("v2ecore.model_options")

    assert issubclass(options_module_.V2ceBurstTimestampMode, StrEnum)
    assert issubclass(options_module_.DvsParamPreset, StrEnum)
    assert issubclass(options_module_.IebcsNoiseSource, StrEnum)
    assert issubclass(options_module_.IebcsNoisePreset, StrEnum)
    assert issubclass(options_module_.RendererColorMode, StrEnum)
    assert json.dumps(
        {
            "v2ce": options_module_.V2ceBurstTimestampMode.SLOPE,
            "preset": options_module_.IebcsNoisePreset.LUX_161,
            "color": options_module_.RendererColorMode.GREEN_RED,
        },
        sort_keys=True,
    ) == '{"color": "green_red", "preset": "161lux", "v2ce": "slope"}'


def test_cli_parses_unchanged_option_strings_to_enums() -> None:
    """Argparse keeps public spellings while returning typed values."""
    options_module_ = importlib.import_module("v2ecore.model_options")
    args_module_ = importlib.import_module("v2ecore.v2e_args")
    parser_ = argparse.ArgumentParser(
        formatter_class=args_module_.SmartFormatter)
    parser_ = args_module_.v2e_args(parser_)

    args_ = parser_.parse_args([
        "--dvs_params", "clean",
        "--iebcs_noise_source", "files",
        "--iebcs_noise_preset", "3klux",
        "--v2ce_burst_timestamps_mode", "slope",
        "--dvs_vid_color_mode", "grayscale",
    ])

    assert args_.dvs_params is options_module_.DvsParamPreset.CLEAN
    assert args_.iebcs_noise_source is options_module_.IebcsNoiseSource.FILES
    assert args_.iebcs_noise_preset is options_module_.IebcsNoisePreset.LUX_3K
    assert (
        args_.v2ce_burst_timestamps_mode
        is options_module_.V2ceBurstTimestampMode.SLOPE
    )
    assert args_.dvs_vid_color_mode is options_module_.RendererColorMode.GRAYSCALE


def test_emulator_normalizes_enum_and_legacy_string_options() -> None:
    """Direct API inputs share one typed internal representation."""
    options_module_ = importlib.import_module("v2ecore.model_options")
    emulator_module_ = importlib.import_module("v2ecore.emulator")
    common_kwargs_ = {
        "output_width": 2,
        "output_height": 2,
        "device": "cpu",
    }
    enum_emulator_ = emulator_module_.EventEmulator(
        v2ce_burst_timestamps_mode=(
            options_module_.V2ceBurstTimestampMode.SLOPE),
        **common_kwargs_,
    )
    string_emulator_ = emulator_module_.EventEmulator(
        v2ce_burst_timestamps_mode="slope",
        **common_kwargs_,
    )

    try:
        enum_emulator_.set_dvs_params(options_module_.DvsParamPreset.CLEAN)
        string_emulator_.set_dvs_params("clean")

        assert (
            enum_emulator_.v2ce_burst_timestamps_mode
            is options_module_.V2ceBurstTimestampMode.SLOPE
        )
        assert (
            string_emulator_.v2ce_burst_timestamps_mode
            is options_module_.V2ceBurstTimestampMode.SLOPE
        )
        assert enum_emulator_.cutoff_hz == string_emulator_.cutoff_hz == 0
        assert enum_emulator_.shot_noise_rate_hz == 0
        assert string_emulator_.shot_noise_rate_hz == 0
    finally:
        enum_emulator_.cleanup()
        string_emulator_.cleanup()


def test_renderer_normalizes_color_mode_at_construction() -> None:
    """Renderer branches operate on one color-mode representation."""
    options_module_ = importlib.import_module("v2ecore.model_options")
    renderer_module_ = importlib.import_module("v2ecore.renderer")
    string_renderer_ = renderer_module_.EventRenderer(color_mode="green_red")
    enum_renderer_ = renderer_module_.EventRenderer(
        color_mode=options_module_.RendererColorMode.GRAYSCALE)

    try:
        assert (
            string_renderer_.color_mode
            is options_module_.RendererColorMode.GREEN_RED
        )
        assert (
            enum_renderer_.color_mode
            is options_module_.RendererColorMode.GRAYSCALE
        )
    finally:
        string_renderer_.cleanup()
        enum_renderer_.cleanup()

    with pytest.raises(ValueError):
        renderer_module_.EventRenderer(color_mode="unsupported")


def test_iebcs_noise_resolver_accepts_enums_and_rejects_unknown_sources() -> None:
    """Preset resolution has one finite source and preset domain."""
    options_module_ = importlib.import_module("v2ecore.model_options")
    cli_module_ = _Load_cli_module()

    enum_paths_ = cli_module_.resolve_iebcs_noise_paths(
        noise_source=options_module_.IebcsNoiseSource.PRESET,
        noise_preset=options_module_.IebcsNoisePreset.LUX_161,
        noise_pos_path=None,
        noise_neg_path=None,
    )
    string_paths_ = cli_module_.resolve_iebcs_noise_paths(
        noise_source="preset",
        noise_preset="161lux",
        noise_pos_path=None,
        noise_neg_path=None,
    )

    assert enum_paths_ == string_paths_
    assert enum_paths_[0].endswith("input/iebcs_noise/noise_pos_161lux.npy")
    assert enum_paths_[1].endswith("input/iebcs_noise/noise_neg_161lux.npy")
    with pytest.raises(ValueError):
        cli_module_.resolve_iebcs_noise_paths(
            noise_source="unsupported",
            noise_preset="161lux",
            noise_pos_path=None,
            noise_neg_path=None,
        )


def test_benchmark_profiles_use_the_emulator_option_domain() -> None:
    """Benchmark profiles cannot drift to undeclared V2CE mode strings."""
    options_module_ = importlib.import_module("v2ecore.model_options")
    benchmark_module_ = importlib.import_module(
        "v2ecore.benchmarks.eventstream_compare")
    profiles_ = benchmark_module_.Build_profiles(
        include_all_features_nofile=True)
    modes_ = {
        profile_.name: profile_.emulator_kwargs.get(
            "v2ce_burst_timestamps_mode")
        for profile_ in profiles_
        if "v2ce_burst_timestamps_mode" in profile_.emulator_kwargs
    }

    assert modes_ == {
        "v2ce_random": options_module_.V2ceBurstTimestampMode.RANDOM,
        "v2ce_slope": options_module_.V2ceBurstTimestampMode.SLOPE,
        "all_features_nofile": options_module_.V2ceBurstTimestampMode.SLOPE,
    }
    assert all(
        isinstance(mode_, options_module_.V2ceBurstTimestampMode)
        for mode_ in modes_.values()
    )
