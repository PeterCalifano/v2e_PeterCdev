"""Define finite option values shared by CLI and model consumers.

Enum values intentionally match the existing public CLI strings. This keeps
argument parsing, manifests, logs, and benchmark output stable while replacing
scattered raw-string comparisons with explicit finite domains.

Example:
    print([mode.value for mode in V2ceBurstTimestampMode])
    print(str(RendererColorMode.GREEN_RED))

Output:
    ['random', 'slope']
    green_red

Authors:
    Pietro Califano <petercalifano.gs@gmail.com>
Repository originally developed by:
    Tobi Delbruck <tobi@ini.uzh.ch>
    Yuhuang Hu <yuhuang.hu@ini.uzh.ch>
    Zhe He <zhehe@student.ethz.ch>
"""

from enum import StrEnum


class V2ceBurstTimestampMode(StrEnum):
    """V2CE-inspired timestamp-placement modes."""

    RANDOM = "random"
    SLOPE = "slope"


class DvsParamPreset(StrEnum):
    """Named DVS parameter presets."""

    CLEAN = "clean"
    NOISY = "noisy"


class IebcsNoiseSource(StrEnum):
    """IEBCS histogram-noise source selectors."""

    PRESET = "preset"
    FILES = "files"


class IebcsNoisePreset(StrEnum):
    """IEBCS histogram-noise preset names.

    Presets resolve to expected files under ``input/iebcs_noise``. Those files
    are not bundled in this repository snapshot.
    """

    LUX_3K = "3klux"
    LUX_161 = "161lux"
    LUX_0_1 = "0.1lux"


class RendererColorMode(StrEnum):
    """DVS video and preview color modes."""

    GRAYSCALE = "grayscale"
    GREEN_RED = "green_red"


IEBCS_NOISE_PRESET_FILES: dict[IebcsNoisePreset, tuple[str, str]] = {
    IebcsNoisePreset.LUX_3K: ("noise_pos_3klux.npy", "noise_neg_3klux.npy"),
    IebcsNoisePreset.LUX_161: ("noise_pos_161lux.npy", "noise_neg_161lux.npy"),
    IebcsNoisePreset.LUX_0_1: ("noise_pos_0.1lux.npy", "noise_neg_0.1lux.npy"),
}
