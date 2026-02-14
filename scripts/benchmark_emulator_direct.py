#!/usr/bin/env python3
"""Compatibility wrapper for unified benchmark_emulator.py.

This keeps old direct-benchmark invocations working by forwarding all arguments
to benchmark_emulator.py with defaults equivalent to the prior direct script.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    argv = [
        sys.argv[0],
        "--benchmark_mode", "repeated",
        "--timing_mode", "wall",
        *sys.argv[1:],
    ]
    sys.argv = argv
    unified_script = Path(__file__).with_name("benchmark_emulator.py")
    runpy.run_path(str(unified_script), run_name="__main__")


if __name__ == "__main__":
    main()
