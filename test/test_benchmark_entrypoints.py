"""Functional tests for benchmark command-line entry points.

Example:
    conda run -n v2e python -m pytest -q test/test_benchmark_entrypoints.py

Output:
    1 passed
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def test_comparative_benchmark_runs_outside_repo(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the repository output default and write real artifacts externally.

    Args:
        tmp_path: Isolated working and artifact directory.
        monkeypatch: Temporary command-line argument override.
    """
    repo_root_ = Path(__file__).resolve().parents[1]
    script_path_ = (
        repo_root_ / "v2ecore" / "benchmarks" /
        "benchmark_error_models_eventstream.py"
    )

    spec_ = importlib.util.spec_from_file_location(
        "benchmark_error_models_eventstream_entrypoint", script_path_)
    assert spec_ is not None and spec_.loader is not None
    benchmark_module_ = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(benchmark_module_)

    monkeypatch.setattr(sys, "argv", [str(script_path_)])
    args_ = benchmark_module_.Parse_args()
    assert args_.output_dir == repo_root_ / "output" / "benchmarks_comparative"

    output_dir_ = tmp_path / "benchmark-output"
    result_ = subprocess.run(
        [
            sys.executable,
            str(script_path_),
            "--output_dir", str(output_dir_),
            "--width", "8",
            "--height", "6",
            "--duration_s", "0.02",
            "--fps", "100",
            "--reference_factor", "1",
            "--runs", "1",
            "--warmup_runs", "0",
            "--device", "cpu",
            "--no-make_plots",
            "--no-cuda_sync",
            "--no-include_all_features_nofile",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result_.returncode == 0, result_.stderr
    assert len(list(output_dir_.glob("*.json"))) == 1
    assert len(list(output_dir_.glob("*.csv"))) == 1
    assert len(list(output_dir_.glob("*.npz"))) == 1
