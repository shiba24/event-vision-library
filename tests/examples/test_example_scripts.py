"""Tests for the public example scripts."""

from __future__ import annotations

import importlib.util
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest

from tests.dataloaders.test_davis import _make_davis_recording
from tests.dataloaders.test_dsec import SEQ as DSEC_SEQUENCE
from tests.dataloaders.test_dsec import _build_cleaned_dsec_tree
from tests.dataloaders.test_mvsec import SEQ as MVSEC_SEQUENCE
from tests.dataloaders.test_mvsec import _make_gt_hdf5
from tests.dataloaders.test_mvsec import _make_hdf5
from tests.datasets.test_ecd import _write_sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ECD_SEQUENCE = "shapes_rotation"


@pytest.fixture(autouse=True)
def example_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: D103
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    matplotlib.use("Agg")


def _load_example(script: str) -> ModuleType:
    path = REPO_ROOT / script
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_example(monkeypatch: pytest.MonkeyPatch, script: str, *argv: str) -> None:
    module = _load_example(script)
    monkeypatch.setattr(sys, "argv", [script, *argv])
    module.main()


def _run_example_process(cache_dir: Path, script: str, *argv: str) -> None:
    environment = dict(os.environ, MPLBACKEND="Agg", XDG_CACHE_HOME=str(cache_dir))
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(REPO_ROOT / script), *argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_flow_batch_example_runs_with_worker_processes(tmp_path: Path) -> None:  # noqa: D103
    _build_cleaned_dsec_tree(tmp_path)
    output = tmp_path / "dsec_flow_batch.png"

    _run_example_process(
        tmp_path / "cache",
        "examples/datasets/dsec_flow_batch.py",
        str(tmp_path),
        DSEC_SEQUENCE,
        "--output",
        str(output),
    )

    assert output.is_file()


def test_event_windows_and_flow_example_writes_a_figure(  # noqa: D103
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_hdf5(tmp_path / f"{MVSEC_SEQUENCE}_data.hdf5")
    _make_gt_hdf5(tmp_path / f"{MVSEC_SEQUENCE}_gt.hdf5")
    output = tmp_path / "mvsec_event_windows.png"

    _run_example(
        monkeypatch,
        "examples/readers/mvsec_event_windows.py",
        str(tmp_path),
        MVSEC_SEQUENCE,
        "--output",
        str(output),
    )

    assert output.is_file()


def test_frame_activity_example_writes_a_figure(  # noqa: D103
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_sequence(tmp_path, ECD_SEQUENCE)
    output = tmp_path / "ecd_frame_activity.png"

    _run_example(
        monkeypatch,
        "examples/datasets/ecd_frame_activity.py",
        str(tmp_path),
        ECD_SEQUENCE,
        "--output",
        str(output),
    )

    assert "inspected 3 frame intervals" in capsys.readouterr().out
    assert output.is_file()


def test_event_count_image_uses_explicit_resolution_without_frames(  # noqa: D103
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording"
    _make_davis_recording(recording)
    (recording / "images.txt").unlink()
    output = tmp_path / "ecd_event_count_image.png"

    _run_example(
        monkeypatch,
        "examples/readers/ecd_event_count_image.py",
        str(recording),
        "--sensor-resolution",
        "3",
        "4",
        "--output",
        str(output),
    )

    assert output.is_file()
