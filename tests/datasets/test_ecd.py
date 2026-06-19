"""Tests for ECDDataset using synthetic ECD format sequences."""

# mypy: disable-error-code=no-untyped-def

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from evlib.dataloaders import DavisRecordingLoader
from evlib.datasets import ECD_DAVIS240C_SENSOR_RESOLUTION
from evlib.datasets import ECD_SEQUENCES
from evlib.datasets import ECDDataset
from evlib.datasets import ECDIterator
from evlib.datasets import ecd_collate_fn
from evlib.datasets import event_sample_collate
from evlib.types import RawEvents


HEIGHT, WIDTH = ECD_DAVIS240C_SENSOR_RESOLUTION


def _write_png(path: Path, value: int) -> npt.NDArray[np.uint8]:
    image_values = np.arange(HEIGHT * WIDTH, dtype=np.uint16)
    image = ((image_values.reshape(HEIGHT, WIDTH) + value) % 256).astype(np.uint8)
    Image.fromarray(image).save(path)
    return image


def _write_depth_png(path: Path, value: int) -> npt.NDArray[np.float32]:
    depth = np.full((HEIGHT, WIDTH), value, dtype=np.uint16)
    Image.fromarray(depth).save(path)
    return depth.astype(np.float32)


def _write_recording(
    recording_dir: Path,
    *,
    with_imu: bool = True,
    with_pose: bool = True,
    with_depth: bool = False,
) -> None:
    recording_dir.mkdir(parents=True, exist_ok=True)
    image_dir = recording_dir / "images"
    image_dir.mkdir()

    _write_png(image_dir / "frame_00000000.png", 0)
    _write_png(image_dir / "frame_00000001.png", 10)
    _write_png(image_dir / "frame_00000002.png", 20)

    (recording_dir / "events.txt").write_text(
        "\n".join(
            [
                "0.000000 1 0 1",
                "0.050000 2 1 0",
                "0.100000 3 2 1",
                "0.150000 4 3 -1",
                "0.200000 5 4 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recording_dir / "images.txt").write_text(
        "\n".join(
            [
                "0.000000 images/frame_00000000.png",
                "0.100000 images/frame_00000001.png",
                "0.200000 images/frame_00000002.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recording_dir / "calib.txt").write_text(
        "120.0 121.0 120.0 90.0 -0.1 0.2 0.01 0.02 0.0\n",
        encoding="utf-8",
    )

    if with_imu:
        (recording_dir / "imu.txt").write_text(
            "\n".join(
                [
                    "0.025000 1.0 2.0 3.0 0.1 0.2 0.3",
                    "0.075000 4.0 5.0 6.0 0.4 0.5 0.6",
                    "0.125000 7.0 8.0 9.0 0.7 0.8 0.9",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    if with_pose:
        (recording_dir / "groundtruth.txt").write_text(
            "\n".join(
                [
                    "0.000000 1.0 2.0 3.0 0.0 0.0 0.0 1.0",
                    "0.100000 4.0 5.0 6.0 0.0 0.0 1.0 0.0",
                    "0.200000 7.0 8.0 9.0 0.0 1.0 0.0 0.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    if with_depth:
        depth_dir = recording_dir / "depthmaps"
        depth_dir.mkdir()
        _write_depth_png(depth_dir / "frame_00000000.png", 100)
        _write_depth_png(depth_dir / "frame_00000001.png", 200)
        (recording_dir / "depthmaps.txt").write_text(
            "\n".join(
                [
                    "0.000000 depthmaps/frame_00000000.png",
                    "0.100000 depthmaps/frame_00000001.png",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def _write_sequence(root: Path, sequence: str, **kwargs) -> Path:
    recording_dir = root / sequence
    _write_recording(recording_dir, **kwargs)
    return recording_dir


def _write_nested_sequence(root: Path, sequence: str, **kwargs) -> Path:
    recording_dir = root / sequence / sequence
    _write_recording(recording_dir, **kwargs)
    return recording_dir


def _cache_dir(root: Path) -> str:
    return str(root / ".cache")


@pytest.fixture()
def ecd_root(tmp_path: Path) -> Path:
    """Synthetic ECD root with one extracted sequence."""
    _write_sequence(tmp_path, "shapes_rotation")
    return tmp_path


class TestECDDataset:
    """ECD map style dataset behavior."""

    def test_sequence_sample_uses_davis_loader_and_loads_default_modalities(
        self,
        ecd_root: Path,
    ) -> None:
        """A sample carries events, frame, IMU, and pose by default."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            sample = ds[1]

            assert isinstance(ds.loader, DavisRecordingLoader)
            assert ds.sequence == "shapes_rotation"
            assert ds.camera_model == "DAVIS240C"
            assert ds.sensor_resolution == (HEIGHT, WIDTH)
            assert len(ds) == 3
            assert ds.num_events == 5
            assert sample["timestamp"] == pytest.approx(0.1)
            assert sample["image"] is not None
            assert sample["image"].shape == (HEIGHT, WIDTH)
            assert sample["imu"] is not None
            assert sample["imu"]["timestamp"].tolist() == [0.025, 0.075]
            assert sample["pose"] is not None
            np.testing.assert_array_equal(sample["pose"][:3], np.array([4.0, 5.0, 6.0]))

    def test_negative_index_returns_last_frame(self, ecd_root: Path) -> None:
        """ds[-1] is the last frame."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            assert ds[-1]["timestamp"] == ds[len(ds) - 1]["timestamp"]

    def test_index_out_of_range_raises(self, ecd_root: Path) -> None:
        """An index past the last frame raises IndexError."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            with pytest.raises(IndexError):
                ds[len(ds)]

    def test_event_time_and_index_helpers_delegate_to_loader(self, ecd_root: Path) -> None:
        """Time and index lookups return the loader's event results."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            events = ds.get_events_by_time(0.05, 0.16)
            indices = ds.times_to_indices(np.array([0.0, 0.101, 0.3]))
            times = ds.indices_to_times(np.array([0, 2, 4]))

        np.testing.assert_array_equal(events.x, np.array([2, 3, 4], dtype=np.int16))
        np.testing.assert_array_equal(indices, np.array([-1, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(times, np.array([0.0, 0.1, 0.2]))

    def test_nested_extracted_archive_layout(self, tmp_path: Path) -> None:
        """A sequence nested one directory deep still resolves."""
        recording_dir = _write_nested_sequence(tmp_path, "boxes_6dof")

        with ECDDataset(str(tmp_path), "boxes_6dof", cache_dir=_cache_dir(tmp_path)) as ds:
            assert ds.recording_dir == str(recording_dir)
            assert ds.num_frames == 3
            assert ds.load_image(0) is not None

    def test_missing_optional_modalities_return_none(self, tmp_path: Path) -> None:
        """A sequence without imu.txt loads, and the imu field is None."""
        _write_sequence(tmp_path, "slider_depth", with_imu=False, with_pose=True)

        with ECDDataset(str(tmp_path), "slider_depth", cache_dir=_cache_dir(tmp_path)) as ds:
            sample = ds[1]

            assert not ds.has_imu
            assert ds.has_gt_pose
            assert sample["imu"] is None
            assert sample["pose"] is not None

    def test_synthetic_sequence_depthmaps_are_exposed(self, tmp_path: Path) -> None:
        """Synthetic sequence depth maps reach the sample dict."""
        _write_sequence(
            tmp_path,
            "simulation_3planes",
            with_imu=False,
            with_pose=True,
            with_depth=True,
        )

        with ECDDataset(str(tmp_path), "simulation_3planes", cache_dir=_cache_dir(tmp_path)) as ds:
            sample = ds[1]
            depth = ds.load_depth(0)

            assert ds.has_depth
            assert ds.num_depth_maps == 2
            assert sample["depth"] is not None
            assert sample["depth"].shape == (HEIGHT, WIDTH)
            assert depth is not None
            assert depth.dtype == np.float32

    def test_available_sequences_reports_extracted_sequences_only(self, tmp_path: Path) -> None:
        """Discovery lists extracted sequences and skips zip only ones."""
        _write_sequence(tmp_path, "shapes_rotation")
        _write_nested_sequence(tmp_path, "boxes_6dof")
        (tmp_path / "slider_depth.zip").write_bytes(b"not a real zip")

        available = ECDDataset.available_sequences(str(tmp_path))

        assert available == ("shapes_rotation", "boxes_6dof")
        assert ECDDataset.available_sequences() == ECD_SEQUENCES

    def test_unknown_sequence_and_path_like_sequence_raise(self, tmp_path: Path) -> None:
        """Unknown names and path like input are rejected."""
        with pytest.raises(ValueError, match="Unknown ECD sequence"):
            ECDDataset(str(tmp_path), "not_a_sequence")

        with pytest.raises(ValueError, match="not a path"):
            ECDDataset(str(tmp_path), "../shapes_rotation")

    def test_zip_only_sequence_has_actionable_error(self, tmp_path: Path) -> None:
        """A zip only sequence tells the user to extract it first."""
        (tmp_path / "slider_depth.zip").write_bytes(b"not a real zip")

        with pytest.raises(FileNotFoundError, match="Extract the text archive"):
            ECDDataset(str(tmp_path), "slider_depth")

    def test_undistort_events_applies_calibration(self, ecd_root: Path) -> None:
        """Calibration is parsed and undistortion moves event coordinates."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            events = ds.load_events(0, ds.num_events)
            undistorted = ds.undistort_events(events)

            assert ds.has_calibration
            assert len(undistorted) == len(events)
            moved = not np.array_equal(undistorted.x, events.x) or not np.array_equal(
                undistorted.y, events.y
            )
            assert moved

    def test_lazy_and_cached_modes_return_same_sample(self, ecd_root: Path) -> None:
        """Cached loading yields the same sample as the lazy default."""
        cache = _cache_dir(ecd_root)
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=cache) as lazy_ds:
            lazy = lazy_ds[1]
        with ECDDataset(
            str(ecd_root),
            "shapes_rotation",
            event_load_mode="cached",
            image_load_mode="cached",
            cache_dir=cache,
        ) as cached_ds:
            cached = cached_ds[1]

        assert cached["timestamp"] == lazy["timestamp"]
        np.testing.assert_array_equal(cached["image"], lazy["image"])
        np.testing.assert_array_equal(cached["events"].x, lazy["events"].x)

    def test_repr_includes_root_and_sequence(self, ecd_root: Path) -> None:
        """Repr shows the root and sequence name."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            text = repr(ds)

        assert "ECDDataset" in text
        assert str(ecd_root) in text
        assert "shapes_rotation" in text


class TestECDIterator:
    """ECD iterator behavior."""

    def test_iterator_reset_replays_sequence(self, ecd_root: Path) -> None:
        """Reset rewinds the iterator for a second pass."""
        with ECDIterator(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as it:
            first_pass = list(it)
            it.reset()
            second_pass = list(it)

        assert len(first_pass) == 3
        assert len(second_pass) == 3
        assert first_pass[0]["timestamp"] == second_pass[0]["timestamp"]

    def test_iterator_stop(self, ecd_root: Path) -> None:
        """Next raises StopIteration past the last frame."""
        with ECDIterator(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as it:
            for _ in it:
                pass

            with pytest.raises(StopIteration):
                next(it)


class TestECDCollate:
    """ECD collate helper behavior."""

    def test_collate_aliases_generic_helper(self) -> None:
        """ecd_collate_fn is the shared event sample collate."""
        assert ecd_collate_fn is event_sample_collate

    def test_collate_preserves_variable_length_events(self, ecd_root: Path) -> None:
        """Collate keeps variable-length event windows as a list."""
        with ECDDataset(str(ecd_root), "shapes_rotation", cache_dir=_cache_dir(ecd_root)) as ds:
            batch = ecd_collate_fn([ds[0], ds[1]])

        assert isinstance(batch["timestamp"], np.ndarray)
        assert batch["timestamp"].shape == (2,)
        assert isinstance(batch["events"], list)
        assert [len(events) for events in batch["events"]] == [0, 2]
        assert all(isinstance(events, RawEvents) for events in batch["events"])
        assert isinstance(batch["image"], list)
        assert isinstance(batch["imu"], list)
        assert isinstance(batch["pose"], list)


def test_real_ecd_slider_depth(tmp_path: Path) -> None:
    """Opt in check against a local extracted ECD copy."""
    ecd_root = os.environ.get("EVLIB_ECD_ROOT")
    if ecd_root is None:
        pytest.skip("Set EVLIB_ECD_ROOT to an extracted ECD text-file dataset root.")

    with ECDDataset(ecd_root, "slider_depth", cache_dir=_cache_dir(tmp_path)) as ds:
        first_event_count = min(128, ds.num_events)
        first_events = ds.load_events(0, first_event_count)
        sample = ds[0]

        assert ds.sequence == "slider_depth"
        assert ds.camera_model == "DAVIS240C"
        assert ds.sensor_resolution == (HEIGHT, WIDTH)
        assert ds.num_events > 0
        assert len(ds) > 0
        assert len(first_events) == first_event_count
        assert sample["image"] is not None
        assert sample["image"].shape == (HEIGHT, WIDTH)
        assert sample["imu"] is None
        assert sample["pose"] is not None
