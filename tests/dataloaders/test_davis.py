"""Tests for DAVIS/RPG text recording loading."""

from __future__ import annotations

import pickle  # noqa: S403
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from evlib.dataloaders import DavisRecordingLoader
from evlib.dataloaders import LoadingType
from evlib.types import RawEvents


HEIGHT = 3
WIDTH = 4


def _write_png(path: Path, value: int) -> np.ndarray:
    image = np.arange(HEIGHT * WIDTH, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    image = image + np.uint8(value)
    Image.fromarray(image, mode="L").save(path)
    return image


def _write_depth_png(path: Path, value: int) -> np.ndarray:
    depth = np.full((HEIGHT, WIDTH), value, dtype=np.uint16)
    Image.fromarray(depth).save(path)
    return depth.astype(np.float32)


def _make_davis_recording(root: Path) -> npt.NDArray[np.uint8]:
    """Create a minimal DAVIS/RPG text recording."""
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir()

    images = [
        _write_png(image_dir / "frame_00000000.png", 0),
        _write_png(image_dir / "frame_00000001.png", 10),
        _write_png(image_dir / "frame_00000002.png", 20),
    ]

    (root / "events.txt").write_text(
        "\n".join(
            [
                "0.000000 1 0 1",
                "0.050000 2 1 0",
                "0.100000 3 2 1",
                "0.150000 0 1 -1",
                "0.200000 1 2 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "images.txt").write_text(
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
    (root / "imu.txt").write_text(
        "\n".join(
            [
                "0.025000 1.0 2.0 3.0 0.1 0.2 0.3",
                "0.075000 4.0 5.0 6.0 0.4 0.5 0.6",
                "0.125000 7.0 8.0 9.0 0.7 0.8 0.9",
                "0.175000 10.0 11.0 12.0 1.0 1.1 1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "groundtruth.txt").write_text(
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
    (root / "calib.txt").write_text(
        "100.0 101.0 2.0 1.5 -0.1 0.2 0.01 0.02 0.0\n",
        encoding="utf-8",
    )
    return images[0]


def _add_depthmaps(root: Path) -> npt.NDArray[np.float32]:
    depth_dir = root / "depthmaps"
    depth_dir.mkdir()
    first_depth = _write_depth_png(depth_dir / "frame_00000000.png", 100)
    _write_depth_png(depth_dir / "frame_00000001.png", 200)
    (root / "depthmaps.txt").write_text(
        "\n".join(
            [
                "0.000000 depthmaps/frame_00000000.png",
                "0.100000 depthmaps/frame_00000001.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return first_depth


def _cache_dir(root: Path) -> str:
    return str(root / ".cache")


def _event_arrays(events: RawEvents) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return events.x, events.y, events.timestamp, events.polarity


class TestDavisRecordingLoader:  # noqa: D101
    def test_load_events_preserves_xy_order_and_polarity(
        self, tmp_path: Path
    ) -> None:  # noqa: D102
        """Event rows are parsed as timestamp x y polarity."""
        _make_davis_recording(tmp_path)
        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            events = loader.load_events(1, 4)

        assert isinstance(events, RawEvents)
        np.testing.assert_array_equal(events.x, np.array([2, 3, 0], dtype=np.int16))
        np.testing.assert_array_equal(events.y, np.array([1, 2, 1], dtype=np.int16))
        np.testing.assert_array_equal(events.timestamp, np.array([0.05, 0.1, 0.15]))
        np.testing.assert_array_equal(
            events.polarity,
            np.array([False, True, False], dtype=np.bool_),
        )

    def test_time_indexing_and_vectorized_helpers(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert loader.num_events == 5
            assert loader.time_to_index(0.0) == -1
            assert loader.time_to_index(0.1) == 1
            assert loader.index_to_time(-1) == pytest.approx(0.2)

            indices = loader.times_to_indices(np.array([0.0, 0.051, 0.3]))
            times = loader.indices_to_times(np.array([0, 2, 4]))

        np.testing.assert_array_equal(indices, np.array([-1, 1, 4], dtype=np.int64))
        np.testing.assert_array_equal(times, np.array([0.0, 0.1, 0.2]))

    def test_lazy_time_to_index_matches_cached_across_stride_boundary(self, tmp_path: Path) -> None:
        """Lazy lookup matches cached lookup even when many events share a timestamp.

        Lazy mode narrows the search with an index sampled every stride events.
        When a run of equal timestamps crosses one of those stride boundaries,
        a search can return an event exactly at the query time instead of just before it.
        """
        from evlib.dataloaders import _davis

        stride = _davis._TEXT_EVENT_TIMESTAMP_INDEX_STRIDE
        num_events = stride + 100
        timestamps = (np.arange(num_events, dtype=np.float64) * 1e-3).copy()
        # make several neighbouring events share one timestamp across a stride boundary
        boundary_time = timestamps[stride]
        timestamps[stride - 3 : stride + 4] = boundary_time
        lines = [f"{timestamps[i]:.9f} {i % 10} {(i * 7) % 10} {i % 2}" for i in range(num_events)]
        (tmp_path / "events.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        cache_dir = _cache_dir(tmp_path)

        probes = [
            boundary_time,
            float(timestamps[0]),
            float(timestamps[stride - 5]),
            float(timestamps[-1]) + 1.0,
        ]
        reference = [int(np.searchsorted(timestamps, t, side="left") - 1) for t in probes]

        with DavisRecordingLoader(str(tmp_path), event_load_mode="lazy", cache_dir=cache_dir) as lz:
            lazy = [lz.time_to_index(t) for t in probes]
        with DavisRecordingLoader(
            str(tmp_path), event_load_mode="cached", cache_dir=cache_dir
        ) as ch:
            cached = [ch.time_to_index(t) for t in probes]

        assert lazy == cached == reference

    def test_lazy_event_mode_builds_and_reuses_sidecar(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        cache_dir = tmp_path / ".cache"

        with DavisRecordingLoader(str(tmp_path), cache_dir=str(cache_dir)) as loader:
            assert loader.event_load_mode is LoadingType.LAZY
            assert loader.num_events == 5

        metadata_files = sorted(cache_dir.rglob("metadata.json"))
        assert len(metadata_files) == 1
        metadata_mtime = metadata_files[0].stat().st_mtime_ns

        with DavisRecordingLoader(str(tmp_path), cache_dir=str(cache_dir)) as loader:
            assert loader.num_events == 5

        assert metadata_files[0].stat().st_mtime_ns == metadata_mtime

    def test_cached_event_mode_matches_lazy_event_mode(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        cache_dir = _cache_dir(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=cache_dir) as lazy_loader:
            lazy_events = lazy_loader.load_events(0, lazy_loader.num_events)

        with DavisRecordingLoader(
            str(tmp_path),
            event_load_mode="cached",
            cache_dir=cache_dir,
        ) as cached_loader:
            cached_events = cached_loader.load_events(0, cached_loader.num_events)

        lazy_arrays = _event_arrays(lazy_events)
        cached_arrays = _event_arrays(cached_events)
        for index, lazy_array in enumerate(lazy_arrays):
            np.testing.assert_array_equal(cached_arrays[index], lazy_array)

        assert cached_loader.event_load_mode is LoadingType.CACHED

    def test_cached_loader_pickles_without_event_columns(self, tmp_path: Path) -> None:
        """Pickling (e.g. for DataLoader workers) never ships event columns."""
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            event_load_mode="cached",
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            expected = loader.load_events(0, loader.num_events)
            assert loader._event_backend._x is not None  # resident in cached mode

            restored = pickle.loads(pickle.dumps(loader))  # noqa: S301
            try:
                assert restored._event_backend._x is None  # not carried through pickle
                restored_events = restored.load_events(0, restored.num_events)
                np.testing.assert_array_equal(restored_events.x, expected.x)
                np.testing.assert_array_equal(restored_events.timestamp, expected.timestamp)
            finally:
                restored.close()

    def test_load_image_and_calibration(self, tmp_path: Path) -> None:  # noqa: D102
        expected_first_image = _make_davis_recording(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            image = loader.load_image(0)
            calibration = loader.calibration

        assert image is not None
        assert image.shape == (HEIGHT, WIDTH)
        assert not image.flags.writeable
        np.testing.assert_array_equal(image, expected_first_image)
        assert calibration is not None
        np.testing.assert_array_equal(
            calibration.camera_matrix,
            np.array(
                [
                    [100.0, 0.0, 2.0],
                    [0.0, 101.0, 1.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(
            calibration.distortion_coefficients,
            np.array([-0.1, 0.2, 0.01, 0.02, 0.0], dtype=np.float64),
        )

    def test_sensor_resolution_and_time_range(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert loader.sensor_resolution == (HEIGHT, WIDTH)
            assert loader.t_start == pytest.approx(0.0)
            assert loader.t_end == pytest.approx(0.2)
            assert loader.duration == pytest.approx(0.2)
            assert loader.num_images == loader.num_frames == 3
            image_timestamps = loader.image_timestamps
            assert image_timestamps is not None
            np.testing.assert_array_equal(image_timestamps, np.array([0.0, 0.1, 0.2]))

    def test_explicit_sensor_resolution_overrides_frame_shape(self, tmp_path: Path) -> None:
        """An explicit sensor_resolution takes precedence over the frame shape."""
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            sensor_resolution=(180, 240),
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            assert loader.sensor_resolution == (180, 240)

    def test_undistort_events_and_image_match_opencv(self, tmp_path: Path) -> None:
        """Undistortion uses the parsed calibration and matches OpenCV."""
        cv2 = pytest.importorskip("cv2")
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            calibration = loader.calibration
            assert calibration is not None
            camera_matrix = calibration.camera_matrix
            distortion = calibration.distortion_coefficients

            # The cached map must equal a direct opencv undistortion of the grid
            grid_x, grid_y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT), indexing="xy")
            grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
            expected_map = cv2.undistortPoints(
                grid.reshape(-1, 1, 2), camera_matrix, distortion, P=camera_matrix
            ).reshape(HEIGHT, WIDTH, 2)
            undistort_map = loader.undistort_map
            assert undistort_map is not None
            assert undistort_map.shape == (HEIGHT, WIDTH, 2)
            np.testing.assert_allclose(undistort_map, expected_map, rtol=1e-5, atol=1e-4)

            events = loader.load_events(0, loader.num_events)
            undistorted = loader.undistort_events(events)
            assert undistorted.x.dtype == np.int16 and undistorted.y.dtype == np.int16
            np.testing.assert_array_equal(undistorted.timestamp, events.timestamp)
            np.testing.assert_array_equal(undistorted.polarity, events.polarity)
            # coordinates equal the rounded map lookup
            expected_x = np.rint(undistort_map[events.y, events.x][:, 0]).astype(np.int16)
            expected_y = np.rint(undistort_map[events.y, events.x][:, 1]).astype(np.int16)
            np.testing.assert_array_equal(undistorted.x, expected_x)
            np.testing.assert_array_equal(undistorted.y, expected_y)

            image = loader.load_image(0)
            assert image is not None
            undistorted_image = loader.undistort_image(image)
            expected_image = cv2.undistort(image, camera_matrix, distortion, None, camera_matrix)
            np.testing.assert_array_equal(undistorted_image, expected_image)
            assert undistorted_image.dtype == np.uint8

    def test_undistort_zero_distortion_is_identity(self, tmp_path: Path) -> None:  # noqa: D102
        pytest.importorskip("cv2")
        _make_davis_recording(tmp_path)
        (tmp_path / "calib.txt").write_text(
            "100.0 101.0 2.0 1.5 0.0 0.0 0.0 0.0 0.0\n",
            encoding="utf-8",
        )

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            events = loader.load_events(0, loader.num_events)
            undistorted = loader.undistort_events(events)

        np.testing.assert_array_equal(undistorted.x, events.x)
        np.testing.assert_array_equal(undistorted.y, events.y)

    def test_undistort_requires_calibration(self, tmp_path: Path) -> None:  # noqa: D102
        pytest.importorskip("cv2")
        _make_davis_recording(tmp_path)
        (tmp_path / "calib.txt").unlink()

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert loader.calibration is None
            assert loader.undistort_map is None
            events = loader.load_events(0, 1)
            with pytest.raises(RuntimeError, match="calib.txt"):
                loader.undistort_events(events)

    def test_load_calibration_without_k3(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        (tmp_path / "calib.txt").write_text(
            "100.0 101.0 2.0 1.5 -0.1 0.2 0.01 0.02\n",
            encoding="utf-8",
        )

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            calibration = loader.calibration

        assert calibration is not None
        np.testing.assert_array_equal(
            calibration.distortion_coefficients,
            np.array([-0.1, 0.2, 0.01, 0.02, 0.0], dtype=np.float64),
        )

    def test_cached_image_mode_uses_predecoded_frames(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        with DavisRecordingLoader(
            str(tmp_path),
            image_load_mode="cached",
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            cached_before_overwrite = loader.load_image(0)
            _write_png(tmp_path / "images" / "frame_00000000.png", 99)
            cached_after_overwrite = loader.load_image(0)

        assert cached_before_overwrite is not None
        assert cached_after_overwrite is not None
        np.testing.assert_array_equal(cached_after_overwrite, cached_before_overwrite)

    def test_imu_window_and_nearest_pose(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            load_imu=True,
            load_gt_pose=True,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            imu = loader.load_imu(0.05, 0.15)
            pose = loader.load_nearest_pose(0.16)

        assert imu is not None
        np.testing.assert_array_equal(imu["timestamp"], np.array([0.075, 0.125]))
        np.testing.assert_array_equal(
            imu["linear_acceleration"],
            np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64),
        )
        assert pose is not None
        np.testing.assert_array_equal(
            pose,
            np.array([7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        )

    def test_nearest_pose_with_empty_trajectory_returns_none(self, tmp_path: Path) -> None:
        """An empty groundtruth.txt yields no pose instead of an IndexError."""
        _make_davis_recording(tmp_path)
        (tmp_path / "groundtruth.txt").write_text("", encoding="utf-8")

        with DavisRecordingLoader(
            str(tmp_path),
            load_gt_pose=True,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            assert loader.has_gt_pose
            assert loader.load_nearest_pose(0.1) is None
            # loading a frame sample must not crash when the trajectory is empty
            sample = loader.load_frame_sample(1)

        assert sample["pose"] is None

    def test_depth_lazy_and_cached_modes(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        expected_depth = _add_depthmaps(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            depth = loader.load_depth(0)
            nearest = loader.find_nearest_depth_index(0.09)

        assert depth is not None
        np.testing.assert_array_equal(depth, expected_depth)
        assert nearest == 1

        with DavisRecordingLoader(
            str(tmp_path),
            depth_load_mode="cached",
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            cached_depth = loader.load_depth(-1)

        assert cached_depth is not None
        np.testing.assert_array_equal(cached_depth, np.full((HEIGHT, WIDTH), 200.0))

    def test_depth_can_be_disabled(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        _add_depthmaps(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            depth_load_mode=False,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            assert not loader.has_depth
            assert loader.load_depth(0) is None

    def test_load_frame_sample(self, tmp_path: Path) -> None:
        """Frame samples use half-open ``[prev_frame, frame)`` event windows.

        ``time_to_index`` is strictly before,
        so the event at exactly the first frame timestamp belongs to frame 1 and frame 0's window is empty.
        """
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            load_imu=True,
            load_gt_pose=True,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            first_sample = loader.load_frame_sample(0)
            sample = loader.load_frame_sample(1)

        assert len(first_sample["events"]) == 0

        assert sample["timestamp"] == pytest.approx(0.1)
        # Frame 1 covers [0.0, 0.1): the events at 0.0 and 0.05.
        np.testing.assert_array_equal(sample["events"].timestamp, np.array([0.0, 0.05]))
        assert sample["image"] is not None
        assert sample["imu"] is not None
        assert sample["pose"] is not None
        assert sample["depth"] is None

    def test_load_frame_sample_includes_nearest_depth(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        _add_depthmaps(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            sample = loader.load_frame_sample(1)

        assert sample["depth"] is not None
        np.testing.assert_array_equal(sample["depth"], np.full((HEIGHT, WIDTH), 200.0))

    def test_frame_event_indices_precomputed_on_request(self, tmp_path: Path) -> None:
        """``precompute_frame_event_indices`` exposes the per-frame event offsets."""
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(
            str(tmp_path),
            precompute_frame_event_indices=True,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            indices = loader.frame_event_indices

        assert indices is not None
        np.testing.assert_array_equal(indices, np.array([0, 2, 4], dtype=np.int64))

    def test_frame_event_indices_stay_on_demand_in_lazy_mode(self, tmp_path: Path) -> None:
        """Lazy event mode skips precomputation yet still aligns frames on demand."""
        _make_davis_recording(tmp_path)

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert loader.frame_event_indices is None
            sample = loader.load_frame_sample(1)

        assert len(sample["events"]) == 2

    @pytest.mark.parametrize("sequence", [None, "shapes_rotation"])
    def test_resolves_sequence_directory(self, tmp_path: Path, sequence: str | None) -> None:
        """A nested sequence resolves both by auto-detection and by explicit name."""
        sequence_dir = tmp_path / "shapes_rotation"
        _make_davis_recording(sequence_dir)

        with DavisRecordingLoader(
            str(tmp_path),
            sequence=sequence,
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            assert loader.sequence == "shapes_rotation"
            assert loader.recording_dir == str(sequence_dir)
            assert loader.num_frames == 3

    def test_missing_events_file_raises(self, tmp_path: Path) -> None:  # noqa: D102
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match="events.txt"):
            DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path))

    def test_rejects_nonmonotonic_events(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        (tmp_path / "events.txt").write_text(
            "0.100000 1 1 1\n0.050000 2 1 0\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="nondecreasing"):
            DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path))

    def test_commented_events_fall_back_to_slow_parser(self, tmp_path: Path) -> None:  # noqa: D102
        _make_davis_recording(tmp_path)
        (tmp_path / "events.txt").write_text(
            "# timestamp x y polarity\n\n0.000000 1 0 1\n0.050000 2 1 0\n",
            encoding="utf-8",
        )

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            events = loader.load_events(0, loader.num_events)

        assert loader.num_events == 2
        np.testing.assert_array_equal(events.x, np.array([1, 2], dtype=np.int16))

    def test_fast_parser_stitches_rows_across_read_blocks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parser rejoins event rows that get split across fixed size reads.

        Other fixtures fit in a single read, so this path never runs. Shrinking the
        read size and the index stride forces the parser to carry a half read row into
        the next read, check timestamp order across reads, and sample the index per read.
        """
        from evlib.dataloaders import _davis

        monkeypatch.setattr(_davis, "_TEXT_EVENT_PARSE_BLOCK_BYTES", 64)
        monkeypatch.setattr(_davis, "_TEXT_EVENT_TIMESTAMP_INDEX_STRIDE", 4)

        num_events = 50
        xs = (np.arange(num_events) % 10).astype(np.int16)
        ys = (np.arange(num_events) * 7 % 10).astype(np.int16)
        ps = (np.arange(num_events) % 2).astype(np.bool_)
        timestamp_text = [f"{i * 1e-3:.9f}" for i in range(num_events)]
        lines = [f"{timestamp_text[i]} {xs[i]} {ys[i]} {int(ps[i])}" for i in range(num_events)]
        (tmp_path / "events.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Parse the text back to floats so the reference matches the loader's rounding.
        timestamps = np.array([float(text) for text in timestamp_text], dtype=np.float64)

        probes = [timestamps[0], timestamps[7], timestamps[-1], float(timestamps[-1]) + 1.0]
        reference = [int(np.searchsorted(timestamps, t, side="left") - 1) for t in probes]

        with DavisRecordingLoader(
            str(tmp_path), event_load_mode="lazy", cache_dir=_cache_dir(tmp_path)
        ) as loader:
            assert loader.num_events == num_events
            events = loader.load_events(0, num_events)
            observed = [loader.time_to_index(t) for t in probes]

        np.testing.assert_array_equal(events.x, xs)
        np.testing.assert_array_equal(events.y, ys)
        np.testing.assert_array_equal(events.timestamp, timestamps)
        np.testing.assert_array_equal(events.polarity, ps)
        assert observed == reference

    @pytest.mark.parametrize(
        ("bad_row", "message"),
        [
            ("0.000000 -1 0 1", "non-negative int16"),
            ("0.000000 40000 0 1", "non-negative int16"),
            ("0.000000 1.5 0 1", "must be integers"),
            ("0.000000 1 0 2", "polarity must be"),
        ],
    )
    def test_fast_parser_rejects_invalid_event_rows(
        self, tmp_path: Path, bad_row: str, message: str
    ) -> None:
        """The block parser validates coordinate range, integrality, and polarity."""
        _make_davis_recording(tmp_path)
        (tmp_path / "events.txt").write_text(bad_row + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path))

    def test_cache_rebuilds_when_source_changes(self, tmp_path: Path) -> None:
        """A modified events.txt invalidates the sidecar instead of serving stale data."""
        _make_davis_recording(tmp_path)
        cache_dir = tmp_path / ".cache"

        with DavisRecordingLoader(str(tmp_path), cache_dir=str(cache_dir)) as loader:
            assert loader.num_events == 5
            first_event = loader.load_events(0, 1)

        # Rewrite with different content; both file size and mtime change.
        (tmp_path / "events.txt").write_text(
            "0.000000 9 8 1\n0.010000 7 6 0\n",
            encoding="utf-8",
        )

        with DavisRecordingLoader(str(tmp_path), cache_dir=str(cache_dir)) as loader:
            assert loader.num_events == 2
            rebuilt_event = loader.load_events(0, 1)

        assert int(first_event.x[0]) == 1
        assert int(rebuilt_event.x[0]) == 9

    def test_zero_event_recording_reports_empty_time_range(self, tmp_path: Path) -> None:
        """An empty events.txt yields a valid empty recording rather than crashing."""
        _make_davis_recording(tmp_path)
        (tmp_path / "events.txt").write_text("", encoding="utf-8")

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert loader.num_events == 0
            assert loader.t_start is None
            assert loader.t_end is None
            assert loader.duration is None
            assert loader.time_to_index(0.0) == -1
            empty = loader.load_events(0, 0)

        assert len(empty.x) == 0

    def test_events_only_recording_without_frames(self, tmp_path: Path) -> None:
        """A recording with only events.txt loads; undistortion needs an explicit resolution."""
        pytest.importorskip("cv2")
        (tmp_path / "events.txt").write_text(
            "0.000000 1 0 1\n0.050000 2 1 0\n",
            encoding="utf-8",
        )
        (tmp_path / "calib.txt").write_text(
            "100.0 101.0 2.0 1.5 -0.1 0.2 0.01 0.02 0.0\n",
            encoding="utf-8",
        )

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            assert not loader.has_images
            assert loader.num_frames == 0
            assert loader.sensor_resolution is None
            assert loader.undistort_map is None
            assert loader.load_image(0) is None
            events = loader.load_events(0, loader.num_events)
            with pytest.raises(RuntimeError, match="sensor resolution"):
                loader.undistort_events(events)

        with DavisRecordingLoader(
            str(tmp_path),
            sensor_resolution=(8, 12),
            cache_dir=_cache_dir(tmp_path),
        ) as loader:
            assert loader.sensor_resolution == (8, 12)
            undistort_map = loader.undistort_map
            assert undistort_map is not None
            assert undistort_map.shape == (8, 12, 2)
            undistorted = loader.undistort_events(loader.load_events(0, loader.num_events))
            assert undistorted.x.dtype == np.int16

    def test_load_image_rejects_shape_mismatch(self, tmp_path: Path) -> None:
        """Frames whose shape differs from the first decoded frame are rejected."""
        pytest.importorskip("cv2")
        _make_davis_recording(tmp_path)
        odd_image = np.zeros((HEIGHT + 1, WIDTH + 2), dtype=np.uint8)
        Image.fromarray(odd_image, mode="L").save(tmp_path / "images" / "frame_00000001.png")

        with DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path)) as loader:
            loader.load_image(0)
            with pytest.raises(ValueError, match="shape mismatch"):
                loader.load_image(1)

    def test_ambiguous_recording_root_raises(self, tmp_path: Path) -> None:
        """A root containing several sequences is rejected as ambiguous."""
        _make_davis_recording(tmp_path / "seq_a")
        _make_davis_recording(tmp_path / "seq_b")

        with pytest.raises(FileNotFoundError, match="exactly one sequence"):
            DavisRecordingLoader(str(tmp_path), cache_dir=_cache_dir(tmp_path))
