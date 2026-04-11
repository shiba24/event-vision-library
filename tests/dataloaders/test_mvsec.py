"""Tests for MVSECDataLoader and DataLoaderBase.

Reuses the same synthetic MVSEC format files as test_mvsec.py.
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

from evlib.dataloaders import MVSECOdometryData
from evlib.dataloaders._base import DataLoaderBase
from evlib.dataloaders._mvsec import MVSECDataLoader
from evlib.dataloaders._mvsec import _resolve_load_mode
from evlib.dataloaders.utils import find_nearest_index
from evlib.types import RawEvents


HEIGHT, WIDTH = 260, 346
N_EVENTS = 500
N_FRAMES = 20
N_GT_FRAMES = 10
N_IMU = 100
N_DEPTH_FRAMES = 10
N_ODOM_FRAMES = 12
N_POSE_FRAMES = 50
N_VELODYNE_SCANS = 8
N_VELODYNE_POINTS = 200  # small for fast tests
SEQ = "indoor_flying1"
CATEGORY = "indoor_flying"
T_START = 1504645177.0
T_END = 1504645247.0


# Synthetic data helpers


def _make_hdf5(
    path: Path, n_events: int = N_EVENTS, n_frames: int = N_FRAMES, camera: str = "left"
) -> None:
    """Create a minimal MVSEC format HDF5 file matching real layout."""
    rng = np.random.RandomState(42)
    x = rng.randint(0, WIDTH, n_events).astype(np.float64)
    y = rng.randint(0, HEIGHT, n_events).astype(np.float64)
    t = np.sort(rng.uniform(T_START, T_END, n_events))
    p = rng.choice([-1.0, 1.0], n_events)
    events = np.stack([x, y, t, p], axis=1)

    with h5py.File(path, "w") as f:
        grp = f.create_group(f"davis/{camera}")
        grp.create_dataset("events", data=events)
        grp.create_dataset(
            "image_raw", data=rng.randint(0, 256, (n_frames, HEIGHT, WIDTH), dtype=np.uint8)
        )
        frame_ts = np.linspace(T_START, T_END, n_frames)
        grp.create_dataset("image_raw_ts", data=frame_ts)
        grp.create_dataset(
            "image_raw_event_inds",
            data=np.linspace(0, n_events - 1, n_frames, dtype=np.int64),
        )


def _add_imu_to_hdf5(path: Path, camera: str = "left") -> None:
    """Add IMU datasets to an existing data HDF5."""
    rng = np.random.RandomState(3)
    with h5py.File(path, "a") as f:
        grp = f[f"davis/{camera}"]
        imu_data = rng.normal(0, 1, (N_IMU, 6))
        grp.create_dataset("imu", data=imu_data)
        imu_ts = np.linspace(T_START, T_END, N_IMU)
        grp.create_dataset("imu_ts", data=imu_ts)


def _add_velodyne_to_hdf5(path: Path) -> None:
    """Add velodyne datasets to an existing data HDF5."""
    rng = np.random.RandomState(5)
    with h5py.File(path, "a") as f:
        vel_grp = f.create_group("velodyne")
        scans = rng.uniform(-10, 10, (N_VELODYNE_SCANS, N_VELODYNE_POINTS, 4)).astype(np.float32)
        vel_grp.create_dataset("scans", data=scans)
        vel_ts = np.linspace(T_START, T_END, N_VELODYNE_SCANS)
        vel_grp.create_dataset("scans_ts", data=vel_ts)


def _make_gt_flow_npz(path: Path, n_frames: int = N_GT_FRAMES) -> None:
    rng = np.random.RandomState(0)
    x_flow_dist = rng.uniform(-5, 5, (n_frames, HEIGHT, WIDTH))
    y_flow_dist = rng.uniform(-5, 5, (n_frames, HEIGHT, WIDTH))
    timestamps = np.linspace(T_START, T_END, n_frames)
    np.savez(path, x_flow_dist=x_flow_dist, y_flow_dist=y_flow_dist, timestamps=timestamps)


def _make_constant_gt_flow_npz(
    path: Path, x_value: float = 4.0, y_value: float = -2.0, n_frames: int = 4
) -> None:
    x_flow_dist = np.full((n_frames, HEIGHT, WIDTH), x_value, dtype=np.float32)
    y_flow_dist = np.full((n_frames, HEIGHT, WIDTH), y_value, dtype=np.float32)
    timestamps = np.linspace(T_START, T_END, n_frames)
    np.savez(path, x_flow_dist=x_flow_dist, y_flow_dist=y_flow_dist, timestamps=timestamps)


def _make_odom_npz(path: Path, n_frames: int = N_ODOM_FRAMES) -> None:
    """Create synthetic odometry NPZ."""
    rng = np.random.RandomState(11)
    np.savez(
        path,
        timestamps=np.linspace(T_START, T_END, n_frames),
        lin_vel=rng.normal(0, 1, (n_frames, 3)),
        pos=rng.normal(0, 1, (n_frames, 3)),
        quat=rng.normal(0, 1, (n_frames, 4)),
        ang_vel=rng.normal(0, 1, (n_frames, 3)),
    )


def _make_gt_hdf5(path: Path, camera: str = "left") -> None:
    """Create synthetic gt.hdf5 with depth, blended, flow, odometry, poses."""
    rng = np.random.RandomState(7)
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"davis/{camera}")

        # Depth (raw)
        depth_raw = rng.uniform(0.5, 50.0, (N_DEPTH_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
        grp.create_dataset("depth_image_raw", data=depth_raw)
        grp.create_dataset("depth_image_raw_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        # Depth (rectified)
        depth_rect = rng.uniform(0.5, 50.0, (N_DEPTH_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
        grp.create_dataset("depth_image_rect", data=depth_rect)
        grp.create_dataset("depth_image_rect_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        # Blended images
        blended = rng.randint(0, 256, (N_DEPTH_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
        grp.create_dataset("blended_image_rect", data=blended)
        grp.create_dataset(
            "blended_image_rect_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES)
        )

        # Flow in HDF5
        flow = rng.uniform(-5, 5, (N_DEPTH_FRAMES, 2, HEIGHT, WIDTH))
        grp.create_dataset("flow_dist", data=flow)
        grp.create_dataset("flow_dist_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        # GT odometry (LOAM)
        odom = np.tile(np.eye(4), (N_ODOM_FRAMES, 1, 1))
        grp.create_dataset("odometry", data=odom)
        grp.create_dataset("odometry_ts", data=np.linspace(T_START, T_END, N_ODOM_FRAMES))

        # GT pose (Cartographer)
        pose = np.tile(np.eye(4), (N_POSE_FRAMES, 1, 1))
        grp.create_dataset("pose", data=pose)
        grp.create_dataset("pose_ts", data=np.linspace(T_START, T_END, N_POSE_FRAMES))


def _assert_raw_events_equal(left: RawEvents, right: RawEvents) -> None:
    """Assert that two RawEvents batches contain identical data."""
    np.testing.assert_array_equal(left.x, right.x)
    np.testing.assert_array_equal(left.y, right.y)
    np.testing.assert_array_equal(left.timestamp, right.timestamp)
    np.testing.assert_array_equal(left.polarity, right.polarity)


# Fixtures


@pytest.fixture()
def mvsec_dir(tmp_path: Path) -> Path:
    """Synthetic MVSEC directory with GT flow (no calibration)."""
    _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
    _make_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")
    return tmp_path


@pytest.fixture()
def mvsec_full_dir(tmp_path: Path) -> Path:
    """Synthetic MVSEC directory with all modalities."""
    data_path = tmp_path / f"{SEQ}_data.hdf5"
    _make_hdf5(data_path)
    _add_imu_to_hdf5(data_path)
    _add_velodyne_to_hdf5(data_path)
    _make_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")
    _make_odom_npz(tmp_path / f"{SEQ}_odom.npz")
    _make_gt_hdf5(tmp_path / f"{SEQ}_gt.hdf5")
    return tmp_path


# Test helpers


class TestResolveLoadMode:
    def test_false(self) -> None:
        should_load, should_cache = _resolve_load_mode(False)
        assert should_load is False
        assert should_cache is False

    def test_true(self) -> None:
        should_load, should_cache = _resolve_load_mode(True)
        assert should_load is True
        assert should_cache is False

    def test_lazy_string(self) -> None:
        should_load, should_cache = _resolve_load_mode("lazy")
        assert should_load is True
        assert should_cache is False

    def test_cached_string(self) -> None:
        should_load, should_cache = _resolve_load_mode("cached")
        assert should_load is True
        assert should_cache is True

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            _resolve_load_mode("invalid")  # type: ignore[arg-type]


class TestResidentLoadModes:
    def test_invalid_event_mode_raises(self, mvsec_dir: Path) -> None:
        with pytest.raises(ValueError, match="event_load_mode"):
            MVSECDataLoader(
                str(mvsec_dir),
                SEQ,
                load_calibration=False,
                event_load_mode="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_image_mode_raises(self, mvsec_dir: Path) -> None:
        with pytest.raises(ValueError, match="image_load_mode"):
            MVSECDataLoader(
                str(mvsec_dir),
                SEQ,
                load_calibration=False,
                image_load_mode="invalid",  # type: ignore[arg-type]
            )


class TestFindNearestIndex:
    def test_exact_match(self) -> None:
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_nearest_index(ts, 3.0) == 2

    def test_between_values(self) -> None:
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_nearest_index(ts, 2.3) == 1
        assert find_nearest_index(ts, 2.7) == 2

    def test_before_start(self) -> None:
        ts = np.array([1.0, 2.0, 3.0])
        assert find_nearest_index(ts, 0.0) == 0

    def test_after_end(self) -> None:
        ts = np.array([1.0, 2.0, 3.0])
        assert find_nearest_index(ts, 99.0) == 2


# Existing dataloader tests


class TestDataLoaderBase:
    def test_cannot_instantiate(self) -> None:
        """DataLoaderBase is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataLoaderBase()  # type: ignore[abstract]


class TestMVSECDataLoader:
    def test_isinstance_dataloader_base(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            assert isinstance(loader, DataLoaderBase)

    def test_load_events(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            events = loader.load_events(0, 100)
            assert isinstance(events, RawEvents)
            assert len(events) == 100
            assert events.x.dtype == np.int16
            assert events.y.dtype == np.int16
            assert events.timestamp.dtype == np.float64
            assert events.polarity.dtype == np.bool_

    def test_num_events(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            assert loader.num_events == N_EVENTS

    def test_time_to_index(self, mvsec_dir: Path) -> None:
        """time_to_index returns the last event strictly before t."""
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            t_mid = (T_START + T_END) / 2
            idx = loader.time_to_index(t_mid)
            assert -1 <= idx < N_EVENTS
            if idx >= 0:
                assert loader.index_to_time(idx) < t_mid
            if idx + 1 < N_EVENTS:
                assert loader.index_to_time(idx + 1) >= t_mid

    def test_index_to_time(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            t0 = loader.index_to_time(0)
            t_last = loader.index_to_time(N_EVENTS - 1)
            assert T_START <= t0 <= t_last <= T_END

    def test_times_to_indices_matches_scalar(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            query_times = np.array([T_START, (T_START + T_END) / 2, T_END], dtype=np.float64)
            bulk_indices = loader.times_to_indices(query_times)
            scalar_indices = np.array([loader.time_to_index(float(t)) for t in query_times])
            np.testing.assert_array_equal(bulk_indices, scalar_indices)

    def test_indices_to_times_matches_scalar(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            query_indices = np.array([0, N_EVENTS // 2, N_EVENTS - 1], dtype=np.int64)
            bulk_timestamps = loader.indices_to_times(query_indices)
            scalar_timestamps = np.array([loader.index_to_time(int(i)) for i in query_indices])
            np.testing.assert_array_equal(bulk_timestamps, scalar_timestamps)

    def test_time_to_index_with_duplicate_timestamps(self, tmp_path: Path) -> None:
        duplicate_timestamps = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
        x = np.array([0, 1, 2, 3], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        p = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        events = np.stack([x, y, duplicate_timestamps, p], axis=1)

        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=events)
            grp.create_dataset("image_raw_ts", data=np.array([1.0, 2.0], dtype=np.float64))

        cache_dir = tmp_path / ".cache"
        with MVSECDataLoader(
            str(tmp_path),
            SEQ,
            load_gt_flow=False,
            load_calibration=False,
            event_load_mode="lazy",
            cache_dir=str(cache_dir),
        ) as loader:
            assert loader.time_to_index(1.0) == -1
            assert loader.time_to_index(2.0) == 1

    def test_get_events_by_time(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            t_mid = (T_START + T_END) / 2
            events = loader.get_events_by_time(T_START, t_mid)
            assert isinstance(events, RawEvents)
            assert len(events) > 0
            assert np.all(events.timestamp >= T_START)
            assert np.all(events.timestamp < t_mid)

    def test_iter_events_by_count(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            chunks = list(loader.iter_events(num_events=100))
            assert len(chunks) > 0
            total = sum(len(c) for c in chunks)
            assert total == N_EVENTS
            for c in chunks[:-1]:
                assert len(c) == 100

    def test_iter_events_by_time(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            window = (T_END - T_START) / 5
            chunks = list(loader.iter_events(time_window=window))
            assert len(chunks) > 0
            total = sum(len(c) for c in chunks)
            assert total == N_EVENTS

    def test_load_optical_flow(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_gt_flow=True,
            load_calibration=False,
        ) as loader:
            assert loader.has_gt_flow
            gt_ts = loader.gt_time_list()
            flow = loader.load_optical_flow(gt_ts[0], gt_ts[2])
            assert flow.shape == (HEIGHT, WIDTH, 2)

    def test_load_optical_flow_within_single_gt_interval_scales_constant_flow(
        self, tmp_path: Path
    ) -> None:
        _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
        _make_constant_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")

        with MVSECDataLoader(
            str(tmp_path), SEQ, load_gt_flow=True, load_calibration=False
        ) as loader:
            gt_ts = loader.gt_time_list()
            gt_interval_duration = gt_ts[1] - gt_ts[0]
            t_start = gt_ts[0] + 0.1 * gt_interval_duration
            t_end = gt_ts[0] + 0.6 * gt_interval_duration
            requested_duration = t_end - t_start
            scale_factor = requested_duration / gt_interval_duration

            flow = loader.load_optical_flow(float(t_start), float(t_end))

            np.testing.assert_allclose(flow[..., 0], 4.0 * scale_factor)
            np.testing.assert_allclose(flow[..., 1], -2.0 * scale_factor)

    def test_load_optical_flow_across_multiple_gt_intervals_accumulates_constant_flow(
        self,
        tmp_path: Path,
    ) -> None:
        _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
        _make_constant_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")

        with MVSECDataLoader(
            str(tmp_path), SEQ, load_gt_flow=True, load_calibration=False
        ) as loader:
            gt_ts = loader.gt_time_list()
            flow = loader.load_optical_flow(float(gt_ts[0]), float(gt_ts[2]))

            valid_region = flow[4:, :-8]
            np.testing.assert_allclose(valid_region[..., 0], 8.0)
            np.testing.assert_allclose(valid_region[..., 1], -4.0)

    def test_load_frame_sample_returns_flow_for_short_frame_interval(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_gt_flow=True,
            load_calibration=False,
        ) as loader:
            sample = loader.load_frame_sample(1)
            assert sample["flow"] is not None

    def test_num_frames(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            assert loader.num_frames == N_FRAMES

    def test_load_image(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            img = loader.load_image(0)
            assert img is not None
            assert img.shape == (HEIGHT, WIDTH)
            assert img.dtype == np.uint8

    def test_lazy_event_mode_matches_cached(self, mvsec_dir: Path) -> None:
        cache_dir = mvsec_dir / ".cache"
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as cached_loader:
            with MVSECDataLoader(
                str(mvsec_dir),
                SEQ,
                load_calibration=False,
                event_load_mode="lazy",
                cache_dir=str(cache_dir),
            ) as lazy_loader:
                cached_events = cached_loader.load_events(10, 120)
                lazy_events = lazy_loader.load_events(10, 120)
                _assert_raw_events_equal(cached_events, lazy_events)

                t_mid = (T_START + T_END) / 2
                assert lazy_loader.time_to_index(t_mid) == cached_loader.time_to_index(t_mid)
                assert lazy_loader.index_to_time(42) == cached_loader.index_to_time(42)

                cached_time_window = cached_loader.get_events_by_time(T_START, t_mid)
                lazy_time_window = lazy_loader.get_events_by_time(T_START, t_mid)
                _assert_raw_events_equal(cached_time_window, lazy_time_window)

    def test_lazy_image_mode_loads_readonly_frame(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            image_load_mode="lazy",
        ) as loader:
            assert loader.has_images
            assert loader.images is None

            img = loader.load_image(0)
            assert img is not None
            assert img.shape == (HEIGHT, WIDTH)
            assert img.dtype == np.uint8
            assert img.flags.writeable is False

    def test_lazy_event_mode_survives_close(self, mvsec_dir: Path) -> None:
        cache_dir = mvsec_dir / ".cache"
        loader = MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            event_load_mode="lazy",
            cache_dir=str(cache_dir),
        )
        before_close = loader.load_events(5, 25)
        loader.close()
        after_close = loader.load_events(5, 25)
        _assert_raw_events_equal(before_close, after_close)

    def test_lazy_event_mode_builds_and_reuses_sidecar(self, mvsec_dir: Path) -> None:
        cache_dir = mvsec_dir / ".cache"
        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            event_load_mode="lazy",
            cache_dir=str(cache_dir),
        ) as first_loader:
            first_loader.load_events(0, 10)

        metadata_files = sorted(cache_dir.rglob("metadata.json"))
        assert len(metadata_files) == 1

        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            event_load_mode="lazy",
            cache_dir=str(cache_dir),
        ) as second_loader:
            second_events = second_loader.load_events(0, 10)
            assert len(second_events) == 10

    def test_cached_event_mode_with_explicit_cache_dir_builds_and_reuses_sidecar(
        self, mvsec_dir: Path
    ) -> None:
        cache_dir = mvsec_dir / ".cache"

        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as direct_loader:
            direct_events = direct_loader.load_events(0, 20)

        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            event_load_mode="cached",
            cache_dir=str(cache_dir),
        ) as first_loader:
            first_events = first_loader.load_events(0, 20)
            _assert_raw_events_equal(first_events, direct_events)

        metadata_files = sorted(cache_dir.rglob("metadata.json"))
        assert len(metadata_files) == 1

        with MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            event_load_mode="cached",
            cache_dir=str(cache_dir),
        ) as second_loader:
            second_events = second_loader.load_events(0, 20)
            _assert_raw_events_equal(second_events, direct_events)

    def test_lazy_image_mode_releases_on_close(self, mvsec_dir: Path) -> None:
        loader = MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            image_load_mode="lazy",
        )
        before_close = loader.load_image(0)
        assert before_close is not None
        loader.close()
        after_close = loader.load_image(0)
        assert after_close is None

    def test_lazy_image_loader_is_pickle_safe(self, mvsec_dir: Path) -> None:
        loader = MVSECDataLoader(
            str(mvsec_dir),
            SEQ,
            load_calibration=False,
            image_load_mode="lazy",
        )
        first_image = loader.load_image(0)
        assert first_image is not None

        restored_loader = pickle.loads(pickle.dumps(loader))
        restored_image = restored_loader.load_image(0)
        assert restored_image is not None
        np.testing.assert_array_equal(restored_image, first_image)
        assert restored_image.flags.writeable is False

    def test_iter_events_invalid_num_events(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            with pytest.raises(ValueError):
                list(loader.iter_events(num_events=0))
            with pytest.raises(ValueError):
                list(loader.iter_events(num_events=-1))

    def test_iter_events_invalid_time_window(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            with pytest.raises(ValueError):
                list(loader.iter_events(time_window=0.0))
            with pytest.raises(ValueError):
                list(loader.iter_events(time_window=-1.0))

    def test_iter_events_empty_sequence(self, tmp_path: Path) -> None:
        """iter_events on an empty sequence yields nothing instead of crashing."""
        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=np.empty((0, 4), dtype=np.float64))
            grp.create_dataset("image_raw_ts", data=np.array([], dtype=np.float64))
        with MVSECDataLoader(
            str(tmp_path), SEQ, load_gt_flow=False, load_calibration=False
        ) as loader:
            assert list(loader.iter_events(num_events=100)) == []
            assert list(loader.iter_events(time_window=1.0)) == []

    def test_context_manager(self, mvsec_dir: Path) -> None:
        loader = MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False)
        with loader:
            _ = loader.load_events(0, 10)
        # Data stays usable because arrays are cached during __init__.
        assert len(loader.load_events(0, 10)) == 10


# IMU tests


class TestMVSECDataLoaderIMU:
    def test_has_imu(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False, load_imu=True
        ) as loader:
            assert loader.has_imu

    def test_imu_not_loaded_by_default(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False
        ) as loader:
            assert not loader.has_imu

    def test_imu_timestamps(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False, load_imu=True
        ) as loader:
            ts = loader.imu_timestamps
            assert ts is not None
            assert len(ts) == N_IMU
            assert ts.dtype == np.float64

    def test_imu_data_shape(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False, load_imu=True
        ) as loader:
            imu = loader.imu_data
            assert imu is not None
            assert imu.shape == (N_IMU, 6)
            assert imu.dtype == np.float64

    def test_load_imu_windowed(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False, load_imu=True
        ) as loader:
            t_mid = (T_START + T_END) / 2
            result = loader.load_imu(T_START, t_mid)
            assert result is not None
            readings, timestamps = result
            assert readings.ndim == 2
            assert readings.shape[1] == 6
            assert len(readings) == len(timestamps)
            assert np.all(timestamps >= T_START)
            assert np.all(timestamps < t_mid)

    def test_load_imu_returns_none_when_not_loaded(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            result = loader.load_imu(T_START, T_END)
            assert result is None


# Odometry NPZ tests


class TestMVSECDataLoaderOdometryNPZ:
    def test_has_odometry_npz(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            assert loader.has_odometry_npz

    def test_odometry_npz_not_loaded_by_default(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False
        ) as loader:
            assert not loader.has_odometry_npz

    def test_odometry_npz_type(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            odom = loader.odometry_npz
            assert isinstance(odom, MVSECOdometryData)

    def test_odometry_npz_shapes(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            odom = loader.odometry_npz
            assert odom is not None
            assert odom.timestamps.shape == (N_ODOM_FRAMES,)
            assert odom.linear_velocity.shape == (N_ODOM_FRAMES, 3)
            assert odom.position.shape == (N_ODOM_FRAMES, 3)
            assert odom.quaternion.shape == (N_ODOM_FRAMES, 4)
            assert odom.angular_velocity.shape == (N_ODOM_FRAMES, 3)

    def test_odometry_npz_len(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            odom = loader.odometry_npz
            assert odom is not None
            assert len(odom) == N_ODOM_FRAMES

    def test_odometry_npz_dtypes(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            odom = loader.odometry_npz
            assert odom is not None
            assert odom.timestamps.dtype == np.float64
            assert odom.linear_velocity.dtype == np.float64
            assert odom.position.dtype == np.float64
            assert odom.quaternion.dtype == np.float64
            assert odom.angular_velocity.dtype == np.float64

    def test_odometry_npz_read_only(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_odometry_npz=True,
        ) as loader:
            odom = loader.odometry_npz
            assert odom is not None
            assert not odom.timestamps.flags.writeable


# GT Depth tests


class TestMVSECDataLoaderGTDepth:
    def test_has_gt_depth_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw="lazy",
        ) as loader:
            assert loader.has_gt_depth
            assert loader.has_gt_depth_raw
            assert not loader.has_gt_depth_rect

    def test_has_gt_depth_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_rect="cached",
        ) as loader:
            assert loader.has_gt_depth
            assert not loader.has_gt_depth_raw
            assert loader.has_gt_depth_rect

    def test_load_depth_raw_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw="lazy",
        ) as loader:
            depth = loader.load_depth_raw(0)
            assert depth is not None
            assert depth.shape == (HEIGHT, WIDTH)
            assert depth.dtype == np.float32

    def test_load_depth_raw_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw="cached",
        ) as loader:
            depth = loader.load_depth_raw(0)
            assert depth is not None
            assert depth.shape == (HEIGHT, WIDTH)
            assert depth.dtype == np.float32
            assert loader.depth_raw_images is not None

    def test_load_depth_rectified_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_rect="lazy",
        ) as loader:
            depth = loader.load_depth_rect(0)
            assert depth is not None
            assert depth.shape == (HEIGHT, WIDTH)
            assert depth.dtype == np.float32

    def test_depth_timestamps(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw=True,
        ) as loader:
            ts = loader.gt_depth_raw_timestamps
            assert ts is not None
            assert len(ts) == N_DEPTH_FRAMES
            assert ts.dtype == np.float64

    def test_num_gt_depth_frames(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_rect=True,
        ) as loader:
            assert loader.num_gt_depth_frames == N_DEPTH_FRAMES
            assert loader.num_gt_depth_rect_frames == N_DEPTH_FRAMES


# GT Blended images tests


class TestMVSECDataLoaderGTBlended:
    def test_has_gt_blended_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_blended="lazy",
        ) as loader:
            assert loader.has_gt_blended

    def test_load_blended_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_blended="lazy",
        ) as loader:
            img = loader.load_blended_image(0)
            assert img is not None
            assert img.shape == (HEIGHT, WIDTH, 3)
            assert img.dtype == np.uint8

    def test_load_blended_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_blended="cached",
        ) as loader:
            img = loader.load_blended_image(0)
            assert img is not None
            assert img.shape == (HEIGHT, WIDTH, 3)
            assert img.dtype == np.uint8
            assert loader.blended_images is not None


# GT Flow HDF5 tests


class TestMVSECDataLoaderGTFlowHDF5:
    def test_has_gt_flow_hdf5_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_flow_hdf5="lazy",
        ) as loader:
            assert loader.has_gt_flow_hdf5

    def test_load_flow_hdf5_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_flow_hdf5="lazy",
        ) as loader:
            flow = loader.load_flow_hdf5(0)
            assert flow is not None
            assert flow.shape == (2, HEIGHT, WIDTH)
            assert flow.dtype == np.float64

    def test_load_flow_hdf5_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_flow_hdf5="cached",
        ) as loader:
            flow = loader.load_flow_hdf5(0)
            assert flow is not None
            assert flow.shape == (2, HEIGHT, WIDTH)
            assert flow.dtype == np.float64
            assert loader.flow_hdf5_frames is not None

    def test_flow_hdf5_timestamps(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_flow_hdf5=True,
        ) as loader:
            ts = loader.gt_flow_hdf5_timestamps
            assert ts is not None
            assert len(ts) == N_DEPTH_FRAMES


# GT Poses tests


class TestMVSECDataLoaderGTPoses:
    def test_has_gt_odometry(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_odometry=True,
        ) as loader:
            assert loader.has_gt_odometry

    def test_gt_odometry_shape(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_odometry=True,
        ) as loader:
            odom = loader.gt_odometry
            assert odom is not None
            assert odom.shape == (N_ODOM_FRAMES, 4, 4)
            assert odom.dtype == np.float64

    def test_has_gt_pose(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_poses=True,
        ) as loader:
            assert loader.has_gt_pose

    def test_gt_pose_shape(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_poses=True,
        ) as loader:
            pose = loader.gt_pose
            assert pose is not None
            assert pose.shape == (N_POSE_FRAMES, 4, 4)
            assert pose.dtype == np.float64

    def test_load_nearest_pose(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_poses=True,
        ) as loader:
            t_mid = (T_START + T_END) / 2
            pose = loader.load_nearest_pose(t_mid, source="pose")
            assert pose is not None
            assert pose.shape == (4, 4)
            assert pose.dtype == np.float64

    def test_load_nearest_odometry(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_odometry=True,
        ) as loader:
            t_mid = (T_START + T_END) / 2
            pose = loader.load_nearest_pose(t_mid, source="odometry")
            assert pose is not None
            assert pose.shape == (4, 4)

    def test_load_nearest_pose_invalid_source(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_poses=True,
        ) as loader:
            with pytest.raises(ValueError, match="source"):
                loader.load_nearest_pose(T_START, source="invalid")

    def test_load_nearest_pose_returns_none_when_not_loaded(self, mvsec_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_dir), SEQ, load_calibration=False) as loader:
            result = loader.load_nearest_pose(T_START, source="pose")
            assert result is None


# Velodyne tests


class TestMVSECDataLoaderVelodyne:
    def test_has_velodyne_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne="lazy",
        ) as loader:
            assert loader.has_velodyne

    def test_has_velodyne_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne="cached",
        ) as loader:
            assert loader.has_velodyne

    def test_load_velodyne_scan_lazy(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne="lazy",
        ) as loader:
            scan = loader.load_velodyne_scan(0)
            assert scan is not None
            assert scan.shape == (N_VELODYNE_POINTS, 4)
            assert scan.dtype == np.float32

    def test_load_velodyne_scan_cached(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne="cached",
        ) as loader:
            scan = loader.load_velodyne_scan(0)
            assert scan is not None
            assert scan.shape == (N_VELODYNE_POINTS, 4)
            assert scan.dtype == np.float32
            assert loader.velodyne_scans is not None

    def test_velodyne_timestamps(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne=True,
        ) as loader:
            ts = loader.velodyne_timestamps
            assert ts is not None
            assert len(ts) == N_VELODYNE_SCANS
            assert ts.dtype == np.float64

    def test_num_velodyne_scans(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne=True,
        ) as loader:
            assert loader.num_velodyne_scans == N_VELODYNE_SCANS


# Close and resource management tests


class TestMVSECDataLoaderClose:
    def test_close_releases_lazy_velodyne(self, mvsec_full_dir: Path) -> None:
        loader = MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_velodyne="lazy",
        )
        # Should work before close
        scan = loader.load_velodyne_scan(0)
        assert scan is not None
        loader.close()
        # Lazy reference should be cleared
        result = loader.load_velodyne_scan(0)
        assert result is None

    def test_close_releases_lazy_depth(self, mvsec_full_dir: Path) -> None:
        loader = MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw="lazy",
        )
        depth = loader.load_depth_raw(0)
        assert depth is not None
        loader.close()
        result = loader.load_depth_raw(0)
        assert result is None

    def test_cached_data_survives_close(self, mvsec_full_dir: Path) -> None:
        loader = MVSECDataLoader(
            str(mvsec_full_dir),
            SEQ,
            load_calibration=False,
            load_gt_flow=False,
            load_gt_depth_raw="cached",
        )
        loader.close()
        # Cached data should still be accessible
        depth = loader.load_depth_raw(0)
        assert depth is not None


# Backward compatibility tests


class TestMVSECDataLoaderBackwardCompat:
    def test_old_signature(self, mvsec_dir: Path) -> None:
        """Existing constructor signature still works identically."""
        with MVSECDataLoader(str(mvsec_dir), SEQ, "left", True, False) as loader:
            assert loader.has_gt_flow
            assert loader.num_events == N_EVENTS

    def test_default_profile_is_lean(self, mvsec_full_dir: Path) -> None:
        with MVSECDataLoader(str(mvsec_full_dir), SEQ) as loader:
            assert not loader.has_gt_flow
            assert not loader.has_calibration

    def test_new_modalities_off_by_default(self, mvsec_full_dir: Path) -> None:
        """New modalities are not loaded unless explicitly requested."""
        with MVSECDataLoader(
            str(mvsec_full_dir), SEQ, load_calibration=False, load_gt_flow=False
        ) as loader:
            assert not loader.has_imu
            assert not loader.has_odometry_npz
            assert not loader.has_gt_odometry
            assert not loader.has_gt_pose
            assert not loader.has_gt_depth
            assert not loader.has_gt_blended
            assert not loader.has_gt_flow_hdf5
            assert not loader.has_velodyne
