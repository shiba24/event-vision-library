"""Tests for MVSECDataset using synthetic MVSEC-format files."""

# mypy: disable-error-code=no-untyped-def

import h5py
import numpy as np
import pytest

from evlib.dataloaders import LoadingType
from evlib.dataloaders import MVSECOdometryData
from evlib.datasets import mvsec_collate_fn
from evlib.datasets._base import BlockAccessDataset
from evlib.datasets._base import EventDataset
from evlib.datasets._base import IteratorAccessDataset
from evlib.datasets.mvsec import MVSECDataset
from evlib.datasets.mvsec import MVSECIterator
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
N_VELODYNE_POINTS = 200
SEQ = "indoor_flying1"
CATEGORY = "indoor_flying"
T_START = 1504645177.0
T_END = 1504645247.0


def _make_hdf5(path, n_events=N_EVENTS, n_frames=N_FRAMES, camera="left"):
    """Create minimal MVSEC format HDF5 file matching real layout."""
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


def _add_imu_to_hdf5(path, camera="left"):
    rng = np.random.RandomState(3)
    with h5py.File(path, "a") as f:
        grp = f[f"davis/{camera}"]
        grp.create_dataset("imu", data=rng.normal(0, 1, (N_IMU, 6)))
        grp.create_dataset("imu_ts", data=np.linspace(T_START, T_END, N_IMU))


def _add_velodyne_to_hdf5(path):
    rng = np.random.RandomState(5)
    with h5py.File(path, "a") as f:
        vel_grp = f.create_group("velodyne")
        scans = rng.uniform(-10, 10, (N_VELODYNE_SCANS, N_VELODYNE_POINTS, 4)).astype(np.float32)
        vel_grp.create_dataset("scans", data=scans)
        vel_grp.create_dataset("scans_ts", data=np.linspace(T_START, T_END, N_VELODYNE_SCANS))


def _make_gt_flow_npz(path, n_frames=N_GT_FRAMES):
    rng = np.random.RandomState(0)
    x_flow_dist = rng.uniform(-5, 5, (n_frames, HEIGHT, WIDTH))
    y_flow_dist = rng.uniform(-5, 5, (n_frames, HEIGHT, WIDTH))
    timestamps = np.linspace(T_START, T_END, n_frames)
    np.savez(path, x_flow_dist=x_flow_dist, y_flow_dist=y_flow_dist, timestamps=timestamps)


def _make_constant_gt_flow_npz(path, x_value=4.0, y_value=-2.0, n_frames=4):
    """Create GT flow NPZ with a constant displacement in every frame."""
    x_flow_dist = np.full((n_frames, HEIGHT, WIDTH), x_value, dtype=np.float32)
    y_flow_dist = np.full((n_frames, HEIGHT, WIDTH), y_value, dtype=np.float32)
    timestamps = np.linspace(T_START, T_END, n_frames)
    np.savez(path, x_flow_dist=x_flow_dist, y_flow_dist=y_flow_dist, timestamps=timestamps)


def _make_odom_npz(path, n_frames=N_ODOM_FRAMES):
    rng = np.random.RandomState(11)
    np.savez(
        path,
        timestamps=np.linspace(T_START, T_END, n_frames),
        lin_vel=rng.normal(0, 1, (n_frames, 3)),
        pos=rng.normal(0, 1, (n_frames, 3)),
        quat=rng.normal(0, 1, (n_frames, 4)),
        ang_vel=rng.normal(0, 1, (n_frames, 3)),
    )


def _make_gt_hdf5(path, camera="left"):
    """Create synthetic gt.hdf5 with depth, blended, flow, odometry, poses."""
    rng = np.random.RandomState(7)
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"davis/{camera}")

        depth_raw = rng.uniform(0.5, 50.0, (N_DEPTH_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
        grp.create_dataset("depth_image_raw", data=depth_raw)
        grp.create_dataset("depth_image_raw_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        depth_rect = rng.uniform(0.5, 50.0, (N_DEPTH_FRAMES, HEIGHT, WIDTH)).astype(np.float32)
        grp.create_dataset("depth_image_rect", data=depth_rect)
        grp.create_dataset("depth_image_rect_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        blended = rng.randint(0, 256, (N_DEPTH_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
        grp.create_dataset("blended_image_rect", data=blended)
        grp.create_dataset(
            "blended_image_rect_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES)
        )

        flow = rng.uniform(-5, 5, (N_DEPTH_FRAMES, 2, HEIGHT, WIDTH))
        grp.create_dataset("flow_dist", data=flow)
        grp.create_dataset("flow_dist_ts", data=np.linspace(T_START, T_END, N_DEPTH_FRAMES))

        odom = np.tile(np.eye(4), (N_ODOM_FRAMES, 1, 1))
        grp.create_dataset("odometry", data=odom)
        grp.create_dataset("odometry_ts", data=np.linspace(T_START, T_END, N_ODOM_FRAMES))

        pose = np.tile(np.eye(4), (N_POSE_FRAMES, 1, 1))
        grp.create_dataset("pose", data=pose)
        grp.create_dataset("pose_ts", data=np.linspace(T_START, T_END, N_POSE_FRAMES))


def _assert_raw_events_equal(left: RawEvents, right: RawEvents) -> None:
    np.testing.assert_array_equal(left.x, right.x)
    np.testing.assert_array_equal(left.y, right.y)
    np.testing.assert_array_equal(left.timestamp, right.timestamp)
    np.testing.assert_array_equal(left.polarity, right.polarity)


def _make_calibration_maps_in_root(root, category, camera="left"):
    rng = np.random.RandomState(1)
    for axis in ("x", "y"):
        data = rng.uniform(0, WIDTH if axis == "x" else HEIGHT, (HEIGHT, WIDTH))
        np.savetxt(root / f"{category}_{camera}_{axis}_map.txt", data)


def _make_calibration_maps_in_subdir(root, category, camera="left"):
    calib_dir = root / f"{category}_calib"
    calib_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(1)
    for axis in ("x", "y"):
        data = rng.uniform(0, WIDTH if axis == "x" else HEIGHT, (HEIGHT, WIDTH))
        np.savetxt(calib_dir / f"{category}_{camera}_{axis}_map.txt", data)


@pytest.fixture()
def mvsec_dir(tmp_path):
    """Synthetic MVSEC directory with calib in root."""
    _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
    _make_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")
    _make_calibration_maps_in_root(tmp_path, CATEGORY)
    return tmp_path


@pytest.fixture()
def mvsec_dir_calib_subdir(tmp_path):
    """Synthetic MVSEC directory with calib in subdirectory."""
    _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
    _make_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")
    _make_calibration_maps_in_subdir(tmp_path, CATEGORY)
    return tmp_path


@pytest.fixture()
def mvsec_dir_no_gt(tmp_path):
    """MVSEC directory without GT flow or calibration."""
    _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
    return tmp_path


@pytest.fixture()
def mvsec_full_dir(tmp_path):
    """Synthetic MVSEC directory with all modalities."""
    data_path = tmp_path / f"{SEQ}_data.hdf5"
    _make_hdf5(data_path)
    _add_imu_to_hdf5(data_path)
    _add_velodyne_to_hdf5(data_path)
    _make_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")
    _make_odom_npz(tmp_path / f"{SEQ}_odom.npz")
    _make_gt_hdf5(tmp_path / f"{SEQ}_gt.hdf5")
    _make_calibration_maps_in_root(tmp_path, CATEGORY)
    return tmp_path


class TestMVSECDataset:
    def test_load_events(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            events = ds.load_events(0, 100)
            assert isinstance(events, RawEvents)
            assert len(events) == 100
            assert events.x.dtype == np.int16
            assert events.y.dtype == np.int16
            assert events.timestamp.dtype == np.float64
            assert events.polarity.dtype == np.bool_

    def test_polarity_conversion(self, mvsec_dir):
        """MVSEC stores polarity as -1/+1; loader converts to bool."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            events = ds.load_events(0, N_EVENTS)
            assert set(np.unique(events.polarity)).issubset({True, False})
            assert np.any(events.polarity) and np.any(~events.polarity)

    def test_coordinate_bounds(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            events = ds.load_events(0, N_EVENTS)
            assert np.all(events.x >= 0) and np.all(events.x < WIDTH)
            assert np.all(events.y >= 0) and np.all(events.y < HEIGHT)

    def test_coordinate_convention(self, mvsec_dir):
        """events.x is width direction, events.y is height direction.
        as_numpy() returns [y, x, t, p]."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            events = ds.load_events(0, 10)
            arr = events.as_numpy()
            assert arr.shape == (10, 4)
            np.testing.assert_array_equal(arr[:, 0], events.y.astype(np.float64))
            np.testing.assert_array_equal(arr[:, 1], events.x.astype(np.float64))

    def test_num_events(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            assert ds.num_events == N_EVENTS

    def test_time_to_index(self, mvsec_dir):
        """time_to_index returns the last event strictly before t."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            t_mid = (T_START + T_END) / 2
            idx = ds.time_to_index(t_mid)
            assert -1 <= idx < N_EVENTS
            if idx >= 0:
                assert ds.index_to_time(idx) < t_mid
            if idx + 1 < N_EVENTS:
                assert ds.index_to_time(idx + 1) >= t_mid

    def test_index_to_time(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            t0 = ds.index_to_time(0)
            t_last = ds.index_to_time(N_EVENTS - 1)
            assert T_START <= t0 <= t_last <= T_END

    def test_times_to_indices_matches_scalar(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            query_times = np.array([T_START, (T_START + T_END) / 2, T_END], dtype=np.float64)
            bulk_indices = ds.times_to_indices(query_times)
            scalar_indices = np.array([ds.time_to_index(float(t)) for t in query_times])
            np.testing.assert_array_equal(bulk_indices, scalar_indices)

    def test_indices_to_times_matches_scalar(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            query_indices = np.array([0, N_EVENTS // 2, N_EVENTS - 1], dtype=np.int64)
            bulk_timestamps = ds.indices_to_times(query_indices)
            scalar_timestamps = np.array([ds.index_to_time(int(i)) for i in query_indices])
            np.testing.assert_array_equal(bulk_timestamps, scalar_timestamps)

    def test_time_index_roundtrip(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            idx = 42
            t = ds.index_to_time(idx)
            recovered = ds.time_to_index(t)
            assert recovered == idx - 1

    def test_load_optical_flow(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=True) as ds:
            assert ds.has_gt_flow
            gt_ts = ds.gt_time_list()
            flow = ds.load_optical_flow(gt_ts[0], gt_ts[2])
            assert flow.shape == (HEIGHT, WIDTH, 2)

    def test_load_optical_flow_within_single_gt_interval_scales_constant_flow(self, tmp_path):
        _make_hdf5(tmp_path / f"{SEQ}_data.hdf5")
        _make_constant_gt_flow_npz(tmp_path / f"{SEQ}_gt_flow_dist.npz")

        with MVSECDataset(str(tmp_path), SEQ, load_gt_flow=True) as ds:
            gt_ts = ds.gt_time_list()
            gt_interval_duration = gt_ts[1] - gt_ts[0]
            t_start = gt_ts[0] + 0.1 * gt_interval_duration
            t_end = gt_ts[0] + 0.6 * gt_interval_duration
            requested_duration = t_end - t_start
            scale_factor = requested_duration / gt_interval_duration

            flow = ds.load_optical_flow(float(t_start), float(t_end))

            np.testing.assert_allclose(flow[..., 0], 4.0 * scale_factor)
            np.testing.assert_allclose(flow[..., 1], -2.0 * scale_factor)

    def test_get_gt_timestamps(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=True) as ds:
            t_before, t_after = ds.get_gt_timestamps(N_EVENTS // 2)
            event_t = ds.index_to_time(N_EVENTS // 2)
            if t_before is not None:
                assert t_before <= event_t
            if t_after is not None:
                assert t_after > event_t

    def test_gt_time_list(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=True) as ds:
            ts = ds.gt_time_list()
            assert len(ts) == N_GT_FRAMES

    def test_frame_timestamps(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            ft = ds.frame_timestamps
            assert ft is not None
            assert len(ft) == N_FRAMES
            assert ft[0] >= T_START
            assert ft[-1] <= T_END

    def test_frame_helpers_are_exposed(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            assert ds.num_frames == N_FRAMES
            assert len(ds) == N_FRAMES
            frame_timestamps = ds.frame_timestamps
            frame_event_indices = ds.frame_event_indices
            assert frame_timestamps is not None
            assert frame_event_indices is not None
            assert len(frame_event_indices) == N_FRAMES
            assert ds.find_nearest_frame_index(float(frame_timestamps[7])) == 7

    def test_event_and_image_modes_are_exposed(self, mvsec_dir):
        cache_dir = mvsec_dir / ".cache"
        with MVSECDataset(
            str(mvsec_dir),
            SEQ,
            event_load_mode=LoadingType.LAZY,
            image_load_mode=LoadingType.LAZY,
            cache_dir=str(cache_dir),
        ) as ds:
            assert ds.event_load_mode is LoadingType.LAZY
            assert ds.image_load_mode is LoadingType.LAZY
            assert ds.has_images

    def test_context_manager(self, mvsec_dir):
        ds = MVSECDataset(str(mvsec_dir), SEQ)
        with ds:
            _ = ds.load_events(0, 10)
        assert len(ds.load_events(0, 10)) == 10

    def test_repr_exposes_identity(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            text = repr(ds)
            assert "MVSECDataset" in text
            assert str(mvsec_dir) in text
            assert SEQ in text
            assert "left" in text

    def test_default_profile_is_lean(self, mvsec_full_dir):
        with MVSECDataset(str(mvsec_full_dir), SEQ) as ds:
            assert not ds.has_gt_flow
            assert not ds.has_calibration

    def test_missing_gt_flow(self, mvsec_dir_no_gt):
        with MVSECDataset(str(mvsec_dir_no_gt), SEQ) as ds:
            assert not ds.has_gt_flow
            assert not ds.has_calibration
            t_before, t_after = ds.get_gt_timestamps(0)
            assert t_before is None and t_after is None

    def test_missing_gt_flow_raises(self, mvsec_dir_no_gt):
        with MVSECDataset(str(mvsec_dir_no_gt), SEQ) as ds:
            with pytest.raises(RuntimeError):
                ds.load_optical_flow(T_START, T_END)
            with pytest.raises(RuntimeError):
                ds.gt_time_list()

    def test_calibration_in_root(self, mvsec_dir):
        """Calibration maps found when placed directly in root."""
        with MVSECDataset(str(mvsec_dir), SEQ, load_calibration=True) as ds:
            assert ds.has_calibration

    def test_calibration_in_subdir(self, mvsec_dir_calib_subdir):
        """Calibration maps found when placed in {category}_calib/ subdir."""
        with MVSECDataset(str(mvsec_dir_calib_subdir), SEQ, load_calibration=True) as ds:
            assert ds.has_calibration

    def test_undistort_events(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ, load_calibration=True) as ds:
            assert ds.has_calibration
            events = ds.load_events(0, 50)
            undistorted = ds.undistort_events(events)
            assert isinstance(undistorted, RawEvents)
            assert len(undistorted) == 50

    def test_undistort_without_calibration_raises(self, mvsec_dir_no_gt):
        with MVSECDataset(str(mvsec_dir_no_gt), SEQ) as ds:
            events = ds.load_events(0, 10)
            with pytest.raises(RuntimeError):
                ds.undistort_events(events)

    def test_invalid_camera(self, mvsec_dir):
        with pytest.raises(ValueError, match="camera"):
            MVSECDataset(str(mvsec_dir), SEQ, camera="center")

    def test_len_returns_frame_count(self, mvsec_dir):
        """len(ds) returns number of frames, not events."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            assert len(ds) == N_FRAMES

    def test_getitem_basic(self, mvsec_dir):
        """ds[0] returns dict with the expected schema."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            sample = ds[0]
            assert isinstance(sample, dict)
            expected_keys = {
                "events",
                "timestamp",
                "image",
                "flow",
                "imu",
                "depth",
                "depth_rect",
                "blended",
                "velodyne",
                "pose",
            }
            assert set(sample) == expected_keys
            assert isinstance(sample["events"], RawEvents)
            assert isinstance(sample["timestamp"], float)

    def test_getitem_with_image(self, mvsec_dir):
        """'image' key is correct shape and present."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            sample = ds[0]
            assert "image" in sample
            assert sample["image"].shape == (HEIGHT, WIDTH)

    def test_getitem_with_flow(self, mvsec_dir):
        """'flow' key contains GT when timestamps overlap the frame window."""
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=True) as ds:
            found_flow = False
            for i in range(1, len(ds)):
                sample = ds[i]
                if sample["flow"] is not None:
                    assert sample["flow"].shape == (HEIGHT, WIDTH, 2)
                    found_flow = True
                    break
            assert found_flow, "Expected at least one frame with GT flow"

    def test_getitem_with_gt_flow_populates_short_interval_flow(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=True) as ds:
            sample = ds[1]
            assert sample["flow"] is not None

    def test_getitem_without_gt_uses_none_flow(self, mvsec_dir_no_gt):
        with MVSECDataset(str(mvsec_dir_no_gt), SEQ) as ds:
            sample = ds[0]
            assert "flow" in sample
            assert sample["flow"] is None

    def test_getitem_without_images_uses_none_image(self, tmp_path):
        rng = np.random.RandomState(42)
        n_events = 50
        x = rng.randint(0, WIDTH, n_events).astype(np.float64)
        y = rng.randint(0, HEIGHT, n_events).astype(np.float64)
        t = np.sort(rng.uniform(T_START, T_END, n_events))
        p = rng.choice([-1.0, 1.0], n_events)
        events = np.stack([x, y, t, p], axis=1)

        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=events)
            grp.create_dataset("image_raw_ts", data=np.linspace(T_START, T_END, 5))

        with MVSECDataset(str(tmp_path), SEQ, load_gt_flow=False, load_calibration=False) as ds:
            sample = ds[0]
            assert "image" in sample
            assert sample["image"] is None

    def test_getitem_negative_index(self, mvsec_dir):
        """ds[-1] returns the last frame."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            sample_last = ds[-1]
            sample_explicit = ds[N_FRAMES - 1]
            assert sample_last["timestamp"] == sample_explicit["timestamp"]

    def test_getitem_out_of_range(self, mvsec_dir):
        """Out of range index raises IndexError."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            with pytest.raises(IndexError):
                ds[N_FRAMES]
            with pytest.raises(IndexError):
                ds[-N_FRAMES - 1]

    def test_getitem_empty_events(self, tmp_path):
        """Frame interval with no events returns empty RawEvents."""
        rng = np.random.RandomState(99)
        n_events = 10
        t_narrow_end = T_START + (T_END - T_START) * 0.01
        x = rng.randint(0, WIDTH, n_events).astype(np.float64)
        y = rng.randint(0, HEIGHT, n_events).astype(np.float64)
        t = np.sort(rng.uniform(T_START, t_narrow_end, n_events))
        p = rng.choice([-1.0, 1.0], n_events)
        events = np.stack([x, y, t, p], axis=1)

        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=events)
            frame_ts = np.linspace(T_START, T_END, N_FRAMES)
            grp.create_dataset("image_raw_ts", data=frame_ts)

        with MVSECDataset(str(tmp_path), SEQ, load_gt_flow=False, load_calibration=False) as ds:
            sample = ds[-1]
            assert isinstance(sample["events"], RawEvents)
            assert len(sample["events"]) == 0

    def test_get_events_by_time(self, mvsec_dir):
        """Convenience temporal access returns events in the window."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            t_mid = (T_START + T_END) / 2
            events = ds.get_events_by_time(T_START, t_mid)
            assert isinstance(events, RawEvents)
            assert len(events) > 0
            assert np.all(events.timestamp >= T_START)
            assert np.all(events.timestamp < t_mid)

    def test_iter_events_by_count(self, mvsec_dir):
        """Chunk iteration by event count."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            chunks = list(ds.iter_events(num_events=100))
            assert len(chunks) > 0
            total = sum(len(c) for c in chunks)
            assert total == N_EVENTS
            for c in chunks[:-1]:
                assert len(c) == 100

    def test_iter_events_by_time(self, mvsec_dir):
        """Chunk iteration by time window."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            window = (T_END - T_START) / 5
            chunks = list(ds.iter_events(time_window=window))
            assert len(chunks) > 0
            total = sum(len(c) for c in chunks)
            assert total == N_EVENTS

    def test_load_image(self, mvsec_dir):
        """Explicit image loading returns correct shape."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            img = ds.load_image(0)
            assert img is not None
            assert img.shape == (HEIGHT, WIDTH)
            assert img.dtype == np.uint8

    def test_lazy_modes_match_cached_sample(self, mvsec_dir):
        cache_dir = mvsec_dir / ".cache"
        with MVSECDataset(str(mvsec_dir), SEQ, load_gt_flow=False) as cached_ds:
            with MVSECDataset(
                str(mvsec_dir),
                SEQ,
                load_gt_flow=False,
                event_load_mode="lazy",
                image_load_mode="lazy",
                cache_dir=str(cache_dir),
            ) as lazy_ds:
                cached_sample = cached_ds[5]
                lazy_sample = lazy_ds[5]

                _assert_raw_events_equal(cached_sample["events"], lazy_sample["events"])
                assert cached_sample["timestamp"] == lazy_sample["timestamp"]
                np.testing.assert_array_equal(cached_sample["image"], lazy_sample["image"])

    def test_load_image_none_when_absent(self, tmp_path):
        """load_image returns None when images not in HDF5."""
        rng = np.random.RandomState(42)
        n_events = 50
        x = rng.randint(0, WIDTH, n_events).astype(np.float64)
        y = rng.randint(0, HEIGHT, n_events).astype(np.float64)
        t = np.sort(rng.uniform(T_START, T_END, n_events))
        p = rng.choice([-1.0, 1.0], n_events)
        events = np.stack([x, y, t, p], axis=1)

        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=events)
            grp.create_dataset("image_raw_ts", data=np.linspace(T_START, T_END, 5))

        with MVSECDataset(str(tmp_path), SEQ, load_gt_flow=False, load_calibration=False) as ds:
            assert ds.load_image(0) is None

    def test_isinstance_block_access(self, mvsec_dir):
        """MVSECDataset is a BlockAccessDataset and EventDataset."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            assert isinstance(ds, BlockAccessDataset)
            assert isinstance(ds, EventDataset)


class TestMVSECDatasetNewModalities:
    def test_getitem_with_imu(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir), SEQ, load_imu=True, load_gt_flow=False, load_calibration=False
        ) as ds:
            sample = ds[5]
            assert "imu" in sample
            if sample["imu"] is not None:
                readings, timestamps = sample["imu"]
                assert readings.ndim == 2
                assert readings.shape[1] == 6
                assert len(readings) == len(timestamps)

    def test_getitem_with_depth(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_depth_raw="lazy",
            load_gt_depth_rect="lazy",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            sample = ds[5]
            assert "depth" in sample
            assert "depth_rect" in sample
            assert sample["depth"] is not None
            assert sample["depth"].shape == (HEIGHT, WIDTH)
            assert sample["depth"].dtype == np.float32
            assert sample["depth_rect"] is not None
            assert sample["depth_rect"].shape == (HEIGHT, WIDTH)

    def test_getitem_with_blended(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_blended="cached",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            sample = ds[5]
            assert "blended" in sample
            assert sample["blended"] is not None
            assert sample["blended"].shape == (HEIGHT, WIDTH, 3)
            assert sample["blended"].dtype == np.uint8
            assert ds.blended_images is not None

    def test_getitem_with_velodyne(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_velodyne="lazy",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            sample = ds[5]
            assert "velodyne" in sample
            assert sample["velodyne"] is not None
            assert sample["velodyne"].shape == (N_VELODYNE_POINTS, 4)
            assert sample["velodyne"].dtype == np.float32

    def test_getitem_with_pose(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_poses=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            sample = ds[5]
            assert "pose" in sample
            assert sample["pose"] is not None
            assert sample["pose"].shape == (4, 4)
            assert sample["pose"].dtype == np.float64

    def test_getitem_new_modalities_none_when_not_loaded(self, mvsec_dir):
        """New modality keys are present but None when not requested."""
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            sample = ds[0]
            assert sample["imu"] is None
            assert sample["depth"] is None
            assert sample["depth_rect"] is None
            assert sample["blended"] is None
            assert sample["velodyne"] is None
            assert sample["pose"] is None

    def test_odometry_npz_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_odometry_npz=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.has_odometry_npz
            odom = ds.odometry_npz
            assert isinstance(odom, MVSECOdometryData)
            assert odom.position.shape == (N_ODOM_FRAMES, 3)

    def test_gt_odometry_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_odometry=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.has_gt_odometry
            odom = ds.gt_odometry
            assert odom is not None
            assert odom.shape == (N_ODOM_FRAMES, 4, 4)

    def test_gt_pose_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_poses=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.has_gt_pose
            pose = ds.gt_pose
            assert pose is not None
            assert pose.shape == (N_POSE_FRAMES, 4, 4)

    def test_load_nearest_pose_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_poses=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            t_mid = (T_START + T_END) / 2
            pose = ds.load_nearest_pose(t_mid)
            assert pose is not None
            assert pose.shape == (4, 4)

    def test_gt_depth_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_depth_raw="cached",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.has_gt_depth_raw
            ts = ds.gt_depth_raw_timestamps
            assert ts is not None
            depth = ds.load_depth_raw(0)
            assert depth is not None
            assert depth.shape == (HEIGHT, WIDTH)
            assert ds.depth_raw_images is not None

    def test_gt_depth_counts_are_exposed(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_depth_raw="lazy",
            load_gt_depth_rect="lazy",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.num_gt_depth_raw_frames == N_DEPTH_FRAMES
            assert ds.num_gt_depth_rect_frames == N_DEPTH_FRAMES
            assert ds.num_gt_depth_frames == N_DEPTH_FRAMES

    def test_gt_blended_timestamps_are_exposed(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_blended="lazy",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            timestamps = ds.gt_blended_timestamps
            assert timestamps is not None
            assert len(timestamps) == N_DEPTH_FRAMES

    def test_gt_flow_hdf5_timestamps_are_exposed(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_flow_hdf5="lazy",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            timestamps = ds.gt_flow_hdf5_timestamps
            assert timestamps is not None
            assert len(timestamps) == N_DEPTH_FRAMES

    def test_velodyne_delegation(self, mvsec_full_dir):
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_velodyne="cached",
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            assert ds.has_velodyne
            assert ds.num_velodyne_scans == N_VELODYNE_SCANS
            scan = ds.load_velodyne_scan(0)
            assert scan is not None
            assert ds.velodyne_scans is not None


class TestMVSECIterator:
    def test_iterator_basic(self, mvsec_dir):
        """MVSECIterator yields dicts with 'events' key."""
        with MVSECIterator(str(mvsec_dir), SEQ) as it:
            samples = list(it)
            assert len(samples) == N_FRAMES
            for s in samples:
                assert isinstance(s, dict)
                assert "events" in s
                assert isinstance(s["events"], RawEvents)

    def test_iterator_reset(self, mvsec_dir):
        """Reset allows reiteration."""
        with MVSECIterator(str(mvsec_dir), SEQ) as it:
            first_pass = list(it)
            it.reset()
            second_pass = list(it)
            assert len(first_pass) == len(second_pass)
            assert first_pass[0]["timestamp"] == second_pass[0]["timestamp"]

    def test_iterator_stop(self, mvsec_dir):
        """Raises StopIteration at end."""
        with MVSECIterator(str(mvsec_dir), SEQ) as it:
            for _ in it:
                pass
            with pytest.raises(StopIteration):
                next(it)

    def test_isinstance_iterator_access(self, mvsec_dir):
        """MVSECIterator is an IteratorAccessDataset and EventDataset."""
        with MVSECIterator(str(mvsec_dir), SEQ) as it:
            assert isinstance(it, IteratorAccessDataset)
            assert isinstance(it, EventDataset)

    def test_iterator_exposes_camera_and_repr(self, mvsec_dir):
        with MVSECIterator(str(mvsec_dir), SEQ) as it:
            assert it.camera == "left"
            text = repr(it)
            assert "MVSECIterator" in text
            assert str(mvsec_dir) in text
            assert SEQ in text
            assert "left" in text

    def test_iterator_with_new_modalities(self, mvsec_full_dir):
        """MVSECIterator forwards new modality kwargs."""
        with MVSECIterator(
            str(mvsec_full_dir),
            SEQ,
            load_gt_depth_raw="lazy",
            load_gt_poses=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as it:
            sample = next(it)
            assert "depth" in sample
            assert "pose" in sample


class TestMVSECCollate:
    def test_collate_returns_lists_for_variable_fields(self, mvsec_dir):
        with MVSECDataset(str(mvsec_dir), SEQ) as ds:
            batch = mvsec_collate_fn([ds[0], ds[1]])

        expected_keys = {
            "events",
            "timestamp",
            "image",
            "flow",
            "imu",
            "depth",
            "depth_rect",
            "blended",
            "velodyne",
            "pose",
        }
        assert set(batch) == expected_keys
        assert isinstance(batch["events"], list)
        assert len(batch["events"]) == 2
        assert all(isinstance(events, RawEvents) for events in batch["events"])
        assert isinstance(batch["timestamp"], np.ndarray)
        assert batch["timestamp"].shape == (2,)
        assert isinstance(batch["image"], list)
        assert len(batch["image"]) == 2
        assert isinstance(batch["flow"], list)
        assert len(batch["flow"]) == 2

    def test_collate_preserves_variable_length_events(self):
        batch = mvsec_collate_fn(
            [
                {
                    "events": RawEvents(
                        x=np.array([0], dtype=np.int16),
                        y=np.array([1], dtype=np.int16),
                        timestamp=np.array([1.0], dtype=np.float64),
                        polarity=np.array([True], dtype=np.bool_),
                    ),
                    "timestamp": 1.0,
                    "image": None,
                    "flow": None,
                },
                {
                    "events": RawEvents(
                        x=np.array([0, 1, 2], dtype=np.int16),
                        y=np.array([1, 2, 3], dtype=np.int16),
                        timestamp=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                        polarity=np.array([True, False, True], dtype=np.bool_),
                    ),
                    "timestamp": 3.0,
                    "image": None,
                    "flow": None,
                },
            ]
        )

        assert [len(events) for events in batch["events"]] == [1, 3]

    def test_collate_empty_batch_raises(self):
        with pytest.raises(ValueError, match="batch must not be empty"):
            mvsec_collate_fn([])

    def test_collate_keeps_optional_fields_as_lists(self, tmp_path):
        rng = np.random.RandomState(7)
        n_events = 20
        x = rng.randint(0, WIDTH, n_events).astype(np.float64)
        y = rng.randint(0, HEIGHT, n_events).astype(np.float64)
        t = np.sort(rng.uniform(T_START, T_END, n_events))
        p = rng.choice([-1.0, 1.0], n_events)
        events = np.stack([x, y, t, p], axis=1)

        with h5py.File(tmp_path / f"{SEQ}_data.hdf5", "w") as f:
            grp = f.create_group("davis/left")
            grp.create_dataset("events", data=events)
            grp.create_dataset("image_raw_ts", data=np.linspace(T_START, T_END, 4))

        with MVSECDataset(str(tmp_path), SEQ, load_gt_flow=False, load_calibration=False) as ds:
            batch = mvsec_collate_fn([ds[0], ds[1]])

        assert batch["image"] == [None, None]
        assert batch["flow"] == [None, None]

    def test_collate_with_new_modalities(self, mvsec_full_dir):
        """Collate correctly handles new modality keys."""
        with MVSECDataset(
            str(mvsec_full_dir),
            SEQ,
            load_gt_depth_raw="cached",
            load_gt_poses=True,
            load_gt_flow=False,
            load_calibration=False,
        ) as ds:
            batch = mvsec_collate_fn([ds[0], ds[1]])

        assert "depth" in batch
        assert isinstance(batch["depth"], list)
        assert len(batch["depth"]) == 2
        assert "pose" in batch
        assert isinstance(batch["pose"], list)
        assert len(batch["pose"]) == 2
