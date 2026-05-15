"""Tests for the DSEC dataloader."""

import builtins
import pickle  # noqa: S403
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import cv2
import h5py
import numpy as np
import pytest

import evlib.dataloaders._dsec as dsec_module
from evlib.dataloaders._dsec import DSECDataLoader
from evlib.dataloaders._dsec import _load_flow_timestamps
from evlib.dataloaders._dsec import _read_disparity_png
from evlib.dataloaders._dsec import _read_flow_png
from evlib.types import RawEvents


HEIGHT, WIDTH = 480, 640
N_EVENTS = 2000
N_IMAGES = 10
N_FLOW = 8
N_DISP = 6
N_IMU = 5
N_LIDAR = 3
SEQ = "zurich_city_01_a"
DRIVE_PREFIX = "zurich_city_01"
T_OFFSET_US = 1_600_000_000_000
T_START_REL_US = 0
T_END_REL_US = 5_000_000


def _make_events_h5(path: Path) -> None:
    rng = np.random.RandomState(42)

    x = rng.randint(0, WIDTH, N_EVENTS, dtype=np.uint16)
    y = rng.randint(0, HEIGHT, N_EVENTS, dtype=np.uint16)
    t = np.sort(rng.randint(T_START_REL_US, T_END_REL_US, N_EVENTS, dtype=np.int64))
    p = rng.choice([True, False], N_EVENTS)

    max_ms = int(T_END_REL_US / 1000) + 1
    ms_to_idx = np.zeros(max_ms, dtype=np.int64)
    for ms in range(max_ms):
        ms_to_idx[ms] = int(np.searchsorted(t, ms * 1000, side="left"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ev = f.create_group("events")
        ev.create_dataset("x", data=x)
        ev.create_dataset("y", data=y)
        ev.create_dataset("t", data=t)
        ev.create_dataset("p", data=p)
        f.create_dataset("ms_to_idx", data=ms_to_idx)
        f.create_dataset("t_offset", data=np.int64(T_OFFSET_US))


def _make_rectify_map_h5(path: Path) -> None:
    rng = np.random.RandomState(7)
    base_y, base_x = np.mgrid[:HEIGHT, :WIDTH]
    rectify_map = np.stack([base_x, base_y], axis=-1).astype(np.float64)
    rectify_map += rng.normal(0, 0.5, rectify_map.shape)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("rectify_map", data=rectify_map)


def _make_image_files(
    img_dir: Path,
    ts_path: Path,
    exposure_ts_path: Optional[Path] = None,
    n_images: int = N_IMAGES,
) -> None:
    rng = np.random.RandomState(11)

    img_dir.mkdir(parents=True, exist_ok=True)
    ts_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.linspace(
        T_OFFSET_US + 100_000,
        T_OFFSET_US + T_END_REL_US - 100_000,
        n_images,
        dtype=np.int64,
    )

    with open(ts_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

    if exposure_ts_path is not None:
        exposure_ts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exposure_ts_path, "w") as f:
            f.write("# exposure_start_us, exposure_end_us\n")
            for ts in timestamps:
                f.write(f"{ts - 1_000}, {ts + 1_000}\n")

    for i in range(n_images):
        img = rng.randint(0, 256, (100, 150), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i:06d}.png"), img)


def _make_flow_files(flow_dir: Path, ts_path: Path, n_flow: int = N_FLOW) -> None:
    rng = np.random.RandomState(13)

    flow_dir.mkdir(parents=True, exist_ok=True)
    ts_path.parent.mkdir(parents=True, exist_ok=True)

    starts = np.linspace(
        T_OFFSET_US + 50_000,
        T_OFFSET_US + T_END_REL_US - 200_000,
        n_flow,
        dtype=np.int64,
    )
    ends = starts + 100_000

    with open(ts_path, "w") as f:
        f.write("# timestamp_start_us, timestamp_end_us\n")
        for index in range(len(starts)):
            s = starts[index]
            e = ends[index]
            f.write(f"{s}, {e}\n")

    for i in range(n_flow):
        flow_x = rng.uniform(-5, 5, (HEIGHT, WIDTH)).astype(np.float32)
        flow_y = rng.uniform(-5, 5, (HEIGHT, WIDTH)).astype(np.float32)

        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint16)
        img[..., 0] = 1
        img[..., 1] = (flow_y * 128 + 32768).astype(np.uint16)
        img[..., 2] = (flow_x * 128 + 32768).astype(np.uint16)

        cv2.imwrite(str(flow_dir / f"{i:06d}.png"), img)


def _make_disparity_files(disp_dir: Path, ts_path: Path, n_disp: int = N_DISP) -> None:
    rng = np.random.RandomState(17)

    disp_dir.mkdir(parents=True, exist_ok=True)
    ts_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.linspace(
        T_OFFSET_US + 100_000,
        T_OFFSET_US + T_END_REL_US - 100_000,
        n_disp,
        dtype=np.int64,
    )

    with open(ts_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

    for i in range(n_disp):
        disp_values = rng.uniform(10, 100, (HEIGHT, WIDTH))
        disp_uint16 = (disp_values * 256).astype(np.uint16)
        disp_uint16[0:10, 0:10] = 0
        cv2.imwrite(str(disp_dir / f"{i:06d}.png"), disp_uint16)


def _make_calibration_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(
            """
intrinsics:
  cam0:
    camera_matrix: [320.0, 240.0, 320.0, 240.0]
    distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  camRect0:
    camera_matrix: [320.0, 240.0, 320.0, 240.0]
extrinsics:
  T_10:
    - [1, 0, 0, 0.1]
    - [0, 1, 0, 0]
    - [0, 0, 1, 0]
    - [0, 0, 0, 1]
  R_rect0:
    - [1, 0, 0]
    - [0, 1, 0]
    - [0, 0, 1]
""".lstrip()
        )


def _time_msg(timestamp_ns: int, time_cls: Any) -> Any:
    return time_cls(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)


def _make_lidar_imu_bag(path: Path) -> None:
    pytest.importorskip("rosbags.rosbag1", reason="DSEC LiDAR/IMU tests need rosbags")

    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores
    from rosbags.typesys import get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    time_cls = typestore.types["builtin_interfaces/msg/Time"]
    header_cls = typestore.types["std_msgs/msg/Header"]
    vector_cls = typestore.types["geometry_msgs/msg/Vector3"]
    quaternion_cls = typestore.types["geometry_msgs/msg/Quaternion"]
    imu_cls = typestore.types["sensor_msgs/msg/Imu"]
    point_field_cls = typestore.types["sensor_msgs/msg/PointField"]
    point_cloud_cls = typestore.types["sensor_msgs/msg/PointCloud2"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with Writer(path) as writer:
        imu_connection = writer.add_connection(
            "/imu/data",
            "sensor_msgs/msg/Imu",
            typestore=typestore,
        )
        lidar_connection = writer.add_connection(
            "/velodyne_points",
            "sensor_msgs/msg/PointCloud2",
            typestore=typestore,
        )

        covariance = np.zeros(9, dtype=np.float64)
        for i in range(N_IMU):
            timestamp_ns = (T_OFFSET_US + 100_000 + i * 100_000) * 1000
            imu_msg = imu_cls(
                header=header_cls(i, _time_msg(timestamp_ns, time_cls), "imu0"),
                orientation=quaternion_cls(0.0, 0.0, 0.0, 1.0),
                orientation_covariance=covariance,
                angular_velocity=vector_cls(float(i), float(i + 1), float(i + 2)),
                angular_velocity_covariance=covariance,
                linear_acceleration=vector_cls(float(i + 3), float(i + 4), float(i + 5)),
                linear_acceleration_covariance=covariance,
            )
            writer.write(
                imu_connection,
                timestamp_ns,
                bytes(typestore.serialize_ros1(imu_msg, "sensor_msgs/msg/Imu")),
            )

        fields = [
            point_field_cls("x", 0, 7, 1),
            point_field_cls("y", 4, 7, 1),
            point_field_cls("z", 8, 7, 1),
            point_field_cls("intensity", 12, 7, 1),
            point_field_cls("ring", 16, 4, 1),
            point_field_cls("time", 18, 7, 1),
        ]
        point_dtype = np.dtype(
            {
                "names": ["x", "y", "z", "intensity", "ring", "time"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f4"],
                "offsets": [0, 4, 8, 12, 16, 18],
                "itemsize": 22,
            }
        )
        for i in range(N_LIDAR):
            timestamp_ns = (T_OFFSET_US + 150_000 + i * 200_000) * 1000
            points = np.zeros(4, dtype=point_dtype)
            points["x"] = np.arange(4, dtype=np.float32) + i
            points["y"] = np.arange(4, dtype=np.float32) + 10 + i
            points["z"] = np.arange(4, dtype=np.float32) + 20 + i
            points["intensity"] = np.float32(30 + i)
            points["ring"] = np.arange(4, dtype=np.uint16)
            points["time"] = np.linspace(0, 0.001, 4, dtype=np.float32)
            cloud_msg = point_cloud_cls(
                header=header_cls(i, _time_msg(timestamp_ns, time_cls), "velodyne"),
                height=1,
                width=len(points),
                fields=fields,
                is_bigendian=False,
                point_step=22,
                row_step=22 * len(points),
                data=points.view(np.uint8).reshape(-1),
                is_dense=True,
            )
            writer.write(
                lidar_connection,
                timestamp_ns,
                bytes(typestore.serialize_ros1(cloud_msg, "sensor_msgs/msg/PointCloud2")),
            )


def _make_auxiliary_calibration(root: Path) -> None:
    drive_dir = root / "data" / DRIVE_PREFIX
    drive_dir.mkdir(parents=True, exist_ok=True)
    with open(drive_dir / "cam_to_lidar.yaml", "w") as f:
        f.write(
            """
T_lidar_camRect1:
  - [1, 0, 0, 0.1]
  - [0, 1, 0, 0.2]
  - [0, 0, 1, 0.3]
  - [0, 0, 0, 1]
""".lstrip()
        )

    imu_dir = root / "imu_calibration"
    imu_dir.mkdir(parents=True, exist_ok=True)
    with open(imu_dir / "cam0_to_imu0.yaml", "w") as f:
        f.write(
            """
T_cam0_imu0:
  - [1, 0, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 0]
  - [0, 0, 0, 1]
""".lstrip()
        )
    with open(imu_dir / "imu0_params.yaml", "w") as f:
        f.write(
            """
%YAML:1.0
---
type: IMU
name: mpu-9250
Gyr:
  unit: " rad/s"
""".lstrip()
        )


def _build_dsec_tree(root: Path) -> None:
    base = root / SEQ

    events_dir = base / "events" / "left"
    events_h5 = events_dir / "events.h5"
    rectify_h5 = events_dir / "rectify_map.h5"

    images_base = base / "images"
    image_ts = images_base / "timestamps.txt"
    image_dir = images_base / "left" / "rectified"
    image_exposure_ts = images_base / "left" / "exposure_timestamps.txt"

    flow_base = base / "flow"
    flow_fwd_dir = flow_base / "forward"
    flow_fwd_ts = flow_base / "forward_timestamps.txt"
    flow_bwd_dir = flow_base / "backward"
    flow_bwd_ts = flow_base / "backward_timestamps.txt"

    disp_base = base / "disparity"
    disp_dir = disp_base / "event"
    disp_ts = disp_base / "timestamps.txt"

    calib_path = base / "calibration" / "cam_to_cam.yaml"

    _make_events_h5(events_h5)
    _make_rectify_map_h5(rectify_h5)
    _make_image_files(image_dir, image_ts, image_exposure_ts)
    _make_flow_files(flow_fwd_dir, flow_fwd_ts)
    _make_flow_files(flow_bwd_dir, flow_bwd_ts, n_flow=N_FLOW)
    _make_disparity_files(disp_dir, disp_ts)
    _make_calibration_yaml(calib_path)


def _build_cleaned_dsec_tree(root: Path, split: str = "train") -> None:
    base = root / split / SEQ

    events_dir = base / "events" / "left"
    events_h5 = events_dir / "events.h5"
    rectify_h5 = events_dir / "rectify_map.h5"

    images_base = base / "images"
    image_ts = images_base / "timestamps.txt"
    image_dir = images_base / "left" / "rectified"
    image_exposure_ts = images_base / "left" / "exposure_timestamps.txt"

    flow_base = base / "flow"
    flow_fwd_dir = flow_base / "forward"
    flow_fwd_ts = flow_base / "forward_timestamps.txt"
    flow_bwd_dir = flow_base / "backward"
    flow_bwd_ts = flow_base / "backward_timestamps.txt"

    disp_base = base / "disparity"
    disp_dir = disp_base / "event"
    disp_ts = disp_base / "timestamps.txt"

    calib_path = base / "calibration" / "cam_to_cam.yaml"

    _make_events_h5(events_h5)
    _make_rectify_map_h5(rectify_h5)
    _make_image_files(image_dir, image_ts, image_exposure_ts)
    _make_flow_files(flow_fwd_dir, flow_fwd_ts)
    _make_flow_files(flow_bwd_dir, flow_bwd_ts, n_flow=N_FLOW)
    _make_disparity_files(disp_dir, disp_ts)
    _make_calibration_yaml(calib_path)


def _build_cleaned_dsec_tree_with_auxiliary(root: Path, split: str = "train") -> None:
    _build_cleaned_dsec_tree(root, split=split)
    _make_lidar_imu_bag(root / "data" / DRIVE_PREFIX / "lidar_imu.bag")
    _make_auxiliary_calibration(root)


@pytest.fixture()
def dsec_dir(tmp_path: Path) -> Path:  # noqa: D103
    _build_dsec_tree(tmp_path)
    return tmp_path


@pytest.fixture()
def dsec_dir_test(tmp_path: Path) -> Path:  # noqa: D103
    _build_dsec_tree(tmp_path)
    return tmp_path


class TestConstruction:  # noqa: D101
    def test_invalid_split_rejected(self, dsec_dir: Path) -> None:  # noqa: D102
        with pytest.raises(ValueError, match="split"):
            DSECDataLoader(str(dsec_dir), SEQ, split="invalid")  # type: ignore[arg-type]

    def test_invalid_camera_rejected(self, dsec_dir: Path) -> None:  # noqa: D102
        with pytest.raises(ValueError, match="camera"):
            DSECDataLoader(str(dsec_dir), SEQ, camera="middle")  # type: ignore[arg-type]

    def test_default_loads_events_and_rectify_map_only(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ) as loader:
            assert loader.num_events == N_EVENTS
            assert loader.has_rectify_map
            assert not loader.has_images
            assert not loader.has_flow_forward
            assert not loader.has_flow_backward
            assert not loader.has_disparity
            assert not loader.has_calibration
            assert loader.events_prerectified is False

    def test_test_split_round_trip(self, dsec_dir_test: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir_test), SEQ, split="test") as loader:
            assert loader.split == "test"
            assert loader.num_events == N_EVENTS

    def test_cleaned_sequence_layout_is_supported(self, tmp_path: Path) -> None:  # noqa: D102
        _build_cleaned_dsec_tree(tmp_path, split="train")

        with DSECDataLoader(
            str(tmp_path),
            SEQ,
            load_images=True,
            load_flow_forward=True,
            load_flow_backward=True,
            load_disparity=True,
            load_calibration=True,
        ) as loader:
            assert loader.num_events == N_EVENTS
            assert loader.has_rectify_map
            assert loader.num_images == N_IMAGES
            assert loader.num_flow_forward == N_FLOW
            assert loader.num_flow_backward == N_FLOW
            assert loader.num_disparity_frames == N_DISP
            assert loader.has_calibration
            assert loader._paths["events_h5"] == str(
                tmp_path / "train" / SEQ / "events" / "left" / "events.h5"
            )

    def test_split_directory_root_is_supported(self, tmp_path: Path) -> None:  # noqa: D102
        _build_cleaned_dsec_tree(tmp_path, split="train")

        with DSECDataLoader(str(tmp_path / "train"), SEQ, load_images=True) as loader:
            assert loader.num_events == N_EVENTS
            assert loader.num_images == N_IMAGES

    def test_sequence_directory_root_is_supported(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir / SEQ), SEQ, load_images=True) as loader:
            assert loader.num_events == N_EVENTS
            assert loader.num_images == N_IMAGES

    def test_official_rectify_maps_filename_is_supported(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        legacy_path = dsec_dir / SEQ / "events" / "left" / "rectify_map.h5"
        official_path = legacy_path.with_name("rectify_maps.h5")
        legacy_path.rename(official_path)

        with DSECDataLoader(str(dsec_dir), SEQ) as loader:
            assert loader.has_rectify_map


EVENT_MODES = ["lazy", "cached"]


class TestEvents:  # noqa: D101
    @pytest.mark.parametrize("mode", EVENT_MODES)
    def test_load_events_returns_typed_raw_events(  # noqa: D102
        self, dsec_dir: Path, mode: str
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode=mode) as loader:  # type: ignore[arg-type]
            events = loader.load_events(0, 100)
            assert isinstance(events, RawEvents)
            assert len(events.x) == 100
            assert events.x.dtype == np.int16
            assert events.y.dtype == np.int16
            assert events.timestamp.dtype == np.float64
            assert events.polarity.dtype == np.bool_

    def test_lazy_and_cached_return_equivalent_events(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="cached") as cached:
            ev_c = cached.load_events(100, 600)
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy") as lazy:
            ev_l = lazy.load_events(100, 600)

        np.testing.assert_array_equal(ev_c.x, ev_l.x)
        np.testing.assert_array_equal(ev_c.y, ev_l.y)
        np.testing.assert_array_equal(ev_c.polarity, ev_l.polarity)
        np.testing.assert_array_almost_equal(ev_c.timestamp, ev_l.timestamp)

    @pytest.mark.parametrize("mode", EVENT_MODES)
    def test_load_events_returns_mutable_copies(  # noqa: D102
        self, dsec_dir: Path, mode: str
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode=mode) as loader:  # type: ignore[arg-type]
            events = loader.load_events(0, 10)
            original_x = int(events.x[0])

            assert events.x.flags.writeable
            events.x[0] = -1

            reloaded = loader.load_events(0, 10)
            assert int(reloaded.x[0]) == original_x

    def test_timestamps_are_absolute_and_monotonic(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="cached") as loader:
            events = loader.load_events(0, loader.num_events)
            assert events.timestamp[0] >= T_OFFSET_US / 1e6
            assert np.all(np.diff(events.timestamp) >= 0)

    @pytest.mark.parametrize("mode", EVENT_MODES)
    def test_time_index_round_trip(self, dsec_dir: Path, mode: str) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode=mode) as loader:  # type: ignore[arg-type]
            t = loader.index_to_time(500)
            assert abs(loader.time_to_index(t) - 500) <= 1

    @pytest.mark.parametrize("mode", EVENT_MODES)
    def test_vectorized_lookups_match_scalar(self, dsec_dir: Path, mode: str) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode=mode) as loader:  # type: ignore[arg-type]
            indices = np.array([0, 100, 500, 1000], dtype=np.int64)
            scalar_times = np.array(
                [loader.index_to_time(int(i)) for i in indices], dtype=np.float64
            )
            np.testing.assert_array_equal(loader.indices_to_times(indices), scalar_times)
            np.testing.assert_array_equal(
                loader.times_to_indices(scalar_times),
                np.array([loader.time_to_index(float(t)) for t in scalar_times], dtype=np.int64),
            )

    @pytest.mark.parametrize("mode", EVENT_MODES)
    def test_indices_to_times_accepts_numpy_style_index_arrays(  # noqa: D102
        self,
        dsec_dir: Path,
        mode: str,
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode=mode) as loader:  # type: ignore[arg-type]
            indices = np.array([[500, 100], [500, -1]], dtype=np.int64)
            expected = np.array(
                [
                    [loader.index_to_time(500), loader.index_to_time(100)],
                    [loader.index_to_time(500), loader.index_to_time(loader.num_events - 1)],
                ],
                dtype=np.float64,
            )

            actual = loader.indices_to_times(indices)

            assert actual.shape == indices.shape
            np.testing.assert_array_equal(actual, expected)

    def test_get_events_by_time_returns_half_open_interval(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="cached") as loader:
            t_start = loader.index_to_time(100)
            t_end = loader.index_to_time(800)
            events = loader.get_events_by_time(t_start, t_end)
            assert len(events.x) > 0
            assert events.timestamp.min() >= t_start
            assert events.timestamp.max() < t_end

    def test_iter_events_covers_all_events(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy") as loader:
            chunks = list(loader.iter_events(num_events=500))
            assert sum(len(c.x) for c in chunks) == N_EVENTS

    def test_load_events_rejects_invalid_bounds(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy") as loader:
            with pytest.raises(IndexError, match="start index"):
                loader.load_events(-1, 10)
            with pytest.raises(ValueError, match="end index must be >= start index"):
                loader.load_events(10, 5)
            with pytest.raises(IndexError, match="end index"):
                loader.load_events(0, loader.num_events + 1)

    def test_index_to_time_normalizes_negative_indices(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy") as loader:
            assert loader.index_to_time(-1) == loader.index_to_time(loader.num_events - 1)
            with pytest.raises(IndexError, match="Event index"):
                loader.index_to_time(loader.num_events)


class TestRectification:  # noqa: D101
    def test_rectify_raises_without_map(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_rectify_map=False) as loader:
            assert loader.rectify_map is None
            events = loader.load_events(0, 10)
            with pytest.raises(RuntimeError, match="Rectification map"):
                loader.rectify_events(events)

    def test_identity_rectify_map_preserves_events(  # noqa: D102
        self, dsec_dir: Path, tmp_path: Path
    ) -> None:
        rectify_path = dsec_dir / SEQ / "events" / "left" / "rectify_map.h5"
        base_y, base_x = np.mgrid[:HEIGHT, :WIDTH]
        identity = np.stack([base_x, base_y], axis=-1).astype(np.float32)
        with h5py.File(rectify_path, "w") as f:
            f.create_dataset("rectify_map", data=identity)

        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="cached") as loader:
            events = loader.load_events(0, loader.num_events)
            rectified = loader.rectify_events(events)
            assert len(rectified.x) == len(events.x)
            np.testing.assert_array_equal(rectified.x, events.x)
            np.testing.assert_array_equal(rectified.y, events.y)

    def test_rectify_drops_invalid_raw_coordinates(self, dsec_dir: Path) -> None:  # noqa: D102
        rectify_path = dsec_dir / SEQ / "events" / "left" / "rectify_map.h5"
        base_y, base_x = np.mgrid[:HEIGHT, :WIDTH]
        identity = np.stack([base_x, base_y], axis=-1).astype(np.float32)
        with h5py.File(rectify_path, "w") as f:
            f.create_dataset("rectify_map", data=identity)

        events = RawEvents(
            x=np.array([-1, 4, WIDTH], dtype=np.int16),
            y=np.array([3, 5, 6], dtype=np.int16),
            timestamp=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            polarity=np.array([True, False, True], dtype=np.bool_),
        )

        with DSECDataLoader(str(dsec_dir), SEQ) as loader:
            rectified = loader.rectify_events(events)

        np.testing.assert_array_equal(rectified.x, np.array([4], dtype=np.int16))
        np.testing.assert_array_equal(rectified.y, np.array([5], dtype=np.int16))
        np.testing.assert_array_equal(rectified.timestamp, np.array([2.0]))


class TestPrerectify:  # noqa: D101
    def test_load_events_matches_explicit_rectify(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="cached") as raw:
            expected = raw.rectify_events(raw.load_events(0, raw.num_events))

        with DSECDataLoader(
            str(dsec_dir),
            SEQ,
            event_load_mode="cached",
            prerectify_events=True,
        ) as pre:
            assert pre.events_prerectified
            assert pre.num_events == expected.x.shape[0]
            got = pre.load_events(0, pre.num_events)
            np.testing.assert_array_equal(got.x, expected.x)
            np.testing.assert_array_equal(got.y, expected.y)
            np.testing.assert_array_equal(got.polarity, expected.polarity)
            np.testing.assert_allclose(got.timestamp, expected.timestamp)

    def test_rectify_events_is_noop_when_prerectified(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(
            str(dsec_dir),
            SEQ,
            event_load_mode="cached",
            prerectify_events=True,
        ) as loader:
            events = loader.load_events(0, loader.num_events)
            rectified = loader.rectify_events(events)
            assert rectified.x is events.x
            assert rectified.y is events.y

    def test_requires_cached_mode(self, dsec_dir: Path) -> None:  # noqa: D102
        with pytest.raises(ValueError, match="event_load_mode='cached'"):
            DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy", prerectify_events=True)

    def test_requires_rectify_map(self, dsec_dir: Path) -> None:  # noqa: D102
        with pytest.raises(ValueError, match="load_rectify_map=True"):
            DSECDataLoader(
                str(dsec_dir),
                SEQ,
                event_load_mode="cached",
                load_rectify_map=False,
                prerectify_events=True,
            )


class TestImages:  # noqa: D101
    def test_image_count_and_timestamps(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_images=True) as loader:
            assert loader.has_images
            assert loader.num_images == N_IMAGES
            ts = loader.image_timestamps
            assert ts is not None and ts.dtype == np.float64
            assert len(ts) == N_IMAGES
            assert np.all(np.diff(ts) > 0)
            exposure_ts = loader.image_exposure_timestamps
            assert exposure_ts is not None
            assert exposure_ts.shape == (N_IMAGES, 2)
            assert np.all(exposure_ts[:, 1] > exposure_ts[:, 0])

    @pytest.mark.parametrize("mode", [True, "cached"])
    def test_load_image(self, dsec_dir: Path, mode: object) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_images=mode) as loader:  # type: ignore[arg-type]
            img = loader.load_image(0)
            assert img.dtype == np.uint8
            assert img.ndim >= 2

    def test_image_count_mismatch_raises(self, dsec_dir: Path) -> None:  # noqa: D102
        image_dir = dsec_dir / SEQ / "images" / "left" / "rectified"
        sorted(image_dir.glob("*.png"))[-1].unlink()

        with pytest.raises(ValueError, match="image count mismatch"):
            DSECDataLoader(str(dsec_dir), SEQ, load_images=True)


class TestFlowForward:  # noqa: D101
    def test_count_and_timestamps(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_flow_forward=True) as loader:
            assert loader.has_flow_forward
            assert loader.num_flow_forward == N_FLOW
            ts = loader.flow_forward_timestamps
            assert ts is not None
            assert ts.shape == (N_FLOW, 2)
            assert np.all(ts[:, 1] > ts[:, 0])

    @pytest.mark.parametrize("mode", [True, "cached"])
    def test_load_flow_forward(self, dsec_dir: Path, mode: object) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_flow_forward=mode) as loader:  # type: ignore[arg-type]
            flow, valid = loader.load_flow_forward(0)
            assert flow.shape == (HEIGHT, WIDTH, 2)
            assert flow.dtype == np.float32
            assert valid.shape == (HEIGHT, WIDTH)
            assert valid.dtype == np.bool_

    def test_flow_count_mismatch_raises(self, dsec_dir: Path) -> None:  # noqa: D102
        flow_dir = dsec_dir / SEQ / "flow" / "forward"
        sorted(flow_dir.glob("*.png"))[-1].unlink()

        with pytest.raises(ValueError, match="forward flow count mismatch"):
            DSECDataLoader(str(dsec_dir), SEQ, load_flow_forward=True)

    def test_png_decode_round_trip_recovers_known_flow(self, tmp_path: Path) -> None:  # noqa: D102
        # DSEC stores flow PNG channels as x, y, valid. OpenCV reads BGR.
        h, w = 8, 12
        rng = np.random.default_rng(0)
        flow_x = rng.uniform(-200, 200, (h, w)).astype(np.float32)
        flow_y = rng.uniform(-200, 200, (h, w)).astype(np.float32)
        valid = rng.integers(0, 2, (h, w), dtype=np.uint16)

        encoded = np.zeros((h, w, 3), dtype=np.uint16)
        encoded[..., 0] = valid
        encoded[..., 1] = (flow_y * 128 + 32768).astype(np.uint16)
        encoded[..., 2] = (flow_x * 128 + 32768).astype(np.uint16)

        png_path = tmp_path / "flow.png"
        assert cv2.imwrite(str(png_path), encoded)
        decoded_flow, decoded_valid = _read_flow_png(str(png_path))

        np.testing.assert_allclose(decoded_flow[..., 0], flow_x, atol=1.0 / 128)
        np.testing.assert_allclose(decoded_flow[..., 1], flow_y, atol=1.0 / 128)
        np.testing.assert_array_equal(decoded_valid, valid.astype(bool))

    def test_flow_png_rejects_wrong_shape(self, tmp_path: Path) -> None:  # noqa: D102
        png_path = tmp_path / "bad_flow.png"
        assert cv2.imwrite(str(png_path), np.zeros((8, 12), dtype=np.uint16))

        with pytest.raises(ValueError, match="flow PNG"):
            _read_flow_png(str(png_path))


class TestFlowBackward:  # noqa: D101
    def test_frame_sample_associates_backward_by_anchor(self, dsec_dir: Path) -> None:  # noqa: D102
        # Backward flow rows are matched by anchor time, not row number.
        flow_root = dsec_dir / SEQ / "flow"
        fwd_ts = _load_flow_timestamps(str(flow_root / "forward_timestamps.txt"))
        new_bwd = np.empty((len(fwd_ts) - 1, 2), dtype=np.int64)
        new_bwd[:, 0] = fwd_ts[1:, 0]
        new_bwd[:, 1] = fwd_ts[:-1, 0]
        with open(flow_root / "backward_timestamps.txt", "w") as f:
            f.write("# from_timestamp_us, to_timestamp_us\n")
            for s, e in new_bwd:
                f.write(f"{s}, {e}\n")
        sorted((flow_root / "backward").glob("*.png"))[-1].unlink()

        with DSECDataLoader(
            str(dsec_dir),
            SEQ,
            load_flow_forward=True,
            load_flow_backward=True,
        ) as loader:
            assert loader.load_frame_sample(0)["flow_backward"] is None
            sample1 = loader.load_frame_sample(1)
            assert sample1["flow_backward"] is not None
            expected_flow, expected_valid = loader.load_flow_backward(0)
            np.testing.assert_array_equal(sample1["flow_backward"][0], expected_flow)
            np.testing.assert_array_equal(sample1["flow_backward"][1], expected_valid)


class TestDisparity:  # noqa: D101
    def test_count_and_timestamps(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_disparity=True) as loader:
            assert loader.has_disparity
            assert loader.num_disparity_frames == N_DISP
            ts = loader.disparity_timestamps
            assert ts is not None and len(ts) == N_DISP

    @pytest.mark.parametrize("mode", [True, "cached"])
    def test_load_disparity(self, dsec_dir: Path, mode: object) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_disparity=mode) as loader:  # type: ignore[arg-type]
            disp, valid = loader.load_disparity(0)
            assert disp.shape == (HEIGHT, WIDTH)
            assert disp.dtype == np.float32
            assert valid.dtype == np.bool_
            assert not valid[:10, :10].any()
            assert valid[10:, 10:].all()

    def test_disparity_count_mismatch_raises(self, dsec_dir: Path) -> None:  # noqa: D102
        disp_dir = dsec_dir / SEQ / "disparity" / "event"
        sorted(disp_dir.glob("*.png"))[-1].unlink()

        with pytest.raises(ValueError, match="disparity count mismatch"):
            DSECDataLoader(str(dsec_dir), SEQ, load_disparity=True)

    def test_png_decode_round_trip_recovers_known_disparity(  # noqa: D102
        self, tmp_path: Path
    ) -> None:
        # DSEC disparity is stored as uint16 with disparity equal to I / 256.
        h, w = 8, 12
        rng = np.random.default_rng(0)
        disp_values = rng.uniform(1, 200, (h, w)).astype(np.float32)
        encoded = (disp_values * 256).astype(np.uint16)
        encoded[0, 0] = 0

        png_path = tmp_path / "disp.png"
        assert cv2.imwrite(str(png_path), encoded)
        decoded_disp, decoded_valid = _read_disparity_png(str(png_path))

        np.testing.assert_allclose(decoded_disp[0, 1:], disp_values[0, 1:], atol=1.0 / 256)
        assert not decoded_valid[0, 0]
        assert decoded_valid[1:, :].all()

    def test_disparity_png_rejects_multichannel_data(self, tmp_path: Path) -> None:  # noqa: D102
        png_path = tmp_path / "bad_disp.png"
        assert cv2.imwrite(str(png_path), np.zeros((8, 12, 3), dtype=np.uint16))

        with pytest.raises(ValueError, match="single channel"):
            _read_disparity_png(str(png_path))


class TestCalibration:  # noqa: D101
    def test_calibration_loaded_with_intrinsics_and_extrinsics(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, load_calibration=True) as loader:
            assert loader.has_calibration
            calib = loader.calibration
            assert calib is not None
            assert "intrinsics" in calib
            assert "extrinsics" in calib

    def test_calibration_property_returns_mutation_safe_copy(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, load_calibration=True) as loader:
            first = loader.calibration
            assert first is not None
            first["intrinsics"]["mutated"] = True

            second = loader.calibration
            assert second is not None
            assert "mutated" not in second["intrinsics"]

    def test_auxiliary_calibration_loaded_from_cleaned_layout(  # noqa: D102
        self, tmp_path: Path
    ) -> None:
        _build_cleaned_dsec_tree_with_auxiliary(tmp_path)

        with DSECDataLoader(str(tmp_path), SEQ, load_calibration=True) as loader:
            cam_to_lidar = loader.cam_to_lidar_calibration
            cam_to_imu = loader.cam_to_imu_calibration
            imu_calibration = loader.imu_calibration

        assert cam_to_lidar is not None and "T_lidar_camRect1" in cam_to_lidar
        assert cam_to_imu is not None and "T_cam0_imu0" in cam_to_imu
        assert imu_calibration is not None and imu_calibration["type"] == "IMU"


class TestLidarImu:  # noqa: D101
    def test_lidar_imu_request_explains_ros_extra_when_rosbags_missing(  # noqa: D102
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _build_cleaned_dsec_tree(tmp_path)
        bag_path = tmp_path / "data" / DRIVE_PREFIX / "lidar_imu.bag"
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        bag_path.touch()

        real_import = builtins.__import__

        def blocked_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "rosbags" or name.startswith("rosbags."):
                raise ImportError("blocked rosbags")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match=r"event-vision-library\[ros\]"):
            DSECDataLoader(str(tmp_path), SEQ, load_imu=True)

    def test_lazy_lidar_and_imu_load_from_cleaned_layout(  # noqa: D102
        self, tmp_path: Path
    ) -> None:
        _build_cleaned_dsec_tree_with_auxiliary(tmp_path)

        with DSECDataLoader(str(tmp_path), SEQ, load_imu=True, load_lidar=True) as loader:
            assert loader.has_imu
            assert loader.num_imu_samples == N_IMU
            assert loader.has_lidar
            assert loader.num_lidar_scans == N_LIDAR
            assert loader.lidar_frame_id == "velodyne"

            imu_start = loader.imu_timestamps[0]  # type: ignore[index]
            imu = loader.load_imu(float(imu_start), float(imu_start + 0.25))
            assert imu["timestamp"].shape == (3,)
            np.testing.assert_array_equal(
                imu["angular_velocity"][0],
                np.array([0.0, 1.0, 2.0]),
            )
            np.testing.assert_array_equal(
                imu["linear_acceleration"][0],
                np.array([3.0, 4.0, 5.0]),
            )

            scan = loader.load_lidar(0)
            assert scan["timestamp"] == pytest.approx(T_OFFSET_US / 1e6 + 0.15)
            assert scan["frame_id"] == "velodyne"
            assert scan["points"].dtype.names == (
                "x",
                "y",
                "z",
                "intensity",
                "ring",
                "time",
            )
            assert scan["points"].shape == (4,)
            np.testing.assert_array_equal(scan["points"]["x"], np.arange(4, dtype=np.float32))
            np.testing.assert_array_equal(scan["points"]["ring"], np.arange(4, dtype=np.uint16))

    def test_cached_lidar_and_imu_match_lazy(self, tmp_path: Path) -> None:  # noqa: D102
        _build_cleaned_dsec_tree_with_auxiliary(tmp_path)

        with DSECDataLoader(str(tmp_path), SEQ, load_imu=True, load_lidar=True) as lazy:
            t_start = lazy.imu_timestamps[0]  # type: ignore[index]
            t_end = lazy.imu_timestamps[-1] + 1e-6  # type: ignore[index]
            lazy_imu = lazy.load_imu(float(t_start), float(t_end))
            lazy_lidar = lazy.load_lidar(1)

        with DSECDataLoader(str(tmp_path), SEQ, load_imu="cached", load_lidar="cached") as cached:
            cached_imu = cached.load_imu(float(t_start), float(t_end))
            cached_lidar = cached.load_lidar(1)
            assert cached.imu_data is not None

        np.testing.assert_array_equal(cached_imu["timestamp"], lazy_imu["timestamp"])
        np.testing.assert_array_equal(
            cached_imu["angular_velocity"],
            lazy_imu["angular_velocity"],
        )
        assert cached_lidar["timestamp"] == lazy_lidar["timestamp"]
        np.testing.assert_array_equal(cached_lidar["points"], lazy_lidar["points"])

    def test_frame_sample_includes_lidar_and_imu_when_loaded(  # noqa: D102
        self, tmp_path: Path
    ) -> None:
        _build_cleaned_dsec_tree_with_auxiliary(tmp_path)

        with DSECDataLoader(
            str(tmp_path),
            SEQ,
            load_images=True,
            load_imu=True,
            load_lidar=True,
        ) as loader:
            sample = loader.load_frame_sample(0)

        assert sample["imu"] is not None
        assert sample["imu"]["timestamp"].shape == (5,)
        assert sample["lidar"] is not None
        assert len(sample["lidar"]) == 3


class TestFrameSampling:  # noqa: D101
    def test_num_samples_prefers_flow_then_images(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataLoader(str(dsec_dir), SEQ, load_flow_forward=True) as loader:
            assert loader.num_samples == N_FLOW
        with DSECDataLoader(str(dsec_dir), SEQ, load_images=True) as loader:
            assert loader.num_samples == N_IMAGES - 1
        with DSECDataLoader(str(dsec_dir), SEQ) as loader:
            assert loader.num_samples == 0

    def test_load_frame_sample_populates_all_loaded_modalities(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        with DSECDataLoader(
            str(dsec_dir),
            SEQ,
            load_flow_forward=True,
            load_flow_backward=True,
            load_images=True,
            load_disparity=True,
            event_load_mode="cached",
        ) as loader:
            sample = loader.load_frame_sample(0)
            t_start, t_end = sample["timestamp"]
            assert t_end > t_start
            assert isinstance(sample["events"], RawEvents)
            assert sample["image_start"] is not None
            assert sample["image_end"] is not None
            assert sample["flow"] is not None
            assert sample["flow_backward"] is not None
            assert sample["disparity"] is not None

    def test_load_frame_sample_rejects_out_of_range_index(  # noqa: D102
        self, dsec_dir: Path
    ) -> None:
        with DSECDataLoader(str(dsec_dir), SEQ, load_flow_forward=True) as loader:
            with pytest.raises(IndexError, match="Sample index"):
                loader.load_frame_sample(loader.num_samples)


class TestClose:  # noqa: D101
    def test_close_releases_lazy_handle(self, dsec_dir: Path) -> None:  # noqa: D102
        loader = DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy")
        loader.load_events(0, 10)
        assert loader._events_h5 is not None
        loader.close()
        assert loader._events_h5 is None

    def test_lazy_events_registers_hdf5_plugins_after_pickle(  # noqa: D102
        self,
        dsec_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        register_calls: List[None] = []

        def register_plugins() -> None:
            register_calls.append(None)

        monkeypatch.setattr(dsec_module, "_register_hdf5_filter_plugins", register_plugins)

        loader = DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy")
        loader.load_events(0, 10)
        assert loader._events_h5 is not None
        register_calls.clear()

        restored_loader = pickle.loads(pickle.dumps(loader))  # noqa: S301
        assert restored_loader._events_h5 is None

        try:
            restored_events = restored_loader.load_events(0, 10)
            assert restored_events.x.shape == (10,)
            assert len(register_calls) == 1
        finally:
            loader.close()
            restored_loader.close()

    def test_close_is_idempotent(self, dsec_dir: Path) -> None:  # noqa: D102
        loader = DSECDataLoader(str(dsec_dir), SEQ, event_load_mode="lazy")
        loader.close()
        loader.close()
