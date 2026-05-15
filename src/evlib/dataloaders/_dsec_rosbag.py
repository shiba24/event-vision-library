"""ROS bag helpers for DSEC LiDAR and IMU loading."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from typing import Iterable
from typing import TypedDict
from typing import cast

import numpy as np
import numpy.typing as npt

from .utils import freeze_array


logger = logging.getLogger(__name__)


_DSEC_IMU_TOPIC = "/imu/data"
_DSEC_LIDAR_TOPIC = "/velodyne_points"
_ROS_NS_TO_SECONDS = 1e-9

_LidarPoints = npt.NDArray[np.void]


class DSECImuData(TypedDict):
    """IMU measurements returned by DSEC IMU loading methods."""

    timestamp: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]
    linear_acceleration: npt.NDArray[np.float64]
    orientation: npt.NDArray[np.float64]


class DSECLidarScan(TypedDict):
    """Decoded DSEC Velodyne scan returned by ``load_lidar``."""

    timestamp: float
    points: _LidarPoints
    frame_id: str


def _empty_imu_data() -> DSECImuData:
    return DSECImuData(
        timestamp=np.empty(0, dtype=np.float64),
        angular_velocity=np.empty((0, 3), dtype=np.float64),
        linear_acceleration=np.empty((0, 3), dtype=np.float64),
        orientation=np.empty((0, 4), dtype=np.float64),
    )


def _copy_imu_data(imu_data: DSECImuData, start: int, end: int) -> DSECImuData:
    return DSECImuData(
        timestamp=imu_data["timestamp"][start:end].copy(),
        angular_velocity=imu_data["angular_velocity"][start:end].copy(),
        linear_acceleration=imu_data["linear_acceleration"][start:end].copy(),
        orientation=imu_data["orientation"][start:end].copy(),
    )


def _freeze_imu_data(imu_data: DSECImuData) -> DSECImuData:
    freeze_array(imu_data["timestamp"])
    freeze_array(imu_data["angular_velocity"])
    freeze_array(imu_data["linear_acceleration"])
    freeze_array(imu_data["orientation"])
    return imu_data


def _rosbags_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "rosbags is required for loading DSEC LiDAR/IMU bags. "
        "Install event-vision-library with the 'ros' extra, for example "
        "`pip install event-vision-library[ros]`."
    )


def _require_rosbag_reader() -> Any:
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise _rosbags_import_error(exc) from exc
    return Reader


@lru_cache(maxsize=1)
def _get_ros1_typestore() -> Any:
    try:
        from rosbags.typesys import Stores
        from rosbags.typesys import get_typestore
    except ImportError as exc:
        raise _rosbags_import_error(exc) from exc
    return get_typestore(Stores.ROS1_NOETIC)


def _connection_for_topic(reader: Any, topic: str, bag_path: str) -> Any | None:
    for connection in reader.connections:
        if connection.topic == topic:
            return connection

    logger.warning("DSEC ROS bag topic %s not found: %s", topic, bag_path)
    return None


def _connection_timestamps_ns(reader: Any, connection: Any) -> npt.NDArray[np.int64]:
    timestamps_ns = np.fromiter(
        (entry.time for entry in reader.indexes[connection.id]),
        dtype=np.int64,
        count=int(connection.msgcount),
    )
    freeze_array(timestamps_ns)
    return timestamps_ns


def _ros_ns_to_seconds(timestamps_ns: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    timestamps = timestamps_ns.astype(np.float64)
    timestamps *= _ROS_NS_TO_SECONDS
    freeze_array(timestamps)
    return timestamps


def _pointcloud2_dtype(fields: list[Any], point_step: int, is_bigendian: bool) -> np.dtype:
    endian = ">" if is_bigendian else "<"
    datatype_map = {
        1: "i1",
        2: "u1",
        3: f"{endian}i2",
        4: f"{endian}u2",
        5: f"{endian}i4",
        6: f"{endian}u4",
        7: f"{endian}f4",
        8: f"{endian}f8",
    }
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    for field in fields:
        datatype = int(field.datatype)
        if datatype not in datatype_map:
            raise ValueError(f"Unsupported PointCloud2 field datatype {datatype}.")
        names.append(str(field.name))
        dtype = np.dtype(datatype_map[datatype])
        count = int(field.count)
        formats.append(dtype if count == 1 else (dtype, (count,)))
        offsets.append(int(field.offset))
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(point_step),
        }
    )


def _decode_pointcloud2_points(msg: Any) -> _LidarPoints:
    height = int(msg.height)
    width = int(msg.width)
    point_step = int(msg.point_step)
    row_step = int(msg.row_step)
    if height < 0 or width < 0:
        raise ValueError(f"PointCloud2 dimensions cannot be negative, got {height}x{width}.")
    if height == 0 or width == 0:
        return np.empty(0, dtype=_pointcloud2_dtype(msg.fields, point_step, bool(msg.is_bigendian)))
    if point_step <= 0 or row_step < point_step * width:
        raise ValueError(
            f"Invalid PointCloud2 layout: point_step={point_step}, row_step={row_step}, "
            f"width={width}."
        )

    data = np.asarray(msg.data, dtype=np.uint8)
    required_bytes = row_step * height
    if data.size < required_bytes:
        raise ValueError(
            f"PointCloud2 data has {data.size} bytes but layout requires {required_bytes}."
        )

    dtype = _pointcloud2_dtype(msg.fields, point_step, bool(msg.is_bigendian))
    if row_step == point_step * width:
        # Compact PointCloud2 can be copied in one pass
        points = np.frombuffer(data[:required_bytes], dtype=dtype, count=height * width).copy()
        return cast(_LidarPoints, points)

    # Padded PointCloud2 rows need compaction
    points_2d: npt.NDArray[np.void] = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=data,
        strides=(row_step, point_step),
    )
    points = np.empty(height * width, dtype=dtype)
    for row in range(height):
        start = row * width
        points[start : start + width] = points_2d[row]
    return cast(_LidarPoints, points)


def _decode_lidar_scan(timestamp_ns: int, raw: bytes, connection: Any) -> DSECLidarScan:
    typestore = _get_ros1_typestore()
    msg = typestore.deserialize_ros1(raw, connection.msgtype)
    scan = DSECLidarScan(
        timestamp=float(timestamp_ns) * _ROS_NS_TO_SECONDS,
        points=_decode_pointcloud2_points(msg),
        frame_id=str(msg.header.frame_id),
    )
    freeze_array(scan["points"])
    return scan


def _decode_imu_messages(
    messages: Iterable[tuple[int, bytes, Any]],
    n_messages: int,
) -> DSECImuData:
    if n_messages < 0:
        raise ValueError(f"n_messages must be >= 0, got {n_messages}.")
    if n_messages == 0:
        return _empty_imu_data()

    typestore = _get_ros1_typestore()
    timestamps = np.empty(n_messages, dtype=np.float64)
    angular_velocity = np.empty((n_messages, 3), dtype=np.float64)
    linear_acceleration = np.empty((n_messages, 3), dtype=np.float64)
    orientation = np.empty((n_messages, 4), dtype=np.float64)

    count = 0
    for timestamp_ns, raw, connection in messages:
        if count >= n_messages:
            raise ValueError(
                f"DSEC IMU timestamp index expected {n_messages} messages, but the ROS bag yielded more."
            )
        msg = typestore.deserialize_ros1(raw, connection.msgtype)
        timestamps[count] = float(timestamp_ns) * _ROS_NS_TO_SECONDS
        angular_velocity[count] = (
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        )
        linear_acceleration[count] = (
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        )
        orientation[count] = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        count += 1

    if count != n_messages:
        raise ValueError(
            f"DSEC IMU timestamp index expected {n_messages} messages, but the ROS bag yielded {count}."
        )

    return _freeze_imu_data(
        DSECImuData(
            timestamp=timestamps,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            orientation=orientation,
        )
    )


def _read_imu_messages_from_reader(
    reader: Any,
    imu_connection: Any,
    start_ns: int | None,
    stop_ns: int | None,
    n_messages: int,
) -> DSECImuData:
    messages = (
        (timestamp_ns, raw, connection)
        for connection, timestamp_ns, raw in reader.messages(
            connections=[imu_connection],
            start=start_ns,
            stop=stop_ns,
        )
    )
    return _decode_imu_messages(messages, n_messages)
