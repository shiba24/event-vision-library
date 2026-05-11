"""Low level I/O loader for the DSEC dataset.

Reference: https://dsec.ifi.uzh.ch/
Gehrig, M., Aarents, W., Gehrig, D., & Scaramuzza, D. (2021).
DSEC: A Stereo Event Camera Dataset for Driving Scenarios.
IEEE Robotics and Automation Letters, 6(3), 4947-4954.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from evlib.codec.fileformat.hdf5 import open_hdf5
from evlib.types import RawEvents

from ._base import DataLoaderBase
from ._storage_common import LoadingType
from ._storage_common import LoadMode
from ._storage_common import ResidentLoadMode
from .utils import LazyDecodeCache
from .utils import cv2
from .utils import decode_in_parallel
from .utils import find_nearest_index
from .utils import freeze_array
from .utils import normalize_index
from .utils import normalize_indices
from .utils import validate_index_interval


logger = logging.getLogger(__name__)


DSECSplit = Literal["train", "test"]
DSECCamera = Literal["left", "right"]
FlowDirection = Literal["forward", "backward"]


_DEFAULT_LAZY_IMAGE_CACHE_ITEMS = 4
_DEFAULT_LAZY_FLOW_CACHE_ITEMS = 2
_DEFAULT_LAZY_DISPARITY_CACHE_ITEMS = 2
_DEFAULT_LAZY_LIDAR_CACHE_ITEMS = 2

_PNG_DECODE_PARALLELISM = 8

_BLOSC_HDF5_FILTER_ID = 32001
_DSEC_IMU_TOPIC = "/imu/data"
_DSEC_LIDAR_TOPIC = "/velodyne_points"
_ROS_NS_TO_SECONDS = 1e-9

_DecodedDenseField = Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]
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


class DSECSample(TypedDict):
    """Synchronized DSEC sample dict returned by ``load_frame_sample``."""

    events: RawEvents
    timestamp: tuple[float, float]
    image_start: npt.NDArray[np.uint8] | None
    image_end: npt.NDArray[np.uint8] | None
    flow: _DecodedDenseField | None
    flow_backward: _DecodedDenseField | None
    disparity: _DecodedDenseField | None
    imu: DSECImuData | None
    lidar: list[DSECLidarScan] | None


def _freeze_dense_field(
    dense_field: _DecodedDenseField,
) -> _DecodedDenseField:
    field_values, valid_mask = dense_field
    freeze_array(field_values)
    freeze_array(valid_mask)
    return dense_field


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


def _sorted_png_paths(directory: str) -> list[str]:
    """Return sorted string paths for PNG files in a directory."""
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return []

    png_paths: list[Path] = []
    for path in directory_path.iterdir():
        suffix = path.suffix.lower()
        if suffix != ".png":
            continue
        png_paths.append(path)

    sorted_png_paths = sorted(png_paths)
    sorted_png_path_strings: list[str] = []
    for path in sorted_png_paths:
        sorted_png_path_strings.append(str(path))
    return sorted_png_path_strings


def _validate_png_count(
    modality: str,
    paths: list[str],
    timestamps: npt.NDArray[Any],
    timestamp_path: str,
) -> None:
    """Check that image like files and timestamp rows line up."""
    if len(paths) != len(timestamps):
        raise ValueError(
            f"DSEC {modality} count mismatch: found {len(paths)} PNG files, "
            f"but {len(timestamps)} timestamp rows in {timestamp_path}."
        )


def _load_integer_rows(path: str, min_columns: int, description: str) -> list[list[int]]:
    """Read integer timestamp metadata rows."""
    rows: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields: list[str] = []
            for field in re.split(r"[\s,]+", stripped):
                if field:
                    fields.append(field)

            try:
                values: list[int] = []
                for field in fields:
                    values.append(int(field))
            except ValueError as exc:
                # Official metadata often starts with a text header
                if not rows and any(ch.isalpha() for ch in stripped):
                    continue
                raise ValueError(
                    f"Failed to parse {description} timestamp row at "
                    f"{path}:{line_number}: {stripped!r}"
                ) from exc

            if len(values) < min_columns:
                raise ValueError(
                    f"Expected at least {min_columns} integer column(s) in "
                    f"{description} timestamp row at {path}:{line_number}, "
                    f"got {len(values)}."
                )
            rows.append(values)
    return rows


def _load_timestamps_us(path: str) -> npt.NDArray[np.int64]:
    rows = _load_integer_rows(path, min_columns=1, description="one-column")
    if not rows:
        return np.empty(0, dtype=np.int64)
    if any(len(row) != 1 for row in rows):
        raise ValueError(f"Expected one timestamp column in {path}.")

    timestamps_us: list[int] = []
    for row in rows:
        timestamps_us.append(row[0])
    return np.asarray(timestamps_us, dtype=np.int64)


def _load_timestamp_pairs_us(path: str, description: str) -> npt.NDArray[np.int64]:
    rows = _load_integer_rows(path, min_columns=2, description=description)
    if not rows:
        return np.empty((0, 2), dtype=np.int64)

    timestamp_pairs_us: list[list[int]] = []
    for row in rows:
        timestamp_pairs_us.append(row[:2])
    return np.asarray(timestamp_pairs_us, dtype=np.int64)


def _load_flow_timestamps(path: str) -> npt.NDArray[np.int64]:
    """Load flow ``(N, 2)`` ``[start_us, end_us]`` pairs.

    File has a header followed by comma separated pairs,
    a third file index column may be present in test metadata.
    """
    return _load_timestamp_pairs_us(path, "flow")


def _validate_strictly_increasing(
    name: str,
    timestamps: npt.NDArray[np.int64],
    timestamp_path: str,
) -> None:
    if timestamps.size > 1 and bool(np.any(np.diff(timestamps) <= 0)):
        raise ValueError(f"DSEC {name} timestamps must be strictly increasing: {timestamp_path}")


def _validate_timestamp_pairs(
    name: str,
    timestamp_pairs: npt.NDArray[np.int64],
    timestamp_path: str,
    *,
    require_positive_duration: bool,
) -> None:
    if timestamp_pairs.ndim != 2 or timestamp_pairs.shape[1] != 2:
        raise ValueError(
            f"DSEC {name} timestamps must have shape (N, 2), "
            f"got {timestamp_pairs.shape} in {timestamp_path}."
        )
    if timestamp_pairs.size == 0:
        return

    if timestamp_pairs.shape[0] > 1 and bool(np.any(np.diff(timestamp_pairs[:, 0]) <= 0)):
        raise ValueError(f"DSEC {name} anchor timestamps must be strictly increasing.")

    durations = timestamp_pairs[:, 1] - timestamp_pairs[:, 0]
    if require_positive_duration and bool(np.any(durations <= 0)):
        raise ValueError(f"DSEC {name} timestamp intervals must have positive duration.")
    if not require_positive_duration and bool(np.any(durations == 0)):
        raise ValueError(f"DSEC {name} timestamp intervals must have nonzero duration.")


def _require_h5_dataset(h5_file: h5py.File, key: str, file_path: str) -> h5py.Dataset:
    if key not in h5_file:
        raise KeyError(f"DSEC HDF5 file {file_path} is missing required dataset {key!r}.")
    dataset = h5_file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"DSEC HDF5 entry {key!r} in {file_path} is not a dataset.")
    return dataset


def _require_1d_h5_dataset(h5_file: h5py.File, key: str, file_path: str) -> h5py.Dataset:
    dataset = _require_h5_dataset(h5_file, key, file_path)
    if len(dataset.shape) != 1:
        raise ValueError(
            f"DSEC HDF5 dataset {key!r} in {file_path} must be 1D, " f"got shape {dataset.shape}."
        )
    return dataset


def _register_hdf5_filter_plugins() -> None:
    """Register bundled HDF5 compression plugins when ``hdf5plugin`` is installed."""
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        return


def _dataset_uses_filter(dataset: h5py.Dataset, filter_id: int) -> bool:
    plist = dataset.id.get_create_plist()
    for filter_index in range(plist.get_nfilters()):
        registered_filter_id = int(plist.get_filter(filter_index)[0])
        if registered_filter_id == filter_id:
            return True
    return False


def _read_h5_array(
    dataset: h5py.Dataset,
    key: str,
    file_path: str,
    index: Any = None,
    dtype: Any = None,
) -> npt.NDArray[Any]:
    if index is None:
        index = slice(None)
    try:
        return np.asarray(dataset[index], dtype=dtype)
    except OSError as exc:
        # show only after hdf5 read failure
        if _dataset_uses_filter(dataset, _BLOSC_HDF5_FILTER_ID):
            raise ImportError(
                f"DSEC HDF5 dataset {key!r} in {file_path} uses Blosc compression. "
                "Install hdf5plugin>=4.4,<6.0 or set HDF5_PLUGIN_PATH to a "
                "Blosc HDF5 plugin directory."
            ) from exc
        raise


def _read_h5_key(
    h5_file: h5py.File,
    key: str,
    file_path: str,
    index: Any = None,
    dtype: Any = None,
) -> npt.NDArray[Any]:
    dataset = _require_h5_dataset(h5_file, key, file_path)
    return _read_h5_array(dataset, key, file_path, index, dtype)


def _load_t_offset(h5_file: h5py.File, file_path: str) -> np.int64:
    dataset = _require_h5_dataset(h5_file, "t_offset", file_path)
    value = _read_h5_array(dataset, "t_offset", file_path, ())
    if value.shape not in ((), (1,)):
        raise ValueError(
            f"DSEC HDF5 dataset 't_offset' in {file_path} must be scalar, "
            f"got shape {value.shape}."
        )
    return np.int64(value.reshape(-1)[0])


def _load_ms_to_idx(
    h5_file: h5py.File,
    file_path: str,
    num_events: int,
) -> npt.NDArray[np.int64]:
    dataset = _require_1d_h5_dataset(h5_file, "ms_to_idx", file_path)
    ms_to_idx = _read_h5_array(dataset, "ms_to_idx", file_path, dtype=np.int64)
    if bool(np.any(ms_to_idx < 0)) or bool(np.any(ms_to_idx > num_events)):
        raise ValueError(
            f"DSEC HDF5 dataset 'ms_to_idx' in {file_path} contains indices "
            f"outside [0, {num_events}]."
        )
    if ms_to_idx.size > 1 and bool(np.any(np.diff(ms_to_idx) < 0)):
        raise ValueError(f"DSEC HDF5 dataset 'ms_to_idx' in {file_path} must be monotonic.")
    return ms_to_idx


def _validate_event_h5_columns(h5_file: h5py.File, file_path: str) -> int:
    event_keys = ("events/x", "events/y", "events/t", "events/p")
    lengths: dict[str, int] = {}
    for key in event_keys:
        dataset = _require_1d_h5_dataset(h5_file, key, file_path)
        lengths[key] = int(dataset.shape[0])

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        details_parts: list[str] = []
        for key, length in lengths.items():
            details_parts.append(f"{key}={length}")
        details = ", ".join(details_parts)
        raise ValueError(f"DSEC event column length mismatch in {file_path}: {details}.")
    return next(iter(unique_lengths))


def _validate_event_timestamps(timestamps_us: npt.NDArray[np.int64], file_path: str) -> None:
    if timestamps_us.size > 1 and bool(np.any(np.diff(timestamps_us) < 0)):
        raise ValueError(f"DSEC event timestamps must be monotonic in {file_path}.")


def _validate_event_coordinates(
    x: npt.NDArray[np.int16],
    y: npt.NDArray[np.int16],
    shape: tuple[int, int],
    file_path: str,
) -> None:
    height, width = shape
    invalid = (x < 0) | (x >= width) | (y < 0) | (y >= height)
    if bool(np.any(invalid)):
        raise ValueError(
            f"DSEC event coordinates in {file_path} fall outside the expected "
            f"{width}x{height} sensor bounds."
        )


@lru_cache(maxsize=32)
def _load_calibration_yaml(path: str, mtime_ns: int) -> dict[str, Any]:
    """Parse DSEC cam to cam calibration YAML.

    Cached per ``(path, mtime_ns)`` so file edits invalidate the entry.
    """
    del mtime_ns  # cache key only
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML is required for loading DSEC calibration. ") from exc

    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.startswith("%YAML:1.0"):
        yaml_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("%YAML:"):
                continue
            yaml_lines.append(line)
        text = "\n".join(yaml_lines)
    data = yaml.safe_load(text)
    if data is None:
        return {}
    return dict(data)


def _polarity_to_bool(raw: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    return np.asarray(raw != 0, dtype=np.bool_)


def _require_cv2() -> Any:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for loading DSEC images. ")
    return cv2


def _read_image(path: str) -> npt.NDArray[np.uint8]:
    _cv2 = _require_cv2()
    image = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"DSEC image PNG must decode to uint8, got {image.dtype}: {path}")
    if image.ndim not in (2, 3):
        raise ValueError(f"DSEC image PNG must be 2D or 3D, got shape {image.shape}: {path}")
    return image  # type: ignore[no-any-return]


def _read_flow_png(
    path: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Decode a DSEC 16 bit three channel optical flow PNG.

    DSEC PNG channel order RGB: R=x_disp, G=y_disp, B=valid
    OpenCV reads BGR: ch0=B=valid, ch1=G=y_disp, ch2=R=x_disp

    Returns:
        flow: ``(H, W, 2)`` float32 with channels ``[flow_x, flow_y]``.
        valid: ``(H, W)`` bool mask.
    """
    _cv2 = _require_cv2()
    raw = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read flow PNG: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"DSEC flow PNG must decode to uint16, got {raw.dtype}: {path}")
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"DSEC flow PNG must have shape (H, W, 3), got {raw.shape}: {path}")

    h, w = raw.shape[0], raw.shape[1]
    flow = np.empty((h, w, 2), dtype=np.float32)
    # 32768 is the zero point for signed 16 bit flow
    np.subtract(raw[..., 2], 32768, out=flow[..., 0], dtype=np.float32)
    np.subtract(raw[..., 1], 32768, out=flow[..., 1], dtype=np.float32)
    # DSEC flow uses 1/128 pixel units
    flow *= np.float32(1.0 / 128.0)
    valid = raw[..., 0] != 0
    return flow, valid


def _decode_flow_stack(flow_paths: list[str]) -> _DecodedDenseField:
    """Decode and stack optical flow PNG files."""
    decoded_fields = decode_in_parallel(
        flow_paths,
        _read_flow_png,
        max_workers=_PNG_DECODE_PARALLELISM,
    )
    flow_fields = []
    validity_masks = []
    for flow_field, validity_mask in decoded_fields:
        flow_fields.append(flow_field)
        validity_masks.append(validity_mask)
    flow_stack = np.stack(flow_fields)
    validity_stack = np.stack(validity_masks)
    return _freeze_dense_field((flow_stack, validity_stack))


def _read_disparity_png(
    path: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Decode a DSEC 16 bit single channel disparity PNG.

    Encoding: ``disparity = I / 256.0``, valid where ``I > 0``.

    Returns:
        disparity: ``(H, W)`` float32.
        valid: ``(H, W)`` bool mask.
    """
    _cv2 = _require_cv2()
    raw = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read disparity PNG: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"DSEC disparity PNG must decode to uint16, got {raw.dtype}: {path}")

    if raw.ndim == 3:
        if raw.shape[2] != 1:
            raise ValueError(
                f"DSEC disparity PNG must be single channel, got shape {raw.shape}: {path}"
            )
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"DSEC disparity PNG must be 2D, got shape {raw.shape}: {path}")
    valid = raw > 0
    disparity = raw.astype(np.float32)
    # DSEC disparity uses 1/256 pixel units
    disparity *= np.float32(1.0 / 256.0)
    return disparity, valid


def _drive_prefix_from_sequence(sequence: str) -> str:
    prefix, separator, suffix = sequence.rpartition("_")
    if separator and suffix.isalpha():
        return prefix
    return sequence


def _require_rosbag_reader() -> Any:
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise ImportError(
            "rosbags is required for loading DSEC LiDAR/IMU bags. "
            "Install event-vision-library with the rosbags dependency."
        ) from exc
    return Reader


@lru_cache(maxsize=1)
def _get_ros1_typestore() -> Any:
    try:
        from rosbags.typesys import Stores
        from rosbags.typesys import get_typestore
    except ImportError as exc:
        raise ImportError(
            "rosbags is required for loading DSEC LiDAR/IMU bags. "
            "Install event-vision-library with the rosbags dependency."
        ) from exc
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
        # xcCompact pointcloud2 can be copied in one pass
        points = np.frombuffer(data[:required_bytes], dtype=dtype, count=height * width).copy()
        return cast(_LidarPoints, points)

    # padded pointcloud2 rows need compaction
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


def _resolve_dsec_paths(  # noqa: C901
    root: str,
    sequence: str,
    split: DSECSplit,
    camera: DSECCamera,
) -> dict[str, str]:
    """Resolve paths across official sequence and split-grouped layouts."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if camera not in ("left", "right"):
        raise ValueError(f"camera must be 'left' or 'right', got {camera!r}")

    # multi-sequence zips unpack as sequence dirs, local copies may group them by split
    split_dir_candidates = (split, split.capitalize())
    drive_prefix = _drive_prefix_from_sequence(sequence)

    def _with_auxiliary_paths(paths: dict[str, str], dataset_root: str) -> dict[str, str]:
        auxiliary_root = dataset_root
        # bags and calibration live at the dataset root rather than inside a split
        if not os.path.isdir(os.path.join(auxiliary_root, "data", drive_prefix)):
            parent = os.path.dirname(auxiliary_root)
            if os.path.isdir(os.path.join(parent, "data", drive_prefix)):
                auxiliary_root = parent

        paths.update(
            {
                "dataset_root": dataset_root,
                "auxiliary_root": auxiliary_root,
                "drive_prefix": drive_prefix,
                "lidar_imu_bag": os.path.join(
                    auxiliary_root,
                    "data",
                    drive_prefix,
                    "lidar_imu.bag",
                ),
                "cam_to_lidar": os.path.join(
                    auxiliary_root,
                    "data",
                    drive_prefix,
                    "cam_to_lidar.yaml",
                ),
                "cam_to_imu": os.path.join(
                    auxiliary_root,
                    "imu_calibration",
                    "cam0_to_imu0.yaml",
                ),
                "imu_params": os.path.join(
                    auxiliary_root,
                    "imu_calibration",
                    "imu0_params.yaml",
                ),
            }
        )
        return paths

    def _rectify_h5_path(event_camera_dir: str) -> str:
        rectify_h5_candidates = [
            os.path.join(event_camera_dir, "rectify_maps.h5"),
            os.path.join(event_camera_dir, "rectify_map.h5"),
        ]
        for path in rectify_h5_candidates:
            if os.path.isfile(path):
                return path
        return rectify_h5_candidates[0]

    def _sequence_layout_paths(sequence_base: str) -> dict[str, str]:
        split_base = os.path.dirname(sequence_base)
        if os.path.basename(split_base).lower() in ("train", "test"):
            dataset_root = os.path.dirname(split_base)
        else:
            dataset_root = split_base
        event_camera_dir = os.path.join(sequence_base, "events", camera)
        return _with_auxiliary_paths(
            {
                "events_h5": os.path.join(event_camera_dir, "events.h5"),
                "rectify_h5": _rectify_h5_path(event_camera_dir),
                "image_timestamps": os.path.join(sequence_base, "images", "timestamps.txt"),
                "image_exposure_timestamps": os.path.join(
                    sequence_base,
                    "images",
                    camera,
                    "exposure_timestamps.txt",
                ),
                "image_dir": os.path.join(sequence_base, "images", camera, "rectified"),
                "flow_forward_dir": os.path.join(sequence_base, "flow", "forward"),
                "flow_forward_ts": os.path.join(
                    sequence_base,
                    "flow",
                    "forward_timestamps.txt",
                ),
                "flow_backward_dir": os.path.join(sequence_base, "flow", "backward"),
                "flow_backward_ts": os.path.join(
                    sequence_base,
                    "flow",
                    "backward_timestamps.txt",
                ),
                "disparity_dir": os.path.join(sequence_base, "disparity", "event"),
                "disparity_ts": os.path.join(sequence_base, "disparity", "timestamps.txt"),
                "calibration": os.path.join(sequence_base, "calibration", "cam_to_cam.yaml"),
            },
            dataset_root,
        )

    sequence_base_candidates = [
        os.path.join(root, split_dir, sequence) for split_dir in split_dir_candidates
    ]
    root_name = os.path.basename(os.path.normpath(root))
    if root_name == sequence or os.path.isdir(os.path.join(root, "events")):
        sequence_base_candidates.append(root)
    sequence_base_candidates.append(os.path.join(root, sequence))
    for sequence_base in sequence_base_candidates:
        paths = _sequence_layout_paths(sequence_base)
        if os.path.isfile(paths["events_h5"]):
            return paths

    for sequence_base in sequence_base_candidates:
        if os.path.isdir(os.path.dirname(sequence_base)):
            return _sequence_layout_paths(sequence_base)

    return _sequence_layout_paths(sequence_base_candidates[0])


class DSECDataLoader(DataLoaderBase):
    """Low level I/O for a single DSEC sequence.

    The expected root contains DSEC sequence directories, either directly as
    produced by the multi-sequence ZIP archives, or grouped by split such as
    ``train/<sequence>/events/left/events.h5`` and
    ``test/<sequence>/images/left/rectified/*.png``.
    Passing a split directory itself, such as ``/path/to/DSEC/train``, is also supported.
    Passing the sequence directory itself is supported when its basename matches
    ``sequence``.

    By default events are returned in raw sensor coordinates and
    :meth:`rectify_events` is an explicit per slice call.
    Pass ``prerectify_events=True`` with ``event_load_mode="cached"`` to
    rectify the cached event arrays once at init.
    Subsequent :meth:`load_events` calls then return pre rectified data and
    :meth:`rectify_events` is idempotent.

    All public timestamps are in seconds (float64).

    ``load_frame_sample`` uses forward flow intervals when forward flow is loaded,
    otherwise it spans consecutive image timestamps.
    Images are selected by nearest timestamp at the sample start/end,
    disparity by nearest timestamp at the sample end, and
    backward flow by matching the same anchor timestamp as the sample start.

    Args:
        root: Root directory containing the DSEC dataset, a directory of
            sequence folders, a split directory such as ``/path/to/DSEC/train``,
            or the sequence directory itself.
        sequence: Sequence name (e.g. ``"zurich_city_01_a"``).
        split: Dataset split, ``"train"`` or ``"test"``.
        camera: Event camera side, ``"left"`` or ``"right"``.
        load_images: LoadMode for rectified frame camera images.
        load_flow_forward: LoadMode for forward optical flow GT.
        load_flow_backward: LoadMode for backward optical flow GT.
        load_disparity: LoadMode for disparity maps.
        load_imu: LoadMode for IMU samples from ``data/<drive_prefix>/lidar_imu.bag``.
        load_lidar: LoadMode for Velodyne point clouds from ``data/<drive_prefix>/lidar_imu.bag``.
        load_calibration: If True, load ``cam_to_cam.yaml`` calibration.
        load_rectify_map: If True, load event rectification map from HDF5.
        event_load_mode: ``"lazy"`` to keep HDF5 open and read on demand
            (default), or ``"cached"`` to load all events into RAM.
        prerectify_events: If True (cached mode only), rectify + filter the
            cached event arrays at init so per call rectification is free.
            Requires ``event_load_mode="cached"`` and ``load_rectify_map=True``.
            With this flag, :meth:`rectify_events` becomes idempotent so
            downstream code does not need to branch on mode.
    """

    EVENT_SHAPE: tuple[int, int] = (480, 640)  # (H, W)
    IMAGE_SHAPE: tuple[int, int] = (1080, 1440)  # (H, W)

    def __init__(  # noqa: C901
        self,
        root: str,
        sequence: str,
        split: DSECSplit = "train",
        camera: DSECCamera = "left",
        load_images: LoadMode = False,
        load_flow_forward: LoadMode = False,
        load_flow_backward: LoadMode = False,
        load_disparity: LoadMode = False,
        load_imu: LoadMode = False,
        load_lidar: LoadMode = False,
        load_calibration: bool = False,
        load_rectify_map: bool = True,
        event_load_mode: ResidentLoadMode = "lazy",
        prerectify_events: bool = False,
    ) -> None:
        """Initialize one DSEC sequence loader."""
        if prerectify_events:
            if event_load_mode != "cached":
                raise ValueError(
                    "prerectify_events=True requires event_load_mode='cached'; "
                    f"got event_load_mode={event_load_mode!r}."
                )
            if not load_rectify_map:
                raise ValueError("prerectify_events=True requires load_rectify_map=True.")

        self.root = root
        self.sequence = sequence
        self.split = split
        self.camera = camera

        event_mode = LoadingType.from_resident_value(event_load_mode, name="event_load_mode")
        image_mode = LoadingType.from_value(load_images, name="load_images")
        flow_fwd_mode = LoadingType.from_value(load_flow_forward, name="load_flow_forward")
        flow_bwd_mode = LoadingType.from_value(load_flow_backward, name="load_flow_backward")
        disparity_mode = LoadingType.from_value(load_disparity, name="load_disparity")
        imu_mode = LoadingType.from_value(load_imu, name="load_imu")
        lidar_mode = LoadingType.from_value(load_lidar, name="load_lidar")

        _register_hdf5_filter_plugins()

        paths = _resolve_dsec_paths(root, sequence, split, camera)
        self._paths = paths
        self._events_h5_path = paths["events_h5"]
        self._event_mode = event_mode

        self._t_offset: np.int64
        self._ms_to_idx: npt.NDArray[np.int64]
        self._num_events: int
        self._ev_x: npt.NDArray[np.int16] | None = None
        self._ev_y: npt.NDArray[np.int16] | None = None
        self._ev_t_sec: npt.NDArray[np.float64] | None = None
        self._ev_p: npt.NDArray[np.bool_] | None = None
        self._events_h5: h5py.File | None = None
        self._events_h5_pid: int | None = None

        with open_hdf5(self._events_h5_path) as h5:
            self._num_events = _validate_event_h5_columns(h5, self._events_h5_path)
            self._t_offset = _load_t_offset(h5, self._events_h5_path)
            self._ms_to_idx = _load_ms_to_idx(h5, self._events_h5_path, self._num_events)

            if event_mode is LoadingType.CACHED:
                self._ev_x = _read_h5_key(h5, "events/x", self._events_h5_path, dtype=np.int16)
                self._ev_y = _read_h5_key(h5, "events/y", self._events_h5_path, dtype=np.int16)
                self._ev_p = _polarity_to_bool(_read_h5_key(h5, "events/p", self._events_h5_path))
                t_raw = _read_h5_key(h5, "events/t", self._events_h5_path, dtype=np.int64)
                _validate_event_timestamps(t_raw, self._events_h5_path)
                _validate_event_coordinates(
                    self._ev_x,
                    self._ev_y,
                    self.EVENT_SHAPE,
                    self._events_h5_path,
                )
                self._ev_t_sec = t_raw.astype(np.float64)
                self._ev_t_sec += float(self._t_offset)
                self._ev_t_sec *= 1e-6
                del t_raw

        freeze_array(self._ms_to_idx)

        if event_mode is LoadingType.CACHED:
            freeze_array(self._ev_x)
            freeze_array(self._ev_y)
            freeze_array(self._ev_t_sec)
            freeze_array(self._ev_p)

        self._rectify_map: npt.NDArray[np.float32] | None = None
        self._rect_x_flat: npt.NDArray[np.int16] | None = None
        self._rect_y_flat: npt.NDArray[np.int16] | None = None
        self._rect_valid_flat: npt.NDArray[np.bool_] | None = None
        self._rect_row_stride: int = 0
        self._events_prerectified: bool = False
        if load_rectify_map:
            rectify_path = paths["rectify_h5"]
            if os.path.isfile(rectify_path):
                with open_hdf5(rectify_path) as rect_h5:
                    rectify_dataset = _require_h5_dataset(rect_h5, "rectify_map", rectify_path)
                    self._rectify_map = _read_h5_array(
                        rectify_dataset,
                        "rectify_map",
                        rectify_path,
                        dtype=np.float32,
                    )

                expected_rectify_shape = (*self.EVENT_SHAPE, 2)
                if self._rectify_map.shape != expected_rectify_shape:
                    raise ValueError(
                        "DSEC rectify_map must have shape "
                        f"{expected_rectify_shape}, got {self._rectify_map.shape}: "
                        f"{rectify_path}"
                    )

                # Tables make rectification a single flat index gather
                h_ev, w_ev = self.EVENT_SHAPE
                rx = np.rint(self._rectify_map[..., 0]).astype(np.int32)
                ry = np.rint(self._rectify_map[..., 1]).astype(np.int32)
                valid_2d = (rx >= 0) & (rx < w_ev) & (ry >= 0) & (ry < h_ev)
                np.clip(rx, 0, w_ev - 1, out=rx)
                np.clip(ry, 0, h_ev - 1, out=ry)
                self._rect_x_flat = rx.astype(np.int16).ravel()
                self._rect_y_flat = ry.astype(np.int16).ravel()
                self._rect_valid_flat = valid_2d.ravel()
                self._rect_row_stride = int(w_ev)

                freeze_array(self._rectify_map)
                freeze_array(self._rect_x_flat)
                freeze_array(self._rect_y_flat)
                freeze_array(self._rect_valid_flat)
            else:
                logger.warning("Rectification map not found: %s", rectify_path)

        if prerectify_events:
            if self._rect_x_flat is None:
                raise RuntimeError(
                    "prerectify_events=True but rectification map file was "
                    f"not found at {paths['rectify_h5']}"
                )
            self._apply_prerectification()

        self._image_paths: list[str] | None = None
        self._image_ts: npt.NDArray[np.float64] | None = None
        self._image_exposure_ts: npt.NDArray[np.float64] | None = None
        self._image_cached: npt.NDArray[np.uint8] | None = None
        self._lazy_image_cache = LazyDecodeCache[npt.NDArray[np.uint8]](
            _DEFAULT_LAZY_IMAGE_CACHE_ITEMS
        )
        self._image_mode = image_mode
        if image_mode.should_load:
            self._init_images(paths, image_mode)

        self._flow_fwd_paths: list[str] | None = None
        self._flow_fwd_ts: npt.NDArray[np.float64] | None = None
        self._flow_fwd_cached: npt.NDArray[np.float32] | None = None
        self._flow_fwd_valid_cached: npt.NDArray[np.bool_] | None = None
        self._lazy_flow_fwd_cache = LazyDecodeCache[_DecodedDenseField](
            _DEFAULT_LAZY_FLOW_CACHE_ITEMS
        )
        self._flow_fwd_mode = flow_fwd_mode
        if flow_fwd_mode.should_load:
            self._init_flow(paths, "forward", flow_fwd_mode)

        self._flow_bwd_paths: list[str] | None = None
        self._flow_bwd_ts: npt.NDArray[np.float64] | None = None
        self._flow_bwd_cached: npt.NDArray[np.float32] | None = None
        self._flow_bwd_valid_cached: npt.NDArray[np.bool_] | None = None
        self._lazy_flow_bwd_cache = LazyDecodeCache[_DecodedDenseField](
            _DEFAULT_LAZY_FLOW_CACHE_ITEMS
        )
        self._flow_bwd_mode = flow_bwd_mode
        if flow_bwd_mode.should_load:
            self._init_flow(paths, "backward", flow_bwd_mode)

        self._disp_paths: list[str] | None = None
        self._disp_ts: npt.NDArray[np.float64] | None = None
        self._disp_cached: npt.NDArray[np.float32] | None = None
        self._disp_valid_cached: npt.NDArray[np.bool_] | None = None
        self._lazy_disparity_cache = LazyDecodeCache[_DecodedDenseField](
            _DEFAULT_LAZY_DISPARITY_CACHE_ITEMS
        )
        self._disp_mode = disparity_mode
        if disparity_mode.should_load:
            self._init_disparity(paths, disparity_mode)

        self._bag_path = paths["lidar_imu_bag"]
        self._rosbag_reader: Any | None = None
        self._rosbag_reader_pid: int | None = None
        self._rosbag_imu_connection: Any | None = None
        self._rosbag_lidar_connection: Any | None = None
        self._imu_ts_ns: npt.NDArray[np.int64] | None = None
        self._imu_ts: npt.NDArray[np.float64] | None = None
        self._imu_cached: DSECImuData | None = None
        self._imu_mode = imu_mode
        self._lidar_ts_ns: npt.NDArray[np.int64] | None = None
        self._lidar_ts: npt.NDArray[np.float64] | None = None
        self._lidar_cached: list[DSECLidarScan] | None = None
        self._lidar_frame_id: str | None = None
        self._lazy_lidar_cache = LazyDecodeCache[DSECLidarScan](_DEFAULT_LAZY_LIDAR_CACHE_ITEMS)
        self._lidar_mode = lidar_mode
        if imu_mode.should_load or lidar_mode.should_load:
            self._init_lidar_imu(paths, imu_mode, lidar_mode)

        self._calibration: dict[str, Any] | None = None
        self._cam_to_lidar_calibration: dict[str, Any] | None = None
        self._cam_to_imu_calibration: dict[str, Any] | None = None
        self._imu_calibration: dict[str, Any] | None = None
        if load_calibration:
            self._init_calibration(paths)

    def _init_images(self, paths: dict[str, str], mode: LoadingType) -> None:
        ts_path = paths["image_timestamps"]
        exposure_ts_path = paths["image_exposure_timestamps"]
        img_dir = paths["image_dir"]

        if not os.path.isfile(ts_path):
            logger.warning("Image timestamps not found: %s", ts_path)
            return
        if not os.path.isdir(img_dir):
            logger.warning("Image directory not found: %s", img_dir)
            return

        raw_ts_us = _load_timestamps_us(ts_path)
        _validate_strictly_increasing("image", raw_ts_us, ts_path)
        self._image_ts = raw_ts_us.astype(np.float64) / 1e6
        freeze_array(self._image_ts)

        self._image_paths = _sorted_png_paths(img_dir)
        _validate_png_count("image", self._image_paths, self._image_ts, ts_path)

        if os.path.isfile(exposure_ts_path):
            raw_exposure_ts_us = _load_timestamp_pairs_us(
                exposure_ts_path,
                "image exposure",
            )
            _validate_timestamp_pairs(
                "image exposure",
                raw_exposure_ts_us,
                exposure_ts_path,
                require_positive_duration=True,
            )
            _validate_png_count(
                "image exposure",
                self._image_paths,
                raw_exposure_ts_us,
                exposure_ts_path,
            )
            self._image_exposure_ts = raw_exposure_ts_us.astype(np.float64) / 1e6
            freeze_array(self._image_exposure_ts)

        if mode.should_cache and self._image_paths:
            images = decode_in_parallel(
                self._image_paths,
                _read_image,
                max_workers=_PNG_DECODE_PARALLELISM,
            )
            self._image_cached = np.stack(images)
            freeze_array(self._image_cached)

    def _init_flow(
        self,
        paths: dict[str, str],
        direction: FlowDirection,
        mode: LoadingType,
    ) -> None:
        is_forward = direction == "forward"
        ts_path = paths["flow_forward_ts"] if is_forward else paths["flow_backward_ts"]
        flow_dir = paths["flow_forward_dir"] if is_forward else paths["flow_backward_dir"]

        if not os.path.isfile(ts_path):
            logger.warning("Flow %s timestamps not found: %s", direction, ts_path)
            return
        if not os.path.isdir(flow_dir):
            logger.warning("Flow %s directory not found: %s", direction, flow_dir)
            return

        raw_ts_us = _load_flow_timestamps(ts_path)
        _validate_timestamp_pairs(
            f"{direction} flow",
            raw_ts_us,
            ts_path,
            require_positive_duration=False,
        )
        ts_seconds = raw_ts_us.astype(np.float64) / 1e6
        flow_paths = _sorted_png_paths(flow_dir)
        _validate_png_count(f"{direction} flow", flow_paths, ts_seconds, ts_path)

        if is_forward:
            self._flow_fwd_ts = ts_seconds
            freeze_array(self._flow_fwd_ts)
            self._flow_fwd_paths = flow_paths
            if mode.should_cache and flow_paths:
                flow_stack, valid_stack = _decode_flow_stack(flow_paths)
                self._flow_fwd_cached = flow_stack
                self._flow_fwd_valid_cached = valid_stack
        else:
            self._flow_bwd_ts = ts_seconds
            freeze_array(self._flow_bwd_ts)
            self._flow_bwd_paths = flow_paths
            if mode.should_cache and flow_paths:
                flow_stack, valid_stack = _decode_flow_stack(flow_paths)
                self._flow_bwd_cached = flow_stack
                self._flow_bwd_valid_cached = valid_stack

    def _init_disparity(self, paths: dict[str, str], mode: LoadingType) -> None:
        ts_path = paths["disparity_ts"]
        disp_dir = paths["disparity_dir"]

        if not os.path.isfile(ts_path):
            logger.warning("Disparity timestamps not found: %s", ts_path)
            return
        if not os.path.isdir(disp_dir):
            logger.warning("Disparity directory not found: %s", disp_dir)
            return

        raw_ts_us = _load_timestamps_us(ts_path)
        _validate_strictly_increasing("disparity", raw_ts_us, ts_path)
        self._disp_ts = raw_ts_us.astype(np.float64) / 1e6
        freeze_array(self._disp_ts)

        self._disp_paths = _sorted_png_paths(disp_dir)
        _validate_png_count("disparity", self._disp_paths, self._disp_ts, ts_path)

        if mode.should_cache and self._disp_paths:
            decoded = decode_in_parallel(
                self._disp_paths,
                _read_disparity_png,
                max_workers=_PNG_DECODE_PARALLELISM,
            )
            disparities = []
            valids = []
            for disparity, valid in decoded:
                disparities.append(disparity)
                valids.append(valid)
            self._disp_cached = np.stack(disparities)
            self._disp_valid_cached = np.stack(valids)
            freeze_array(self._disp_cached)
            freeze_array(self._disp_valid_cached)

    def _init_lidar_imu(
        self,
        paths: dict[str, str],
        imu_mode: LoadingType,
        lidar_mode: LoadingType,
    ) -> None:
        bag_path = paths["lidar_imu_bag"]
        if not os.path.isfile(bag_path):
            logger.warning("DSEC LiDAR/IMU bag not found: %s", bag_path)
            return

        reader_type = _require_rosbag_reader()
        with reader_type(bag_path) as reader:
            if imu_mode.should_load:
                self._init_imu(reader, imu_mode)

            if lidar_mode.should_load:
                self._init_lidar(reader, lidar_mode)

    def _init_imu(self, reader: Any, mode: LoadingType) -> None:
        imu_connection = _connection_for_topic(reader, _DSEC_IMU_TOPIC, self._bag_path)
        if imu_connection is None:
            return

        imu_timestamps_ns = _connection_timestamps_ns(reader, imu_connection)
        self._imu_ts_ns = imu_timestamps_ns
        self._imu_ts = _ros_ns_to_seconds(imu_timestamps_ns)

        if not mode.should_cache:
            return

        if imu_timestamps_ns.size:
            start_ns = int(imu_timestamps_ns[0])
            # rosbags stop is exclusive so add one ns for the last sample
            stop_ns = int(imu_timestamps_ns[-1]) + 1
        else:
            start_ns = None
            stop_ns = None

        self._imu_cached = _read_imu_messages_from_reader(
            reader,
            imu_connection,
            start_ns,
            stop_ns,
            int(imu_timestamps_ns.size),
        )

    def _init_lidar(self, reader: Any, mode: LoadingType) -> None:
        lidar_connection = _connection_for_topic(
            reader,
            _DSEC_LIDAR_TOPIC,
            self._bag_path,
        )
        if lidar_connection is None:
            return

        lidar_timestamps_ns = _connection_timestamps_ns(reader, lidar_connection)
        self._lidar_ts_ns = lidar_timestamps_ns
        self._lidar_ts = _ros_ns_to_seconds(lidar_timestamps_ns)

        if mode.should_cache:
            cached_scans: list[DSECLidarScan] = []
            for connection, timestamp_ns, raw in reader.messages(connections=[lidar_connection]):
                scan = _decode_lidar_scan(timestamp_ns, raw, connection)
                cached_scans.append(scan)
            self._lidar_cached = cached_scans
            if cached_scans:
                self._lidar_frame_id = cached_scans[0]["frame_id"]
            return

        if lidar_timestamps_ns.size == 0:
            return

        # read one lazy scan so lidar_frame_id is available after init
        first_scan = self._read_lidar_scans_from_reader(
            reader,
            lidar_connection,
            0,
            1,
        )[0]
        self._lidar_frame_id = first_scan["frame_id"]

    def _read_imu_interval_from_bag(
        self,
        start_ns: int | None,
        stop_ns: int | None,
        n_messages: int,
    ) -> DSECImuData:
        if start_ns is not None and stop_ns is not None and stop_ns < start_ns:
            raise ValueError(f"Expected stop_ns >= start_ns, got [{start_ns}, {stop_ns}).")
        if not os.path.isfile(self._bag_path):
            return _empty_imu_data()

        reader, imu_connection = self._rosbag_connection(_DSEC_IMU_TOPIC)
        if imu_connection is None:
            return _empty_imu_data()
        return _read_imu_messages_from_reader(
            reader,
            imu_connection,
            start_ns,
            stop_ns,
            n_messages,
        )

    def _read_lidar_scans_from_reader(
        self,
        reader: Any,
        lidar_connection: Any,
        start_index: int,
        end_index: int,
    ) -> list[DSECLidarScan]:
        if self._lidar_ts_ns is None:
            raise IndexError(f"LiDAR interval [{start_index}, {end_index}) out of range")
        start_index, end_index = validate_index_interval(
            start_index,
            end_index,
            len(self._lidar_ts_ns),
            "LiDAR",
        )
        if start_index == end_index:
            return []

        # Use bag timestamps to seek then check count
        start_ns = int(self._lidar_ts_ns[start_index])
        if end_index < len(self._lidar_ts_ns):
            stop_ns = int(self._lidar_ts_ns[end_index])
        else:
            stop_ns = int(self._lidar_ts_ns[-1]) + 1

        scans: list[DSECLidarScan] = []
        expected_index = start_index
        for connection, timestamp_ns, raw in reader.messages(
            connections=[lidar_connection],
            start=start_ns,
            stop=stop_ns,
        ):
            if expected_index >= end_index:
                break

            expected_timestamp_ns = int(self._lidar_ts_ns[expected_index])
            if timestamp_ns != expected_timestamp_ns:
                # Ignore bag entries outside the DSEC timestamp index
                resolved_index = int(np.searchsorted(self._lidar_ts_ns, timestamp_ns, side="left"))
                timestamp_matches_index = (
                    start_index <= resolved_index < end_index
                    and int(self._lidar_ts_ns[resolved_index]) == timestamp_ns
                )
                if not timestamp_matches_index:
                    continue
                expected_index = resolved_index

            scan = _decode_lidar_scan(timestamp_ns, raw, connection)
            if self._lidar_frame_id is None:
                self._lidar_frame_id = scan["frame_id"]
            scans.append(self._lazy_lidar_cache.put(expected_index, scan))
            expected_index += 1

        expected_count = end_index - start_index
        if len(scans) != expected_count:
            raise ValueError(
                "DSEC LiDAR timestamp index expected "
                f"{expected_count} scans in [{start_index}, {end_index}), "
                f"but the ROS bag yielded {len(scans)}."
            )
        return scans

    def _read_lidar_scans(self, start_index: int, end_index: int) -> list[DSECLidarScan]:
        reader, lidar_connection = self._rosbag_connection(_DSEC_LIDAR_TOPIC)
        if lidar_connection is None:
            raise IndexError(f"LiDAR interval [{start_index}, {end_index}) out of range")
        return self._read_lidar_scans_from_reader(reader, lidar_connection, start_index, end_index)

    def _apply_prerectification(self) -> None:  # noqa: C901
        """Rectify and filter cached event arrays in place.

        Any events that fall out of bounds are deleted.

        Chunks avoid a second full event table allocation.
        The write cursor compacts kept events, then ``ms_to_idx`` is rebuilt
        for the shorter table.
        """
        ev_x = self._ev_x
        ev_y = self._ev_y
        ev_t_sec = self._ev_t_sec
        ev_p = self._ev_p
        rect_x_flat = self._rect_x_flat
        rect_y_flat = self._rect_y_flat
        rect_valid_flat = self._rect_valid_flat
        if (
            ev_x is None
            or ev_y is None
            or ev_t_sec is None
            or ev_p is None
            or rect_x_flat is None
            or rect_y_flat is None
            or rect_valid_flat is None
        ):
            raise RuntimeError(
                "Cached events and rectification lookup tables are required "
                "for prerectification."
            )

        num_events = int(ev_x.shape[0])
        if num_events == 0:
            self._events_prerectified = True
            return

        stride = np.int32(self._rect_row_stride)
        chunk_size: int = 1 << 25

        view_descriptors: list[tuple[str, np.dtype]] = []
        for attr, dtype in (
            ("_ev_x", np.dtype(np.int16)),
            ("_ev_y", np.dtype(np.int16)),
            ("_ev_t_sec", np.dtype(np.float64)),
            ("_ev_p", np.dtype(np.bool_)),
        ):
            arr = getattr(self, attr)
            arr.setflags(write=True)
            if arr.base is not None:
                arr.base.setflags(write=True)
                view_descriptors.append((attr, dtype))

        write = 0
        for start in range(0, num_events, chunk_size):
            end = min(start + chunk_size, num_events)
            # work on one chunk so cached prerectification does not double memory use
            y_c = ev_y[start:end].astype(np.int32)
            x_c = ev_x[start:end].astype(np.int32)
            h_ev, w_ev = self.EVENT_SHAPE
            raw_valid = (x_c >= 0) & (x_c < w_ev) & (y_c >= 0) & (y_c < h_ev)
            if not bool(np.any(raw_valid)):
                continue

            valid_positions = np.nonzero(raw_valid)[0]
            flat = y_c[valid_positions]
            flat *= stride
            flat += x_c[valid_positions]
            del x_c
            valid_chunk = rect_valid_flat[flat]
            # keep_positions are source rows that survive rectification
            keep_positions = valid_positions[valid_chunk]
            valid_flat = flat[valid_chunk]
            del flat
            kept_count = int(valid_flat.shape[0])

            # Write kept events toward the front as we scan forward
            ev_x[write : write + kept_count] = rect_x_flat[valid_flat]
            ev_y[write : write + kept_count] = rect_y_flat[valid_flat]
            del valid_flat
            ev_t_sec[write : write + kept_count] = ev_t_sec[start:end][keep_positions]
            ev_p[write : write + kept_count] = ev_p[start:end][keep_positions]
            del keep_positions
            del valid_chunk
            write += kept_count
        compacted_count = write
        if compacted_count < 0 or compacted_count > num_events:
            raise RuntimeError("Invalid DSEC prerectification compaction count.")

        view_attrs = {attr for attr, _ in view_descriptors}
        for attr, dtype in (
            ("_ev_x", np.dtype(np.int16)),
            ("_ev_y", np.dtype(np.int16)),
            ("_ev_t_sec", np.dtype(np.float64)),
            ("_ev_p", np.dtype(np.bool_)),
        ):
            arr = getattr(self, attr)
            if attr in view_attrs:
                base = arr.base
                if not isinstance(base, np.ndarray):
                    raise RuntimeError("Expected cached event array to have a NumPy base array.")
                setattr(self, attr, None)
                del arr
                base.resize(compacted_count, refcheck=False)
                new_view = base.view(dtype)
                new_view.setflags(write=False)
                base.setflags(write=False)
                setattr(self, attr, new_view)
                continue

            arr.resize(compacted_count, refcheck=False)
            arr.setflags(write=False)

        self._num_events = compacted_count
        self._events_prerectified = True

        # Rebuild the lookup after event compaction
        n_ms = int(self._ms_to_idx.shape[0])
        t_offset_sec = float(self._t_offset) * 1e-6
        ms_bounds_sec = np.arange(n_ms, dtype=np.float64) * 1e-3 + t_offset_sec
        event_timestamps = self._ev_t_sec
        if event_timestamps is None:
            raise RuntimeError("Cached DSEC event timestamps are missing after prerectification.")
        self._ms_to_idx = np.searchsorted(event_timestamps, ms_bounds_sec, side="left").astype(
            np.int64
        )
        freeze_array(self._ms_to_idx)

    def _init_calibration(self, paths: dict[str, str]) -> None:
        calib_path = paths["calibration"]
        try:
            mtime_ns = os.stat(calib_path).st_mtime_ns
        except FileNotFoundError:
            logger.warning("Calibration file not found: %s", calib_path)
        else:
            self._calibration = copy.deepcopy(_load_calibration_yaml(calib_path, mtime_ns))

        for path_key, attr_name in (
            ("cam_to_lidar", "_cam_to_lidar_calibration"),
            ("cam_to_imu", "_cam_to_imu_calibration"),
            ("imu_params", "_imu_calibration"),
        ):
            path = paths[path_key]
            try:
                mtime_ns = os.stat(path).st_mtime_ns
            except FileNotFoundError:
                logger.debug("DSEC auxiliary calibration file not found: %s", path)
                continue
            setattr(self, attr_name, copy.deepcopy(_load_calibration_yaml(path, mtime_ns)))

    def __getstate__(self) -> dict[str, Any]:
        """Drop process local handles before pickling."""
        # Workers need to reopen HDF5 and ROS bag handles
        state = self.__dict__.copy()
        state["_events_h5"] = None
        state["_events_h5_pid"] = None
        state["_rosbag_reader"] = None
        state["_rosbag_reader_pid"] = None
        state["_rosbag_imu_connection"] = None
        state["_rosbag_lidar_connection"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled loader state."""
        self.__dict__.update(state)

    def _ensure_rosbag_reader(self) -> Any:
        """Return an open ROS bag reader, reopening after fork."""
        pid = os.getpid()
        if self._rosbag_reader is not None and self._rosbag_reader_pid == pid:
            return self._rosbag_reader

        self._drop_rosbag_reader()
        reader_type = _require_rosbag_reader()
        reader = reader_type(self._bag_path)
        try:
            reader.open()
        except Exception:
            self._rosbag_reader = None
            self._rosbag_reader_pid = None
            self._rosbag_imu_connection = None
            self._rosbag_lidar_connection = None
            raise
        self._rosbag_reader = reader
        self._rosbag_reader_pid = pid
        self._rosbag_imu_connection = None
        self._rosbag_lidar_connection = None
        return reader

    def _drop_rosbag_reader(self) -> None:
        reader = self._rosbag_reader
        if reader is not None:
            try:
                reader.close()
            except Exception:
                logger.debug("Failed to close DSEC ROS bag reader.", exc_info=True)
        self._rosbag_reader = None
        self._rosbag_reader_pid = None
        self._rosbag_imu_connection = None
        self._rosbag_lidar_connection = None

    def _rosbag_connection(self, topic: str) -> tuple[Any, Any | None]:
        reader = self._ensure_rosbag_reader()
        if topic == _DSEC_IMU_TOPIC:
            if self._rosbag_imu_connection is None:
                self._rosbag_imu_connection = _connection_for_topic(reader, topic, self._bag_path)
            return reader, self._rosbag_imu_connection
        if topic == _DSEC_LIDAR_TOPIC:
            if self._rosbag_lidar_connection is None:
                self._rosbag_lidar_connection = _connection_for_topic(reader, topic, self._bag_path)
            return reader, self._rosbag_lidar_connection
        return reader, _connection_for_topic(reader, topic, self._bag_path)

    def _ensure_events_h5(self) -> h5py.File:
        """Return the open events HDF5 handle, reopening after fork."""
        pid = os.getpid()
        if self._events_h5 is not None and self._events_h5_pid == pid:
            return self._events_h5

        self._drop_events_h5()
        # Spawn workers may not inherit hdf5plugin imports
        _register_hdf5_filter_plugins()
        h5_file = open_hdf5(self._events_h5_path)
        self._events_h5 = h5_file
        self._events_h5_pid = pid
        return h5_file

    def _drop_events_h5(self) -> None:
        h5_file = self._events_h5
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                logger.debug("Failed to close DSEC HDF5 file.", exc_info=True)
        self._events_h5 = None
        self._events_h5_pid = None

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in ``[start_index, end_index)``. Returns mutable copies."""
        start_index, end_index = validate_index_interval(
            start_index,
            end_index,
            self._num_events,
            "Event",
        )
        s = slice(start_index, end_index)
        if self._ev_x is not None:
            ev_y = self._ev_y
            ev_t_sec = self._ev_t_sec
            ev_p = self._ev_p
            if ev_y is None or ev_t_sec is None or ev_p is None:
                raise RuntimeError("Cached DSEC event columns are incomplete.")
            return RawEvents(
                x=self._ev_x[s].copy(),
                y=ev_y[s].copy(),
                timestamp=ev_t_sec[s].copy(),
                polarity=ev_p[s].copy(),
            )

        h5 = self._ensure_events_h5()
        x = _read_h5_key(h5, "events/x", self._events_h5_path, s, np.int16)
        y = _read_h5_key(h5, "events/y", self._events_h5_path, s, np.int16)
        t_raw = _read_h5_key(h5, "events/t", self._events_h5_path, s, np.int64)
        p = _polarity_to_bool(_read_h5_key(h5, "events/p", self._events_h5_path, s))

        t_seconds = t_raw.astype(np.float64)
        t_seconds += float(self._t_offset)
        t_seconds *= 1e-6
        return RawEvents(x=x, y=y, timestamp=t_seconds, polarity=p)

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return self._num_events

    def time_to_index(self, t: float) -> int:
        """Find the last event index strictly before time *t* (seconds)."""
        if self._ev_t_sec is not None:
            return int(np.searchsorted(self._ev_t_sec, t, side="left")) - 1

        t_rel_us = t * 1e6 - float(self._t_offset)
        return self._lazy_first_at_or_after(t_rel_us) - 1

    def index_to_time(self, index: int) -> float:
        """Return the timestamp (absolute seconds) of event at *index*."""
        normalized_index = normalize_index(index, self._num_events, "Event")
        if self._ev_t_sec is not None:
            return float(self._ev_t_sec[normalized_index])

        h5 = self._ensure_events_h5()
        t_rel = int(
            _read_h5_key(
                h5,
                "events/t",
                self._events_h5_path,
                normalized_index,
                np.int64,
            )
        )
        return (t_rel + float(self._t_offset)) / 1e6

    def _lazy_first_at_or_after(self, t_rel_us: float) -> int:
        """First event index whose relative us timestamp is >= ``t_rel_us``."""
        if t_rel_us <= 0:
            return 0
        n_ms = len(self._ms_to_idx)
        bucket = int(t_rel_us // 1000)
        if bucket >= n_ms:
            return self._num_events
        lo = int(self._ms_to_idx[bucket])
        hi = int(self._ms_to_idx[bucket + 1]) if (bucket + 1) < n_ms else self._num_events
        if lo >= hi:
            return lo
        h5 = self._ensure_events_h5()
        # ms_to_idx keeps lazy reads to one ms
        bucket_t = _read_h5_key(h5, "events/t", self._events_h5_path, slice(lo, hi))
        offset = int(np.searchsorted(bucket_t, t_rel_us, side="left"))
        return lo + offset

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Vectorized :meth:`time_to_index`."""
        ts_array = np.asarray(timestamps, dtype=np.float64)

        if self._ev_t_sec is not None:
            return (np.searchsorted(self._ev_t_sec, ts_array, side="left") - 1).astype(np.int64)

        flat = ts_array.reshape(-1)
        out = np.empty(flat.shape, dtype=np.int64)
        t_offset = float(self._t_offset)
        for i, t in enumerate(flat):
            out[i] = self._lazy_first_at_or_after(float(t) * 1e6 - t_offset) - 1
        return out.reshape(ts_array.shape)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Vectorized :meth:`index_to_time`."""
        idx_array = normalize_indices(indices, self._num_events, "Event")

        if self._ev_t_sec is not None:
            return np.ascontiguousarray(self._ev_t_sec[idx_array])

        h5 = self._ensure_events_h5()
        flat_idx = idx_array.reshape(-1)
        if flat_idx.size == 0:
            return np.empty(idx_array.shape, dtype=np.float64)

        unique_idx, inverse = np.unique(flat_idx, return_inverse=True)
        unique_t_rel = _read_h5_key(
            h5,
            "events/t",
            self._events_h5_path,
            unique_idx,
            np.int64,
        )
        t_rel = unique_t_rel[inverse].reshape(idx_array.shape)
        return (t_rel.astype(np.float64) + float(self._t_offset)) / 1e6

    def get_events_by_time(self, t_start: float, t_end: float) -> RawEvents:
        """Load events whose timestamps fall in ``[t_start, t_end)``."""
        if t_end < t_start:
            raise ValueError(f"Expected t_end >= t_start, got [{t_start}, {t_end}).")

        if self._ev_t_sec is not None:
            start, end = np.searchsorted(self._ev_t_sec, [t_start, t_end], side="left")
            return self.load_events(int(start), int(end))

        t_offset = float(self._t_offset)
        start = self._lazy_first_at_or_after(t_start * 1e6 - t_offset)
        end = self._lazy_first_at_or_after(t_end * 1e6 - t_offset)
        return self.load_events(start, end)

    def close(self) -> None:
        """Release open file handles and drop small lazy decode caches."""
        self._drop_events_h5()
        self._drop_rosbag_reader()
        for attr in (
            "_lazy_image_cache",
            "_lazy_flow_fwd_cache",
            "_lazy_flow_bwd_cache",
            "_lazy_disparity_cache",
            "_lazy_lidar_cache",
        ):
            cache = getattr(self, attr, None)
            if cache is not None:
                cache.clear()

    def __del__(self) -> None:
        """Warn when file handles are left open."""
        h5 = getattr(self, "_events_h5", None)
        rosbag = getattr(self, "_rosbag_reader", None)
        if h5 is not None or rosbag is not None:
            warnings.warn(
                f"DSECDataLoader for {getattr(self, 'sequence', '?')!r} was not closed. "
                "Call .close() or use a context manager to release file handles.",
                ResourceWarning,
                stacklevel=2,
            )
            try:
                self.close()
            except Exception:
                logger.debug("Failed to close DSECDataLoader during finalization.", exc_info=True)

    @property
    def t_offset(self) -> np.int64:
        """Temporal offset in microseconds added to raw event timestamps."""
        return self._t_offset

    @property
    def ms_to_idx(self) -> npt.NDArray[np.int64]:
        """Millisecond to event index lookup table.

        ``ms_to_idx[ms]`` gives the first event index whose timestamp
        is >= ``ms * 1000`` microseconds (relative to ``t_offset``).
        """
        return self._ms_to_idx

    @property
    def event_load_mode(self) -> LoadingType:
        """Configured event loading mode."""
        return self._event_mode

    @property
    def has_rectify_map(self) -> bool:
        """Whether the event rectification map is loaded."""
        return self._rectify_map is not None

    @property
    def events_prerectified(self) -> bool:
        """Whether cached events were prerectified at init.

        When True, ``load_events`` returns rectified event coordinates and
        :meth:`rectify_events` is idempotent.
        """
        return self._events_prerectified

    @property
    def rectify_map(self) -> npt.NDArray[np.float32] | None:
        """Event rectification map ``(H, W, 2)`` or None."""
        return self._rectify_map

    def rectify_events(self, events: RawEvents) -> RawEvents:
        """Apply the DSEC event rectification map.

        Returns a new ``RawEvents`` with rectified coordinates.
        Events whose raw coordinates are outside the DSEC sensor bounds,
        or whose rectified pixel falls outside the sensor, are dropped.

        If the loader was constructed with ``prerectify_events=True`` the
        input is returned unchanged.

        Raises:
            RuntimeError: If the rectification map is not loaded.
        """
        if self._events_prerectified:
            return events

        if self._rect_x_flat is None:
            raise RuntimeError(
                "Rectification map not loaded. Pass load_rectify_map=True to the constructor."
            )
        rect_x_flat = self._rect_x_flat
        rect_y_flat = self._rect_y_flat
        rect_valid_flat = self._rect_valid_flat
        if rect_y_flat is None or rect_valid_flat is None:
            raise RuntimeError("Rectification lookup tables are incomplete.")

        y32 = events.y.astype(np.int32, copy=False)
        x32 = events.x.astype(np.int32, copy=False)
        h_ev, w_ev = self.EVENT_SHAPE
        raw_valid = (x32 >= 0) & (x32 < w_ev) & (y32 >= 0) & (y32 < h_ev)
        if not bool(np.any(raw_valid)):
            return RawEvents(
                x=np.empty(0, dtype=np.int16),
                y=np.empty(0, dtype=np.int16),
                timestamp=events.timestamp[:0].copy(),
                polarity=events.polarity[:0].copy(),
            )

        raw_positions = np.nonzero(raw_valid)[0]
        flat = y32[raw_positions] * np.int32(self._rect_row_stride)
        flat += x32[raw_positions]
        rectified_valid = rect_valid_flat[flat]
        keep_positions = raw_positions[rectified_valid]
        flat_valid = flat[rectified_valid]
        return RawEvents(
            x=rect_x_flat[flat_valid],
            y=rect_y_flat[flat_valid],
            timestamp=events.timestamp[keep_positions],
            polarity=events.polarity[keep_positions],
        )

    @property
    def has_images(self) -> bool:
        """Whether images are available."""
        return self._image_paths is not None and len(self._image_paths) > 0

    @property
    def image_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Image timestamps in seconds, or None."""
        return self._image_ts

    @property
    def image_exposure_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Image exposure intervals ``(N, 2)`` in seconds, or None."""
        return self._image_exposure_ts

    @property
    def num_images(self) -> int:
        """Number of available images."""
        if self._image_paths is None:
            return 0
        return len(self._image_paths)

    def load_image(self, index: int) -> npt.NDArray[np.uint8]:
        """Load a single image by index.

        Returns an ``(H, W)`` or ``(H, W, C)`` uint8 array.
        Cached mode returns a view into the preloaded stack.
        Lazy mode decodes from disk with a small LRU over recent indices.
        """
        image_paths = self._image_paths
        image_stack = self._image_cached

        if image_stack is not None:
            normalized_index = normalize_index(index, image_stack.shape[0], "Image")
            return cast(npt.NDArray[np.uint8], image_stack[normalized_index])

        if image_paths is None:
            raise IndexError(f"Image index {index} out of range")

        normalized_index = normalize_index(index, len(image_paths), "Image")
        cached_image = self._lazy_image_cache.get(normalized_index)
        if cached_image is not None:
            return cached_image

        decoded_image = _read_image(image_paths[normalized_index])
        freeze_array(decoded_image)
        return self._lazy_image_cache.put(normalized_index, decoded_image)

    def find_nearest_image_index(self, t: float) -> int:
        """Return the image index nearest to time *t* (in seconds)."""
        if self._image_ts is None or len(self._image_ts) == 0:
            raise RuntimeError("Image timestamps are not available.")
        return find_nearest_index(self._image_ts, t)

    @property
    def has_flow_forward(self) -> bool:
        """Whether forward optical flow is available."""
        return self._flow_fwd_paths is not None and len(self._flow_fwd_paths) > 0

    @property
    def has_flow_backward(self) -> bool:
        """Whether backward optical flow is available."""
        return self._flow_bwd_paths is not None and len(self._flow_bwd_paths) > 0

    @property
    def flow_forward_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Forward flow timestamps ``(N, 2)`` in seconds, or None."""
        return self._flow_fwd_ts

    @property
    def flow_backward_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Backward flow timestamps ``(N, 2)`` in seconds, or None."""
        return self._flow_bwd_ts

    @property
    def num_flow_forward(self) -> int:
        """Number of forward flow frames."""
        if self._flow_fwd_paths is None:
            return 0
        return len(self._flow_fwd_paths)

    @property
    def num_flow_backward(self) -> int:
        """Number of backward flow frames."""
        if self._flow_bwd_paths is None:
            return 0
        return len(self._flow_bwd_paths)

    def load_flow_forward(
        self,
        index: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load a single forward optical flow field by index."""
        return self._load_flow(index, backward=False)

    def load_flow_backward(
        self,
        index: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load a single backward optical flow field by index."""
        return self._load_flow(index, backward=True)

    def _load_flow(
        self,
        index: int,
        backward: bool,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        if backward:
            cached_flow = self._flow_bwd_cached
            cached_valid = self._flow_bwd_valid_cached
            paths = self._flow_bwd_paths
            lazy_decode_cache = self._lazy_flow_bwd_cache
            direction_name = "Backward flow"
        else:
            cached_flow = self._flow_fwd_cached
            cached_valid = self._flow_fwd_valid_cached
            paths = self._flow_fwd_paths
            lazy_decode_cache = self._lazy_flow_fwd_cache
            direction_name = "Forward flow"

        if cached_flow is not None and cached_valid is not None:
            normalized_index = normalize_index(
                index,
                cached_flow.shape[0],
                direction_name,
            )
            return cached_flow[normalized_index], cached_valid[normalized_index]

        if paths is None:
            raise IndexError(f"{direction_name} index {index} out of range")

        normalized_index = normalize_index(index, len(paths), direction_name)
        cached_payload = lazy_decode_cache.get(normalized_index)
        if cached_payload is not None:
            return cached_payload

        decoded_payload = _freeze_dense_field(_read_flow_png(paths[normalized_index]))
        return lazy_decode_cache.put(normalized_index, decoded_payload)

    @property
    def has_disparity(self) -> bool:
        """Whether disparity maps are available."""
        return self._disp_paths is not None and len(self._disp_paths) > 0

    @property
    def disparity_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Disparity timestamps in seconds, or None."""
        return self._disp_ts

    @property
    def num_disparity_frames(self) -> int:
        """Number of disparity frames."""
        if self._disp_paths is None:
            return 0
        return len(self._disp_paths)

    def load_disparity(
        self,
        index: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load a single disparity map by index.

        Returns:
            disparity: ``(H, W)`` float32.
            valid: ``(H, W)`` bool validity mask.
        """
        cached_disparity = self._disp_cached
        cached_validity = self._disp_valid_cached
        disparity_paths = self._disp_paths

        if cached_disparity is not None and cached_validity is not None:
            normalized_index = normalize_index(
                index,
                cached_disparity.shape[0],
                "Disparity",
            )
            return cached_disparity[normalized_index], cached_validity[normalized_index]

        if disparity_paths is None:
            raise IndexError(f"Disparity index {index} out of range")

        normalized_index = normalize_index(index, len(disparity_paths), "Disparity")
        cached_payload = self._lazy_disparity_cache.get(normalized_index)
        if cached_payload is not None:
            return cached_payload

        decoded_payload = _freeze_dense_field(
            _read_disparity_png(disparity_paths[normalized_index])
        )
        return self._lazy_disparity_cache.put(normalized_index, decoded_payload)

    @property
    def has_imu(self) -> bool:
        """Whether IMU samples are available."""
        return self._imu_ts is not None

    @property
    def imu_timestamps(self) -> npt.NDArray[np.float64] | None:
        """IMU sample timestamps in seconds, or None."""
        return self._imu_ts

    @property
    def imu_data(self) -> DSECImuData | None:
        """Cached IMU data, or None when IMU is lazy or unavailable."""
        return self._imu_cached

    @property
    def num_imu_samples(self) -> int:
        """Number of available IMU samples."""
        if self._imu_ts is None:
            return 0
        return len(self._imu_ts)

    def load_imu(self, t_start: float, t_end: float) -> DSECImuData:
        """Load IMU samples whose timestamps fall in ``[t_start, t_end)``."""
        if t_end < t_start:
            raise ValueError(f"Expected t_end >= t_start, got [{t_start}, {t_end}).")
        if self._imu_ts is None:
            raise RuntimeError("IMU data is not available.")

        start, end = np.searchsorted(self._imu_ts, [t_start, t_end], side="left")
        start_index = int(start)
        end_index = int(end)

        if self._imu_cached is not None:
            return _copy_imu_data(self._imu_cached, start_index, end_index)

        imu_ts_ns = self._imu_ts_ns
        if imu_ts_ns is None:
            raise RuntimeError("IMU timestamp index is not available.")
        if start_index == end_index:
            return _empty_imu_data()
        start_ns = int(imu_ts_ns[start_index])
        if end_index < len(imu_ts_ns):
            stop_ns = int(imu_ts_ns[end_index])
        else:
            stop_ns = int(imu_ts_ns[-1]) + 1
        n_messages = end_index - start_index
        return self._read_imu_interval_from_bag(start_ns, stop_ns, n_messages)

    @property
    def has_lidar(self) -> bool:
        """Whether Velodyne LiDAR scans are available."""
        return self._lidar_ts is not None

    @property
    def lidar_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Return LiDAR scan timestamps in seconds, or None."""
        return self._lidar_ts

    @property
    def num_lidar_scans(self) -> int:
        """Number of available LiDAR scans."""
        if self._lidar_ts is None:
            return 0
        return len(self._lidar_ts)

    @property
    def lidar_frame_id(self) -> str | None:
        """Frame id used by the LiDAR PointCloud2 messages, or None."""
        return self._lidar_frame_id

    def load_lidar(self, index: int) -> DSECLidarScan:
        """Load one Velodyne scan by index.

        The returned ``points`` array is a compact structured NumPy array
        using the PointCloud2 field names and dtypes, commonly
        ``x``, ``y``, ``z``, ``intensity``, ``ring``, and ``time`` for DSEC.
        """
        cached_scans = self._lidar_cached
        if cached_scans is not None:
            normalized_index = normalize_index(index, len(cached_scans), "LiDAR")
            return cached_scans[normalized_index]

        if self._lidar_ts is None:
            raise IndexError(f"LiDAR index {index} out of range")

        normalized_index = normalize_index(index, len(self._lidar_ts), "LiDAR")
        cached_scan = self._lazy_lidar_cache.get(normalized_index)
        if cached_scan is not None:
            return cached_scan

        scans = self._read_lidar_scans(normalized_index, normalized_index + 1)
        return scans[0]

    def load_lidar_by_time(self, t_start: float, t_end: float) -> list[DSECLidarScan]:
        """Load LiDAR scans whose timestamps fall in ``[t_start, t_end)``."""
        if t_end < t_start:
            raise ValueError(f"Expected t_end >= t_start, got [{t_start}, {t_end}).")
        if self._lidar_ts is None:
            raise RuntimeError("LiDAR data is not available.")
        start, end = np.searchsorted(self._lidar_ts, [t_start, t_end], side="left")
        start_index = int(start)
        end_index = int(end)
        if start_index == end_index:
            return []

        cached_scans = self._lidar_cached
        if cached_scans is not None:
            selected_scans: list[DSECLidarScan] = []
            for index in range(start_index, end_index):
                selected_scans.append(cached_scans[index])
            return selected_scans

        cached_range: list[DSECLidarScan] = []
        # Use the lazy cache for repeated overlapping ranges
        for index in range(start_index, end_index):
            cached_scan = self._lazy_lidar_cache.get(index)
            if cached_scan is None:
                break
            cached_range.append(cached_scan)
        if len(cached_range) == end_index - start_index:
            return cached_range
        return self._read_lidar_scans(start_index, end_index)

    @property
    def has_calibration(self) -> bool:
        """Whether cam to cam calibration data is loaded."""
        return self._calibration is not None

    @property
    def calibration(self) -> dict[str, Any] | None:
        """Parsed calibration YAML dict, or None.

        A deep copy is returned so callers cannot mutate process global
        calibration cache entries shared by later loader instances.
        """
        if self._calibration is None:
            return None
        return copy.deepcopy(self._calibration)

    @property
    def cam_to_lidar_calibration(self) -> dict[str, Any] | None:
        """Parsed ``data/<drive_prefix>/cam_to_lidar.yaml`` dict, or None."""
        if self._cam_to_lidar_calibration is None:
            return None
        return copy.deepcopy(self._cam_to_lidar_calibration)

    @property
    def cam_to_imu_calibration(self) -> dict[str, Any] | None:
        """Parsed ``imu_calibration/cam0_to_imu0.yaml`` dict, or None."""
        if self._cam_to_imu_calibration is None:
            return None
        return copy.deepcopy(self._cam_to_imu_calibration)

    @property
    def imu_calibration(self) -> dict[str, Any] | None:
        """Parsed ``imu_calibration/imu0_params.yaml`` dict, or None."""
        if self._imu_calibration is None:
            return None
        return copy.deepcopy(self._imu_calibration)

    @property
    def num_samples(self) -> int:
        """Number of iterable samples.

        Returns ``num_flow_forward`` if flow is loaded,
        else ``num_images - 1`` if images are loaded (each sample spans two consecutive images),
        else 0.
        """
        if self.has_flow_forward:
            return self.num_flow_forward
        if self.has_images and self.num_images > 1:
            return self.num_images - 1
        return 0

    def load_frame_sample(self, index: int) -> DSECSample:
        """Load a synchronized sample dict for the given index.

        The sample time interval is taken from the forward flow timestamps
        file (if loaded) or from consecutive image timestamps.
        Per the DSEC documentation, ``forward[k] = (t_k, t_{k+1})`` and
        ``backward[k] = (t_k, t_{k-1})`` are both anchored at ``t_k``.
        Backward flow is associated with the sample by matching its anchor
        (``t_start``) to the sample's ``t_start`` rather than by reusing
        ``index``.
        ``flow_backward`` is ``None`` when no backward entry shares the anchor.
        Backward coverage can be a strict subset of forward coverage.
        """
        sample_index = normalize_index(index, self.num_samples, "Sample")
        sample_start_time, sample_end_time = self._sample_time_interval(sample_index)

        events = self.get_events_by_time(sample_start_time, sample_end_time)

        image_start: npt.NDArray[np.uint8] | None = None
        image_end: npt.NDArray[np.uint8] | None = None
        if self.has_images and self._image_ts is not None:
            sample_start_image_index = find_nearest_index(
                self._image_ts,
                sample_start_time,
            )
            sample_end_image_index = find_nearest_index(self._image_ts, sample_end_time)
            image_start = self.load_image(sample_start_image_index)
            if sample_end_image_index == sample_start_image_index:
                image_end = image_start
            else:
                image_end = self.load_image(sample_end_image_index)

        flow: _DecodedDenseField | None = None
        if self.has_flow_forward and sample_index < self.num_flow_forward:
            flow = self.load_flow_forward(sample_index)

        flow_backward: _DecodedDenseField | None = None
        backward_index = self._find_backward_flow_index(sample_start_time)
        if backward_index is not None:
            flow_backward = self.load_flow_backward(backward_index)

        disparity: _DecodedDenseField | None = None
        if self.has_disparity and self._disp_ts is not None and len(self._disp_ts) > 0:
            sample_disparity_index = find_nearest_index(self._disp_ts, sample_end_time)
            disparity = self.load_disparity(sample_disparity_index)

        imu: DSECImuData | None = None
        if self.has_imu:
            imu = self.load_imu(sample_start_time, sample_end_time)

        lidar: list[DSECLidarScan] | None = None
        if self.has_lidar:
            lidar = self.load_lidar_by_time(sample_start_time, sample_end_time)

        return DSECSample(
            events=events,
            timestamp=(sample_start_time, sample_end_time),
            image_start=image_start,
            image_end=image_end,
            flow=flow,
            flow_backward=flow_backward,
            disparity=disparity,
            imu=imu,
            lidar=lidar,
        )

    def _find_backward_flow_index(self, anchor_time: float) -> int | None:
        """Return the backward flow row whose anchor matches ``anchor_time``.

        Per DSEC, ``backward[k]`` is the displacement from ``t_k`` to
        ``t_{k-1}``, so ``flow_backward_timestamps[k, 0] == t_k``.
        Returns ``None`` if backward flow is not loaded or no row's anchor
        falls within one microsecond of ``anchor_time``.
        """
        if not self.has_flow_backward or self._flow_bwd_ts is None:
            return None
        anchors = self._flow_bwd_ts[:, 0]
        if len(anchors) == 0:
            return None
        candidate = int(np.searchsorted(anchors, anchor_time, side="left"))
        for index in (candidate, candidate - 1):
            if 0 <= index < len(anchors) and abs(anchors[index] - anchor_time) <= 1e-6:
                return index
        return None

    def _sample_time_interval(self, index: int) -> tuple[float, float]:
        """Return ``(t_start, t_end)`` for the given sample index."""
        if self.has_flow_forward and self._flow_fwd_ts is not None:
            t_start = float(self._flow_fwd_ts[index, 0])
            t_end = float(self._flow_fwd_ts[index, 1])
            return t_start, t_end

        if self.has_images and self._image_ts is not None and self.num_images > 1:
            t_start = float(self._image_ts[index])
            t_end = float(self._image_ts[index + 1])
            return t_start, t_end

        raise RuntimeError(
            "Cannot determine sample time interval. "
            "Load images or flow to enable frame sampling."
        )
