"""Low level I/O loader for DAVIS/RPG text recordings.

The loader targets the recording layout used by the RPG Event Camera Dataset
and Simulator and by related DAVIS text exports:

    events.txt       timestamp x y polarity
    images.txt       timestamp relative/path.png
    imu.txt          timestamp ax ay az gx gy gz
    groundtruth.txt  timestamp px py pz qx qy qz qw
    calib.txt        fx fy cx cy k1 k2 p1 p2 k3
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import warnings
from dataclasses import dataclass
from typing import BinaryIO
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.lib import format as np_format
from numpy.lib.format import open_memmap

from evlib.types import RawEvents

from ._base import DataLoaderBase
from ._storage_common import LoadingType
from ._storage_common import LoadMode
from ._storage_common import ResidentLoadMode
from .utils import LazyDecodeCache
from .utils import cv2
from .utils import decode_to_array_stack
from .utils import find_nearest_index
from .utils import freeze_array
from .utils import normalize_index
from .utils import validate_index_interval


_TEXT_EVENT_CACHE_SCHEMA_VERSION = 2
_TEXT_EVENT_PARSE_BLOCK_BYTES = 4 * 1024 * 1024
_TEXT_EVENT_TIMESTAMP_INDEX_STRIDE = 4096
_IMAGE_CACHE_ITEMS = 8
_DEPTH_CACHE_ITEMS = 2
_IMAGE_DECODE_PARALLELISM = 8
_DEPTH_DECODE_PARALLELISM = 4
_MMapMode = Optional[Literal["r"]]


class DavisImuData(TypedDict):
    """IMU measurements returned by :class:`DavisRecordingLoader`."""

    timestamp: npt.NDArray[np.float64]
    linear_acceleration: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]


class DavisPoseData(TypedDict):
    """Ground-truth pose trajectory stored in DAVIS text recordings."""

    timestamp: npt.NDArray[np.float64]
    position: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]


class DavisFrameSample(TypedDict):
    """Synchronized DAVIS sample returned by ``load_frame_sample``."""

    events: RawEvents
    timestamp: float
    image: npt.NDArray[np.uint8] | None
    imu: DavisImuData | None
    pose: npt.NDArray[np.float64] | None
    depth: npt.NDArray[np.float32] | None


class _TextEventCacheMetadata(TypedDict):
    schema_version: int
    source_path: str
    source_size: int
    source_mtime_ns: int
    num_events: int


class _UseSlowTextEventParser(RuntimeError):
    """Signal that a text event file needs the line by line parser."""


class _NpyColumnWriter:
    """Streaming writer for one 1D NPY sidecar column.

    Appends chunks sequentially for the fast block parser, slow parser
    uses ``open_memmap`` instead, which is suitable for its random per element writes.
    """

    def __init__(self, path: str, dtype: npt.DTypeLike, num_rows: int) -> None:
        """Create a writable NPY file with a final, known shape."""
        self._path = path
        self._dtype = np.dtype(dtype)
        self._num_rows = num_rows
        self._rows_written = 0
        self._file: BinaryIO = open(path, "wb")
        header = {
            "descr": np_format.dtype_to_descr(self._dtype),
            "fortran_order": False,
            "shape": (num_rows,),
        }
        np_format.write_array_header_1_0(self._file, header)

    def write(self, values: npt.ArrayLike) -> None:
        """Append one chunk of values to the NPY data section."""
        array = np.ascontiguousarray(values, dtype=self._dtype)
        new_total = self._rows_written + int(array.shape[0])
        if new_total > self._num_rows:
            raise RuntimeError(
                f"Too many rows written to {self._path}: {new_total} > {self._num_rows}."
            )
        array.tofile(self._file)
        self._rows_written = new_total

    def close(self, *, validate: bool = True) -> None:
        """Close the writer after validating the expected row count."""
        try:
            if validate and self._rows_written != self._num_rows:
                raise RuntimeError(
                    f"Expected to write {self._num_rows} rows to {self._path}, "
                    f"wrote {self._rows_written}."
                )
        finally:
            self._file.close()


@dataclass(frozen=True)
class DavisCameraCalibration:
    """OpenCV pinhole calibration parsed from ``calib.txt``.

    Attributes:
        camera_matrix: ``(3, 3)`` matrix with ``fx``, ``fy``, ``cx``, and ``cy``.
        distortion_coefficients: ``(5,)`` OpenCV coefficients
            ``[k1, k2, p1, p2, k3]``.
        parameters: Original ``(9,)`` row
            ``[fx, fy, cx, cy, k1, k2, p1, p2, k3]``.  ECD synthetic
            sequences may omit ``k3``; it is normalized to ``0.0``.
    """

    camera_matrix: npt.NDArray[np.float64]
    distortion_coefficients: npt.NDArray[np.float64]
    parameters: npt.NDArray[np.float64]


def resolve_davis_cache_dir(cache_dir: str | None) -> str:
    """Return the root directory for DAVIS recording sidecar caches."""
    if cache_dir is not None:
        expanded_path = os.path.expanduser(cache_dir)
        absolute_path = os.path.abspath(expanded_path)
        return absolute_path

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home is not None:
        cache_base = os.path.expanduser(xdg_cache_home)
    else:
        home_dir = os.path.expanduser("~")
        cache_base = os.path.join(home_dir, ".cache")

    cache_dir_path = os.path.join(cache_base, "evlib", "davis_recordings")
    return os.path.abspath(cache_dir_path)


def _make_cache_signature(source_path: str) -> str:
    stat_result = os.stat(source_path)
    parts = [
        os.path.abspath(source_path),
        str(int(stat_result.st_size)),
        str(int(stat_result.st_mtime_ns)),
        str(_TEXT_EVENT_CACHE_SCHEMA_VERSION),
    ]
    joined = "|".join(parts)
    signature = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return signature


def _make_text_event_cache_dir(cache_root: str, source_path: str) -> str:
    signature = _make_cache_signature(source_path)
    source_name = os.path.splitext(os.path.basename(source_path))[0]
    directory_name = f"{source_name}_{signature[:16]}"
    return os.path.join(cache_root, "text_event_sidecars", directory_name)


def _make_text_event_cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "x": os.path.join(cache_dir, "events_x.npy"),
        "y": os.path.join(cache_dir, "events_y.npy"),
        "timestamp": os.path.join(cache_dir, "events_t.npy"),
        "timestamp_index": os.path.join(cache_dir, "events_t_index.npy"),
        "polarity": os.path.join(cache_dir, "events_p.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def _write_json(path: str, data: _TextEventCacheMetadata) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def _load_text_event_cache_metadata(path: str) -> _TextEventCacheMetadata | None:
    if not os.path.isfile(path):
        return None

    with open(path, encoding="utf-8") as file_handle:
        metadata = cast(_TextEventCacheMetadata, json.load(file_handle))
    return metadata


def _text_event_cache_is_complete(
    cache_dir: str,
    source_path: str,
) -> _TextEventCacheMetadata | None:
    cache_paths = _make_text_event_cache_paths(cache_dir)
    cache_metadata = _load_text_event_cache_metadata(cache_paths["metadata"])
    if cache_metadata is None:
        return None

    required_keys = ("x", "y", "timestamp", "timestamp_index", "polarity")
    for required_key in required_keys:
        required_path = cache_paths[required_key]
        if not os.path.isfile(required_path):
            return None

    stat_result = os.stat(source_path)
    source_path_abs = os.path.abspath(source_path)

    schema_matches = cache_metadata["schema_version"] == _TEXT_EVENT_CACHE_SCHEMA_VERSION
    source_path_matches = cache_metadata["source_path"] == source_path_abs
    source_size_matches = cache_metadata["source_size"] == int(stat_result.st_size)
    source_mtime_matches = cache_metadata["source_mtime_ns"] == int(stat_result.st_mtime_ns)

    if not schema_matches:
        return None
    if not source_path_matches:
        return None
    if not source_size_matches:
        return None
    if not source_mtime_matches:
        return None

    return cache_metadata


def _is_data_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _count_text_event_rows(path: str) -> int:
    """Count event rows for the ECD/RPG event layout."""
    count = 0
    last_byte = b""
    has_data = False
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(_TEXT_EVENT_PARSE_BLOCK_BYTES)
            if not chunk:
                break
            has_data = True
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if has_data and last_byte != b"\n":
        count += 1
    return count


def _count_text_event_rows_slow(path: str) -> int:
    count = 0
    with open(path, encoding="utf-8") as file_handle:
        for line in file_handle:
            if _is_data_line(line):
                count += 1
    return count


def _parse_event_line(
    line: str,
    path: str,
    line_number: int,
) -> tuple[float, int, int, bool]:
    fields = line.split()
    field_count = len(fields)
    if field_count != 4:
        raise ValueError(
            f"Expected 4 columns in DAVIS events file {path}:{line_number}, " f"got {field_count}."
        )

    try:
        timestamp = float(fields[0])
        x_value = int(fields[1])
        y_value = int(fields[2])
        polarity_value = int(fields[3])
    except ValueError as exc:
        raise ValueError(f"Failed to parse DAVIS event row at {path}:{line_number}.") from exc

    x_is_negative = x_value < 0
    y_is_negative = y_value < 0
    if x_is_negative or y_is_negative:
        raise ValueError(f"DAVIS event coordinates must be non-negative at {path}:{line_number}.")

    int16_max = np.iinfo(np.int16).max
    x_exceeds_int16 = x_value > int16_max
    y_exceeds_int16 = y_value > int16_max
    if x_exceeds_int16 or y_exceeds_int16:
        raise ValueError(f"DAVIS event coordinates exceed int16 range at {path}:{line_number}.")

    if polarity_value not in (-1, 0, 1):
        raise ValueError(
            f"DAVIS event polarity must be -1, 0, or 1 at {path}:{line_number}, "
            f"got {polarity_value}."
        )

    polarity = polarity_value > 0
    return timestamp, x_value, y_value, polarity


def _parse_event_block(
    data: bytes,
    *,
    path: str,
) -> npt.NDArray[np.float64]:
    if b"#" in data:
        raise _UseSlowTextEventParser("Commented DAVIS event files need the slow parser.")

    values = np.fromstring(data, dtype=np.float64, sep=" ")
    if values.size == 0:
        if data.strip():
            raise _UseSlowTextEventParser("Unparseable DAVIS event block.")
        return np.empty((0, 4), dtype=np.float64)

    if values.size % 4 != 0:
        raise _UseSlowTextEventParser(
            f"DAVIS event file {path} is not a dense four-column numeric table."
        )
    return values.reshape((-1, 4))


def _iter_event_blocks(path: str) -> Iterator[npt.NDArray[np.float64]]:
    carry = b""
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(_TEXT_EVENT_PARSE_BLOCK_BYTES)
            if not chunk:
                if carry:
                    yield _parse_event_block(carry, path=path)
                return

            data = carry + chunk
            # A fixed size read can split a row mid line so parse only through the last newline and carry the partial
            # tail into the next chunk.
            newline_index = data.rfind(b"\n")
            if newline_index < 0:
                carry = data
                continue

            block_data = data[: newline_index + 1]
            carry = data[newline_index + 1 :]
            if block_data:
                yield _parse_event_block(block_data, path=path)


def _validate_event_block(
    block: npt.NDArray[np.float64],
    *,
    path: str,
    row_start: int,
    previous_timestamp: float | None,
) -> float | None:
    if block.ndim != 2 or block.shape[1] != 4:
        raise ValueError(f"DAVIS event block must have shape (N, 4), got {block.shape}.")
    if block.shape[0] == 0:
        return previous_timestamp

    timestamp_column = block[:, 0]
    x_column = block[:, 1]
    y_column = block[:, 2]
    polarity_column = block[:, 3]
    row_end = row_start + block.shape[0] - 1

    first_timestamp = float(timestamp_column[0])
    if previous_timestamp is not None and first_timestamp < previous_timestamp:
        raise ValueError(
            f"DAVIS event timestamps must be nondecreasing in {path}; "
            f"row {row_start} has {first_timestamp} after {previous_timestamp}."
        )

    has_decreasing_timestamps = False
    if timestamp_column.size > 1:
        timestamp_deltas = np.diff(timestamp_column)
        has_decreasing_timestamps = bool(np.any(timestamp_deltas < 0.0))
    if has_decreasing_timestamps:
        raise ValueError(
            f"DAVIS event timestamps must be nondecreasing in {path}; "
            f"rows {row_start}-{row_end} contain a decreasing timestamp."
        )

    int16_max = np.iinfo(np.int16).max
    x_nonnegative = bool(np.all(x_column >= 0.0))
    y_nonnegative = bool(np.all(y_column >= 0.0))
    x_fits_int16 = bool(np.all(x_column <= int16_max))
    y_fits_int16 = bool(np.all(y_column <= int16_max))
    coordinate_range_valid = all(
        [
            x_nonnegative,
            y_nonnegative,
            x_fits_int16,
            y_fits_int16,
        ]
    )
    if not coordinate_range_valid:
        raise ValueError(
            f"DAVIS event coordinates must be non-negative int16 values in {path}; "
            f"rows {row_start}-{row_end} contain an out-of-range coordinate."
        )

    x_values_are_int = bool(np.all(x_column == np.floor(x_column)))
    y_values_are_int = bool(np.all(y_column == np.floor(y_column)))
    if not x_values_are_int or not y_values_are_int:
        raise ValueError(
            f"DAVIS event coordinates must be integers in {path}; "
            f"rows {row_start}-{row_end} contain a noninteger coordinate."
        )

    polarity_is_valid = np.isin(polarity_column, [-1.0, 0.0, 1.0])
    if not bool(np.all(polarity_is_valid)):
        raise ValueError(
            f"DAVIS event polarity must be -1, 0, or 1 in {path}; "
            f"rows {row_start}-{row_end} contain an invalid polarity."
        )
    return float(timestamp_column[-1])


def _write_fast_event_cache_columns(
    source_path: str,
    *,
    num_events: int,
    x_writer: _NpyColumnWriter,
    y_writer: _NpyColumnWriter,
    timestamp_writer: _NpyColumnWriter,
    timestamp_index_path: str,
    polarity_writer: _NpyColumnWriter,
) -> None:
    previous_timestamp: float | None = None
    row_index = 0
    timestamp_index_chunks: list[npt.NDArray[np.float64]] = []

    for block in _iter_event_blocks(source_path):
        row_start = row_index + 1
        block_event_count = int(block.shape[0])
        row_end_index = row_index + block_event_count
        if row_end_index > num_events:
            raise _UseSlowTextEventParser("DAVIS event row count mismatch.")

        previous_timestamp = _validate_event_block(
            block,
            path=source_path,
            row_start=row_start,
            previous_timestamp=previous_timestamp,
        )

        first_indexed_event = (
            (row_index + _TEXT_EVENT_TIMESTAMP_INDEX_STRIDE - 1)
            // _TEXT_EVENT_TIMESTAMP_INDEX_STRIDE
        ) * _TEXT_EVENT_TIMESTAMP_INDEX_STRIDE
        if first_indexed_event < row_end_index:
            local_start = first_indexed_event - row_index
            indexed_timestamps = block[local_start::_TEXT_EVENT_TIMESTAMP_INDEX_STRIDE, 0]
            timestamp_index_chunks.append(indexed_timestamps.copy())

        timestamp_column = block[:, 0]
        x_column = block[:, 1]
        y_column = block[:, 2]
        polarity_column = block[:, 3] > 0.0

        x_writer.write(x_column)
        y_writer.write(y_column)
        timestamp_writer.write(timestamp_column)
        polarity_writer.write(polarity_column)
        row_index = row_end_index

    if row_index != num_events:
        raise _UseSlowTextEventParser(
            f"Parsed {row_index} event rows from {source_path}, expected {num_events}."
        )

    if not timestamp_index_chunks:
        timestamp_index = np.empty((0,), dtype=np.float64)
    else:
        timestamp_index = np.concatenate(timestamp_index_chunks)
        timestamp_index = timestamp_index.astype(np.float64, copy=False)
    np.save(timestamp_index_path, timestamp_index, allow_pickle=False)


def _make_text_event_cache_metadata(
    source_path: str,
    num_events: int,
) -> _TextEventCacheMetadata:
    stat_result = os.stat(source_path)
    source_size = int(stat_result.st_size)
    source_mtime_ns = int(stat_result.st_mtime_ns)
    source_path_abs = os.path.abspath(source_path)
    return {
        "schema_version": _TEXT_EVENT_CACHE_SCHEMA_VERSION,
        "source_path": source_path_abs,
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "num_events": num_events,
    }


def _close_memmaps(memmaps: tuple[np.memmap, ...]) -> None:
    """Flush and release a group of writable NumPy memmaps."""
    for memmap_array in memmaps:
        memmap_array.flush()
        # np.memmap has no public close, closing the backing mmap is the way to unmap deterministically
        memmap_array._mmap.close()  # type: ignore[attr-defined]


def _replace_directory(source_dir: str, destination_dir: str) -> None:
    if os.path.isdir(destination_dir):
        shutil.rmtree(destination_dir)
    os.replace(source_dir, destination_dir)


def _close_npy_writers(
    writers: tuple[_NpyColumnWriter, ...],
    *,
    validate: bool,
) -> None:
    for writer in writers:
        writer.close(validate=validate)


def _create_temp_cache_dir(cache_dir: str) -> str:
    """Create a unique sibling dir for an atomic cache build.

    Build here, then swap into ``cache_dir`` so a crash never leaves a partial
    cache.
    """
    parent_dir = os.path.dirname(cache_dir)
    os.makedirs(parent_dir, exist_ok=True)
    temp_dir = os.path.join(parent_dir, f".tmp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir)
    return temp_dir


def _build_text_event_cache_fast(
    source_path: str,
    cache_dir: str,
) -> _TextEventCacheMetadata:
    temp_dir = _create_temp_cache_dir(cache_dir)
    cache_paths = _make_text_event_cache_paths(temp_dir)
    # Two pass:
    #   Count rows to size the NPY headers, then stream the file into the columns
    #   Second read is cheap from page cache and keeps peak memory bounded
    #   buffers all events , first would cost gigabytes on multi gb recordings.
    expected_event_count = _count_text_event_rows(source_path)

    try:
        x_writer = _NpyColumnWriter(cache_paths["x"], np.int16, expected_event_count)
        y_writer = _NpyColumnWriter(cache_paths["y"], np.int16, expected_event_count)
        timestamp_writer = _NpyColumnWriter(
            cache_paths["timestamp"],
            np.float64,
            expected_event_count,
        )
        polarity_writer = _NpyColumnWriter(
            cache_paths["polarity"],
            np.bool_,
            expected_event_count,
        )
        writers = (x_writer, y_writer, timestamp_writer, polarity_writer)
        try:
            _write_fast_event_cache_columns(
                source_path,
                num_events=expected_event_count,
                x_writer=x_writer,
                y_writer=y_writer,
                timestamp_writer=timestamp_writer,
                timestamp_index_path=cache_paths["timestamp_index"],
                polarity_writer=polarity_writer,
            )
            _close_npy_writers(writers, validate=True)
        except Exception:
            _close_npy_writers(writers, validate=False)
            raise

        metadata = _make_text_event_cache_metadata(source_path, expected_event_count)
        _write_json(cache_paths["metadata"], metadata)

        _replace_directory(temp_dir, cache_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _build_text_event_cache_slow(
    source_path: str,
    cache_dir: str,
) -> _TextEventCacheMetadata:
    temp_dir = _create_temp_cache_dir(cache_dir)
    cache_paths = _make_text_event_cache_paths(temp_dir)
    expected_event_count = _count_text_event_rows_slow(source_path)

    try:
        x_values = open_memmap(
            cache_paths["x"],
            mode="w+",
            dtype=np.int16,
            shape=(expected_event_count,),
        )
        y_values = open_memmap(
            cache_paths["y"],
            mode="w+",
            dtype=np.int16,
            shape=(expected_event_count,),
        )
        timestamp_values = open_memmap(
            cache_paths["timestamp"],
            mode="w+",
            dtype=np.float64,
            shape=(expected_event_count,),
        )
        polarity_values = open_memmap(
            cache_paths["polarity"],
            mode="w+",
            dtype=np.bool_,
            shape=(expected_event_count,),
        )

        row_index = 0
        previous_timestamp: float | None = None
        timestamp_index_values: list[float] = []
        with open(source_path, encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                if not _is_data_line(line):
                    continue

                timestamp, x_value, y_value, polarity = _parse_event_line(
                    line,
                    source_path,
                    line_number,
                )
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(
                        f"DAVIS event timestamps must be nondecreasing in {source_path}; "
                        f"row {line_number} has {timestamp} after {previous_timestamp}."
                    )

                x_values[row_index] = x_value
                y_values[row_index] = y_value
                timestamp_values[row_index] = timestamp
                polarity_values[row_index] = polarity
                if row_index % _TEXT_EVENT_TIMESTAMP_INDEX_STRIDE == 0:
                    timestamp_index_values.append(timestamp)
                previous_timestamp = timestamp
                row_index += 1

        if row_index != expected_event_count:
            raise RuntimeError(
                f"Parsed {row_index} event rows from {source_path}, "
                f"expected {expected_event_count}."
            )

        timestamp_index_array = np.asarray(timestamp_index_values, dtype=np.float64)
        np.save(
            cache_paths["timestamp_index"],
            timestamp_index_array,
            allow_pickle=False,
        )

        # Release the memmaps before the swap
        _close_memmaps((x_values, y_values, timestamp_values, polarity_values))

        metadata = _make_text_event_cache_metadata(source_path, expected_event_count)
        _write_json(cache_paths["metadata"], metadata)

        _replace_directory(temp_dir, cache_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _build_text_event_cache(source_path: str, cache_dir: str) -> _TextEventCacheMetadata:
    try:
        return _build_text_event_cache_fast(source_path, cache_dir)
    except _UseSlowTextEventParser:
        return _build_text_event_cache_slow(source_path, cache_dir)


def _prepare_text_event_cache(
    source_path: str,
    cache_root: str,
) -> tuple[dict[str, str], _TextEventCacheMetadata]:
    source_path_abs = os.path.abspath(source_path)
    cache_dir = _make_text_event_cache_dir(cache_root, source_path_abs)
    cache_paths = _make_text_event_cache_paths(cache_dir)
    cache_metadata = _text_event_cache_is_complete(cache_dir, source_path_abs)
    if cache_metadata is None:
        cache_metadata = _build_text_event_cache(source_path_abs, cache_dir)
    return cache_paths, cache_metadata


class _TextEventBackend:
    """Typed column event backend for DAVIS text recordings."""

    def __init__(self, source_path: str, cache_root: str, mode: LoadingType) -> None:
        cache_paths, cache_metadata = _prepare_text_event_cache(source_path, cache_root)
        self._paths = cache_paths
        self._mode = mode
        self._num_events = int(cache_metadata["num_events"])
        self._x: npt.NDArray[np.int16] | None = None
        self._y: npt.NDArray[np.int16] | None = None
        self._timestamp: npt.NDArray[np.float64] | None = None
        self._polarity: npt.NDArray[np.bool_] | None = None
        self._timestamp_index = np.load(cache_paths["timestamp_index"])
        freeze_array(self._timestamp_index)
        self._pid: int | None = None

        if mode is LoadingType.CACHED:
            self._load_columns(mmap_mode=None)

    def __getstate__(self) -> dict[str, object]:
        # Never pickle the event columns,
        # they are reloaded from the sidecars on first use in the receiving process
        # Shipping them would serialize gbs to each worker for nothing, since handles are process local
        state = self.__dict__.copy()
        state["_x"] = None
        state["_y"] = None
        state["_timestamp"] = None
        state["_polarity"] = None
        state["_pid"] = None
        return state

    @property
    def num_events(self) -> int:
        return self._num_events

    def _mmap_mode_for_current_loading_type(self) -> _MMapMode:
        if self._mode is LoadingType.LAZY:
            return "r"
        return None

    def _event_columns_are_open(self) -> bool:
        x_is_open = self._x is not None
        y_is_open = self._y is not None
        timestamp_is_open = self._timestamp is not None
        polarity_is_open = self._polarity is not None
        return all(
            [
                x_is_open,
                y_is_open,
                timestamp_is_open,
                polarity_is_open,
            ]
        )

    def _load_columns(self, mmap_mode: _MMapMode) -> None:
        self._x = np.load(self._paths["x"], mmap_mode=mmap_mode)
        self._y = np.load(self._paths["y"], mmap_mode=mmap_mode)
        self._timestamp = np.load(self._paths["timestamp"], mmap_mode=mmap_mode)
        self._polarity = np.load(self._paths["polarity"], mmap_mode=mmap_mode)
        freeze_array(self._x)
        freeze_array(self._y)
        freeze_array(self._timestamp)
        freeze_array(self._polarity)
        self._pid = os.getpid()

    def _drop_columns_for_new_process(self) -> None:
        if self._pid == os.getpid():
            return

        self._x = None
        self._y = None
        self._timestamp = None
        self._polarity = None
        self._pid = None

    def _ensure_timestamp_open(self) -> None:
        self._drop_columns_for_new_process()
        if self._timestamp is not None:
            return

        mmap_mode = self._mmap_mode_for_current_loading_type()
        self._timestamp = np.load(self._paths["timestamp"], mmap_mode=mmap_mode)
        freeze_array(self._timestamp)
        self._pid = os.getpid()

    def _ensure_event_columns_open(self) -> None:
        self._drop_columns_for_new_process()
        if self._event_columns_are_open():
            return

        mmap_mode = self._mmap_mode_for_current_loading_type()
        self._load_columns(mmap_mode=mmap_mode)

    def _timestamp_column(self) -> npt.NDArray[np.float64]:
        self._ensure_timestamp_open()
        if self._timestamp is None:
            raise RuntimeError("DAVIS event timestamp sidecar column is not open.")
        return self._timestamp

    def _event_columns(
        self,
    ) -> tuple[
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
        npt.NDArray[np.float64],
        npt.NDArray[np.bool_],
    ]:
        self._ensure_event_columns_open()
        columns = (self._x, self._y, self._timestamp, self._polarity)
        if any(column is None for column in columns):
            raise RuntimeError("DAVIS event sidecar columns are not open.")

        return (
            cast(npt.NDArray[np.int16], self._x),
            cast(npt.NDArray[np.int16], self._y),
            cast(npt.NDArray[np.float64], self._timestamp),
            cast(npt.NDArray[np.bool_], self._polarity),
        )

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        start_index, end_index = validate_index_interval(
            start_index,
            end_index,
            self._num_events,
            "event",
        )
        x_values, y_values, timestamp_values, polarity_values = self._event_columns()

        event_slice = slice(start_index, end_index)
        return RawEvents(
            x=x_values[event_slice].copy(),
            y=y_values[event_slice].copy(),
            timestamp=timestamp_values[event_slice].copy(),
            polarity=polarity_values[event_slice].copy(),
        )

    def time_to_index(self, t: float) -> int:
        timestamp_values = self._timestamp_column()
        if self._mode is LoadingType.LAZY:
            return self._time_to_index_lazy(timestamp_values, t)
        # Index of the last event before t (-1 when none precede it)
        first_at_or_after = int(timestamp_values.searchsorted(t, side="left"))
        return first_at_or_after - 1

    def _time_to_index_lazy(self, timestamp_values: npt.NDArray[np.float64], t: float) -> int:
        if self._num_events == 0:
            return -1

        # Sampled timestamps sit at global indices 0, STRIDE, 2*STRIDE, ...
        # samples_below counts these strictly
        # below t, so the first event >= t falls in ((samples_below - 1) * STRIDE, samples_below * STRIDE]
        # Searching only that window matches a full searchsorted even when equal timestamps straddle a stride boundary
        samples_below = int(self._timestamp_index.searchsorted(t, side="left"))
        if samples_below == 0:
            return -1

        stride = _TEXT_EVENT_TIMESTAMP_INDEX_STRIDE
        window_start = (samples_below - 1) * stride
        window_end = min(samples_below * stride + 1, self._num_events)
        window = timestamp_values[window_start:window_end]
        first_at_or_after = int(window.searchsorted(t, side="left"))
        return window_start + first_at_or_after - 1

    def index_to_time(self, index: int) -> float:
        normalized_index = normalize_index(index, self._num_events, "event")
        timestamp_values = self._timestamp_column()
        return float(timestamp_values[normalized_index])

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        timestamp_values = self._timestamp_column()
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        if self._mode is LoadingType.LAZY:
            flat_timestamps = timestamp_array.reshape(-1)
            indices = np.fromiter(
                (
                    self._time_to_index_lazy(timestamp_values, float(timestamp))
                    for timestamp in flat_timestamps
                ),
                dtype=np.int64,
                count=flat_timestamps.size,
            )
            return indices.reshape(timestamp_array.shape)

        first_at_or_after = timestamp_values.searchsorted(timestamp_array, side="left")
        return np.asarray(first_at_or_after - 1, dtype=np.int64)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        timestamp_values = self._timestamp_column()
        index_array = np.asarray(indices, dtype=np.int64)
        timestamps = timestamp_values[index_array]
        return np.asarray(timestamps, dtype=np.float64)

    def close(self) -> None:
        self._x = None
        self._y = None
        self._timestamp = None
        self._polarity = None
        self._pid = None


def _resolve_recording_dir(root: str, sequence: str | None) -> str:
    root_path = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"DAVIS recording root does not exist: {root}")

    if sequence is not None:
        sequence_path = os.path.abspath(os.path.join(root_path, sequence))
        if not os.path.isdir(sequence_path):
            raise FileNotFoundError(
                f"DAVIS recording sequence directory does not exist: {sequence_path}"
            )
        return sequence_path

    direct_event_file = os.path.join(root_path, "events.txt")
    if os.path.isfile(direct_event_file):
        return root_path

    candidate_dirs: list[str] = []
    for name in sorted(os.listdir(root_path)):
        child_path = os.path.join(root_path, name)
        if not os.path.isdir(child_path):
            continue
        child_event_file = os.path.join(child_path, "events.txt")
        if os.path.isfile(child_event_file):
            candidate_dirs.append(child_path)

    if len(candidate_dirs) == 1:
        return candidate_dirs[0]

    raise FileNotFoundError(
        f"Could not find events.txt in DAVIS recording root {root_path}. "
        "Pass the extracted sequence directory, or a parent containing exactly one sequence."
    )


def _require_file(path: str, description: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DAVIS {description} file not found: {path}")
    return path


def _optional_file(path: str) -> str | None:
    if os.path.isfile(path):
        return path
    return None


def _load_numeric_table(
    path: str,
    *,
    expected_columns: int,
    description: str,
) -> npt.NDArray[np.float64]:
    try:
        with warnings.catch_warnings():
            # The empty file case is handled below
            # silence NumPy's "input contained no data" warning for it
            warnings.simplefilter("ignore", category=UserWarning)
            data = np.loadtxt(path, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Failed to parse DAVIS {description} file: {path}") from exc

    if data.size == 0:
        return np.empty((0, expected_columns), dtype=np.float64)

    if data.ndim == 1:
        data = data[None, :]

    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(
            f"DAVIS {description} file {path} must have {expected_columns} columns, "
            f"got shape {data.shape}."
        )
    return np.asarray(data, dtype=np.float64)


def _validate_strictly_increasing(
    timestamps: npt.NDArray[np.float64],
    *,
    path: str,
    description: str,
) -> None:
    if timestamps.size > 1 and bool(np.any(np.diff(timestamps) <= 0.0)):
        raise ValueError(f"DAVIS {description} timestamps must be strictly increasing: {path}")


def _load_image_index(
    path: str,
    *,
    recording_dir: str,
    description: str,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    timestamps: list[float] = []
    item_paths: list[str] = []
    with open(path, encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not _is_data_line(line):
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"Expected 2 columns in DAVIS {description} file {path}:{line_number}, "
                    f"got {len(fields)}."
                )
            try:
                timestamp = float(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse DAVIS {description} timestamp at {path}:{line_number}."
                ) from exc

            relative_item_path = fields[1]
            item_path = os.path.join(recording_dir, relative_item_path)
            if not os.path.isfile(item_path):
                raise FileNotFoundError(
                    f"DAVIS {description} references missing file at "
                    f"{path}:{line_number}: {item_path}"
                )
            timestamps.append(timestamp)
            item_paths.append(item_path)

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    _validate_strictly_increasing(timestamp_array, path=path, description=description)
    freeze_array(timestamp_array)
    return timestamp_array, item_paths


def _load_calibration(path: str | None) -> DavisCameraCalibration | None:
    if path is None:
        return None

    loaded_parameters = np.loadtxt(path, dtype=np.float64)
    calibration_parameters = np.asarray(loaded_parameters, dtype=np.float64)
    calibration_parameters = calibration_parameters.reshape(-1)
    if calibration_parameters.shape == (8,):
        missing_k3 = np.array([0.0], dtype=np.float64)
        calibration_parameters = np.concatenate([calibration_parameters, missing_k3])
    if calibration_parameters.shape != (9,):
        raise ValueError(
            f"DAVIS calibration file {path} must contain 8 or 9 values "
            "(fx fy cx cy k1 k2 p1 p2 [k3])."
        )

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = calibration_parameters
    camera_matrix = np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion_coefficients = np.asarray([k1, k2, p1, p2, k3], dtype=np.float64)
    freeze_array(calibration_parameters)
    freeze_array(camera_matrix)
    freeze_array(distortion_coefficients)
    return DavisCameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        parameters=calibration_parameters,
    )


def _load_imu(path: str | None) -> DavisImuData | None:
    if path is None:
        return None

    imu_table = _load_numeric_table(path, expected_columns=7, description="IMU")
    timestamps = np.asarray(imu_table[:, 0], dtype=np.float64)
    acceleration = np.asarray(imu_table[:, 1:4], dtype=np.float64)
    angular_velocity = np.asarray(imu_table[:, 4:7], dtype=np.float64)
    _validate_strictly_increasing(timestamps, path=path, description="IMU")
    freeze_array(timestamps)
    freeze_array(acceleration)
    freeze_array(angular_velocity)
    return DavisImuData(
        timestamp=timestamps,
        linear_acceleration=acceleration,
        angular_velocity=angular_velocity,
    )


def _load_pose(path: str | None) -> DavisPoseData | None:
    if path is None:
        return None

    pose_table = _load_numeric_table(path, expected_columns=8, description="groundtruth")
    timestamps = np.asarray(pose_table[:, 0], dtype=np.float64)
    position = np.asarray(pose_table[:, 1:4], dtype=np.float64)
    quaternion = np.asarray(pose_table[:, 4:8], dtype=np.float64)
    _validate_strictly_increasing(timestamps, path=path, description="groundtruth")
    freeze_array(timestamps)
    freeze_array(position)
    freeze_array(quaternion)
    return DavisPoseData(
        timestamp=timestamps,
        position=position,
        quaternion=quaternion,
    )


def _decode_grayscale_image(path: str) -> npt.NDArray[np.uint8]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for loading DAVIS images.")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode DAVIS image: {path}")

    image_array = np.asarray(image, dtype=np.uint8)
    if image_array.ndim != 2:
        raise ValueError(f"DAVIS image must decode to a 2D grayscale array: {path}")
    freeze_array(image_array)
    return image_array


def _decode_depth(path: str) -> npt.NDArray[np.float32]:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for loading DAVIS depth maps.")

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to decode DAVIS depth map: {path}")

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError(f"DAVIS depth map must decode to a 2D array: {path}")
    freeze_array(depth_array)
    return depth_array


def _validate_sensor_resolution(
    sensor_resolution: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Validate an explicit ``(height, width)`` sensor resolution."""
    if sensor_resolution is None:
        return None

    if len(sensor_resolution) != 2:
        raise ValueError(f"sensor_resolution must be (height, width), got {sensor_resolution!r}.")
    height, width = int(sensor_resolution[0]), int(sensor_resolution[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"sensor_resolution must be positive, got {(height, width)!r}.")
    return (height, width)


class DavisRecordingLoader(DataLoaderBase):
    """Low level I/O for one DAVIS/RPG text recording.

    The class is intended to be reused by dataset wrappers such as a future ECD dataset class.
    It handles file discovery, validation, typed event sidecar caching, and access to optional DAVIS modalities.

    Args:
        root: Extracted recording directory, dataset root, or a parent
            containing exactly one extracted recording directory.
        sequence: Optional sequence directory under ``root``.  This is a
            convenience for dataset roots that contain many recordings while
            keeping this class a recording format loader.
        load_imu: If True, parse ``imu.txt`` when present.
        load_gt_pose: If True, parse ``groundtruth.txt`` when present.
        event_load_mode: ``"lazy"`` uses read-only NumPy memmaps from a sidecar;
            ``"cached"`` loads the sidecar columns into RAM.
        image_load_mode: ``"lazy"`` decodes images on demand; ``"cached"``
            decodes all referenced images during initialization.
        depth_load_mode: Load mode for ``depthmaps.txt``.  ``False`` ignores
            depth maps, ``True`` or ``"lazy"`` decodes on demand, and
            ``"cached"`` decodes all depth maps during initialization.
        precompute_frame_event_indices: Whether to precompute frame-to-event
            index alignment during initialization.  By default this is enabled
            for cached event mode and disabled for lazy event mode to avoid
            faulting large timestamp sidecars into memory.
        cache_dir: Optional root directory for event sidecar caches.
        sensor_resolution: Optional explicit ``(height, width)`` of the event
            sensor.  When omitted it is inferred from the frame shape.  Pass it
            for recordings without ``images.txt`` so that undistortion and
            sensor-bounds queries still work.  ECD DAVIS240C recordings are
            ``(180, 240)``.
    """

    def __init__(
        self,
        root: str,
        *,
        sequence: str | None = None,
        load_imu: bool = False,
        load_gt_pose: bool = False,
        event_load_mode: ResidentLoadMode = "lazy",
        image_load_mode: ResidentLoadMode = "lazy",
        depth_load_mode: LoadMode = True,
        precompute_frame_event_indices: bool | None = None,
        cache_dir: str | None = None,
        sensor_resolution: tuple[int, int] | None = None,
    ) -> None:
        """Initialize one DAVIS/RPG text recording."""
        self.root = os.path.abspath(os.path.expanduser(root))
        self.recording_dir = _resolve_recording_dir(root, sequence)
        self.sequence = sequence or os.path.basename(self.recording_dir)
        self._event_load_mode = LoadingType.from_resident_value(
            event_load_mode,
            name="event_load_mode",
        )
        self._image_load_mode = LoadingType.from_resident_value(
            image_load_mode,
            name="image_load_mode",
        )
        self._depth_load_mode = LoadingType.from_value(
            depth_load_mode,
            name="depth_load_mode",
        )
        if precompute_frame_event_indices is None:
            should_precompute_frame_indices = self._event_load_mode is LoadingType.CACHED
        else:
            should_precompute_frame_indices = precompute_frame_event_indices
        self._precompute_frame_event_indices = should_precompute_frame_indices
        self._cache_dir = resolve_davis_cache_dir(cache_dir)

        self._events_path = _require_file(
            os.path.join(self.recording_dir, "events.txt"),
            "events",
        )
        self._images_path = _optional_file(os.path.join(self.recording_dir, "images.txt"))
        self._imu_path = _optional_file(os.path.join(self.recording_dir, "imu.txt"))
        self._pose_path = _optional_file(os.path.join(self.recording_dir, "groundtruth.txt"))
        self._calib_path = _optional_file(os.path.join(self.recording_dir, "calib.txt"))
        self._depth_path = _optional_file(os.path.join(self.recording_dir, "depthmaps.txt"))

        self._event_backend = _TextEventBackend(
            self._events_path,
            self._cache_dir,
            self._event_load_mode,
        )

        self._frame_timestamps: npt.NDArray[np.float64] | None = None
        self._frame_event_indices: npt.NDArray[np.int64] | None = None
        self._frame_event_index_cache: dict[int, int] = {}
        self._image_paths: list[str] = []
        self._images_cached: npt.NDArray[np.uint8] | None = None
        self._lazy_image_cache = LazyDecodeCache[npt.NDArray[np.uint8]](_IMAGE_CACHE_ITEMS)
        self._image_shape: tuple[int, int] | None = None
        self._explicit_sensor_resolution = _validate_sensor_resolution(sensor_resolution)
        self._undistort_map: npt.NDArray[np.float32] | None = None
        self._undistort_map_ready = False
        self._init_images()

        self._depth_timestamps: npt.NDArray[np.float64] | None = None
        self._depth_paths: list[str] = []
        self._depth_cached: npt.NDArray[np.float32] | None = None
        self._lazy_depth_cache = LazyDecodeCache[npt.NDArray[np.float32]](_DEPTH_CACHE_ITEMS)
        self._init_depth()

        self._calibration = _load_calibration(self._calib_path)
        self._imu = None
        if load_imu:
            self._imu = _load_imu(self._imu_path)

        self._pose = None
        if load_gt_pose:
            self._pose = _load_pose(self._pose_path)
        self._closed = False

    def _init_images(self) -> None:
        if self._images_path is None:
            return

        frame_timestamps, image_paths = _load_image_index(
            self._images_path,
            recording_dir=self.recording_dir,
            description="image index",
        )
        self._frame_timestamps = frame_timestamps
        self._image_paths = image_paths
        if self._precompute_frame_event_indices:
            frame_event_indices = self.times_to_indices(frame_timestamps)
            self._frame_event_indices = frame_event_indices + 1
            freeze_array(self._frame_event_indices)

        if not image_paths:
            return

        if self._image_load_mode is LoadingType.CACHED:
            images = cast(
                npt.NDArray[np.uint8],
                decode_to_array_stack(
                    image_paths,
                    _decode_grayscale_image,
                    dtype=np.uint8,
                    max_workers=_IMAGE_DECODE_PARALLELISM,
                    description="DAVIS image",
                ),
            )
            if images.ndim != 3:
                raise ValueError(f"DAVIS images must decode to a 3D stack, got {images.shape}.")
            image_height = int(images.shape[1])
            image_width = int(images.shape[2])
            self._image_shape = (image_height, image_width)
            freeze_array(images)
            self._images_cached = images

    def _init_depth(self) -> None:
        if self._depth_path is None or not self._depth_load_mode.should_load:
            return

        depth_timestamps, depth_paths = _load_image_index(
            self._depth_path,
            recording_dir=self.recording_dir,
            description="depth index",
        )
        self._depth_timestamps = depth_timestamps
        self._depth_paths = depth_paths

        if self._depth_load_mode is LoadingType.CACHED and depth_paths:
            depth_stack = cast(
                npt.NDArray[np.float32],
                decode_to_array_stack(
                    depth_paths,
                    _decode_depth,
                    dtype=np.float32,
                    max_workers=_DEPTH_DECODE_PARALLELISM,
                    description="DAVIS depth",
                ),
            )
            if depth_stack.ndim != 3:
                raise ValueError(
                    f"DAVIS depth maps must decode to a 3D stack, got {depth_stack.shape}."
                )
            freeze_array(depth_stack)
            self._depth_cached = depth_stack

    # DataLoaderBase interface

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in ``[start_index, end_index)``."""
        return self._event_backend.load_events(start_index, end_index)

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return self._event_backend.num_events

    def time_to_index(self, t: float) -> int:
        """Find the last event strictly before time ``t``."""
        return self._event_backend.time_to_index(t)

    def index_to_time(self, index: int) -> float:
        """Return the timestamp of one event by index."""
        return self._event_backend.index_to_time(index)

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Vectorized form of :meth:`time_to_index`."""
        return self._event_backend.times_to_indices(timestamps)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Vectorized form of :meth:`index_to_time`."""
        return self._event_backend.indices_to_times(indices)

    def close(self) -> None:
        """Release sidecar handles and process-local decode caches."""
        self._event_backend.close()
        self._lazy_image_cache.clear()
        self._lazy_depth_cache.clear()
        self._frame_event_index_cache.clear()
        self._closed = True

    def __del__(self) -> None:
        """Warn if process local resources were not released explicitly."""
        if getattr(self, "_closed", True):
            return

        warnings.warn(
            f"DavisRecordingLoader for {getattr(self, 'recording_dir', '?')!r} "
            "was not closed. Call .close() or use a context manager.",
            ResourceWarning,
            stacklevel=2,
        )
        self.close()

    # Recording metadata

    @property
    def event_load_mode(self) -> LoadingType:
        """Configured event loading mode."""
        return self._event_load_mode

    @property
    def image_load_mode(self) -> LoadingType:
        """Configured image loading mode."""
        return self._image_load_mode

    @property
    def depth_load_mode(self) -> LoadingType:
        """Configured depth-map loading mode."""
        return self._depth_load_mode

    @property
    def cache_dir(self) -> str:
        """Root directory for DAVIS sidecar caches."""
        return self._cache_dir

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """Frame shape as ``(height, width)``, if frames are available.

        In lazy image mode the shape is discovered by decoding the first frame on first access,
        so the value agrees with cached image mode.
        """
        if self._image_shape is None and self._image_paths:
            self.load_image(0)
        return self._image_shape

    @property
    def calibration(self) -> DavisCameraCalibration | None:
        """Parsed OpenCV pinhole calibration, if ``calib.txt`` exists."""
        return self._calibration

    @property
    def sensor_resolution(self) -> tuple[int, int] | None:
        """Event sensor resolution as ``(height, width)``.

        Uses the explicit value passed at construction, otherwise the frame
        shape.

        Returns None when neither is available.
        """
        if self._explicit_sensor_resolution is not None:
            return self._explicit_sensor_resolution
        return self.image_shape

    # Time range

    @property
    def t_start(self) -> float | None:
        """Timestamp of the first event, or None if there are no events."""
        if self.num_events == 0:
            return None
        return self.index_to_time(0)

    @property
    def t_end(self) -> float | None:
        """Timestamp of the last event, or None if there are no events."""
        if self.num_events == 0:
            return None
        return self.index_to_time(-1)

    @property
    def duration(self) -> float | None:
        """Span between the first and last event timestamps, in seconds."""
        if self.num_events == 0:
            return None
        return self.index_to_time(-1) - self.index_to_time(0)

    # Undistortion

    def _require_undistort_inputs(self) -> tuple[DavisCameraCalibration, tuple[int, int]]:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for DAVIS undistortion.")
        if self._calibration is None:
            raise RuntimeError("DAVIS undistortion requires calib.txt; none was found.")
        resolution = self.sensor_resolution
        if resolution is None:
            raise RuntimeError(
                "DAVIS undistortion requires a sensor resolution. Pass "
                "sensor_resolution=(height, width) when the recording has no frames."
            )
        return self._calibration, resolution

    @property
    def undistort_map(self) -> npt.NDArray[np.float32] | None:
        """Per-pixel map from distorted to undistorted coordinates.

        Shape ``(height, width, 2)``; ``map[y, x]`` is the undistorted
        ``(x, y)`` location of source pixel ``(x, y)`` in the same camera frame.

        Returns None when calibration or resolution is unavailable.
        Computed once and cached.
        """
        if self._undistort_map_ready:
            return self._undistort_map

        if cv2 is None or self._calibration is None or self.sensor_resolution is None:
            self._undistort_map_ready = True
            return None

        height, width = self.sensor_resolution
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            indexing="xy",
        )
        source_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        undistorted_points = cv2.undistortPoints(
            source_points.reshape(-1, 1, 2),
            self._calibration.camera_matrix,
            self._calibration.distortion_coefficients,
            P=self._calibration.camera_matrix,
        )
        undistort_map = np.asarray(
            undistorted_points.reshape(height, width, 2),
            dtype=np.float32,
        )
        freeze_array(undistort_map)
        self._undistort_map = undistort_map
        self._undistort_map_ready = True
        return self._undistort_map

    def undistort_events(self, events: RawEvents) -> RawEvents:
        """Map event coordinates into the undistorted camera frame.

        Coordinates are looked up in ``undistort_map`` and rounded to the nearest pixel,
        undistorted points may fall outside the sensor bounds.

        Raises ``ImportError`` if OpenCV is unavailable
        and ``RuntimeError`` if calibration or sensor resolution is unavailable.
        """
        self._require_undistort_inputs()
        # _require_undistort_inputs guarantees the map is built
        undistort_map = cast(npt.NDArray[np.float32], self.undistort_map)
        height, width = undistort_map.shape[:2]

        clipped_x = np.clip(events.x.astype(np.intp), 0, width - 1)
        clipped_y = np.clip(events.y.astype(np.intp), 0, height - 1)
        mapped = undistort_map[clipped_y, clipped_x]
        return RawEvents(
            x=np.rint(mapped[:, 0]).astype(np.int16),
            y=np.rint(mapped[:, 1]).astype(np.int16),
            timestamp=events.timestamp.copy(),
            polarity=events.polarity.copy(),
        )

    def undistort_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Return an undistorted copy of one grayscale frame.

        Raises ``ImportError`` if OpenCV is unavailable
        and ``RuntimeError`` if calibration is unavailable.
        """
        calibration, _ = self._require_undistort_inputs()
        undistorted = cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            calibration.camera_matrix,
        )
        return np.asarray(undistorted, dtype=image.dtype)

    # Frames / images

    @property
    def has_images(self) -> bool:
        """Whether ``images.txt`` and referenced frames are available."""
        return self._frame_timestamps is not None

    @property
    def frame_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Frame timestamps from ``images.txt``, or None if unavailable."""
        return self._frame_timestamps

    @property
    def frame_event_indices(self) -> npt.NDArray[np.int64] | None:
        """First event index at or after each frame timestamp."""
        return self._frame_event_indices

    @property
    def num_frames(self) -> int:
        """Number of referenced grayscale frames."""
        return len(self._image_paths)

    def normalize_frame_index(self, frame_index: int) -> int:
        """Normalize negative frame index and validate bounds."""
        return normalize_index(frame_index, self.num_frames, "frame")

    def _nearest_timestamp_index(
        self,
        timestamps: npt.NDArray[np.float64] | None,
        t: float,
        *,
        description: str,
    ) -> int:
        """Return the index of the ``timestamps`` entry nearest to ``t``."""
        if timestamps is None or timestamps.size == 0:
            raise RuntimeError(f"{description} timestamps are not available for this recording.")
        return find_nearest_index(timestamps, t)

    def find_nearest_frame_index(self, t: float) -> int:
        """Return the frame index nearest to ``t``."""
        return self._nearest_timestamp_index(self._frame_timestamps, t, description="Frame")

    # Aliases matching the image oriented naming used by the other loaders

    @property
    def num_images(self) -> int:
        """Alias of ``num_frames``."""
        return self.num_frames

    @property
    def image_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Alias of ``frame_timestamps``."""
        return self._frame_timestamps

    def find_nearest_image_index(self, t: float) -> int:
        """Alias of :meth:`find_nearest_frame_index`."""
        return self.find_nearest_frame_index(t)

    def _event_index_at_frame(self, frame_index: int) -> int:
        if self._frame_event_indices is not None:
            return int(self._frame_event_indices[frame_index])

        cached_index = self._frame_event_index_cache.get(frame_index)
        if cached_index is not None:
            return cached_index

        if self._frame_timestamps is None:
            raise RuntimeError("Frame timestamps are not available for this recording.")
        frame_time = float(self._frame_timestamps[frame_index])
        event_index = self.time_to_index(frame_time) + 1
        self._frame_event_index_cache[frame_index] = event_index
        return event_index

    def load_image(self, frame_index: int) -> npt.NDArray[np.uint8] | None:
        """Load one grayscale frame by index."""
        if not self.has_images:
            return None

        index = self.normalize_frame_index(frame_index)
        if self._images_cached is not None:
            return self._images_cached[index]  # type: ignore[no-any-return]

        cached = self._lazy_image_cache.get(index)
        if cached is not None:
            return cached

        image = _decode_grayscale_image(self._image_paths[index])
        if self._image_shape is None:
            image_height = int(image.shape[0])
            image_width = int(image.shape[1])
            self._image_shape = (image_height, image_width)
        elif image.shape != self._image_shape:
            expected_shape = self._image_shape
            actual_shape = image.shape
            image_path = self._image_paths[index]
            raise ValueError(
                f"DAVIS image shape mismatch: expected {expected_shape}, "
                f"got {actual_shape} for {image_path}."
            )
        return self._lazy_image_cache.put(index, image)

    def load_frame_sample(self, frame_index: int) -> DavisFrameSample:
        """Load events and optional modalities associated with one frame."""
        if self._frame_timestamps is None:
            raise RuntimeError("Frame timestamps are not available for this recording.")

        index = self.normalize_frame_index(frame_index)
        frame_time = float(self._frame_timestamps[index])
        event_end = self._event_index_at_frame(index)

        # Frame 0 spans from the recording start; later frames span from the previous frame's timestamp
        if index == 0:
            event_start = 0
            start_time = self.index_to_time(0) if self.num_events else frame_time
        else:
            previous_frame_index = index - 1
            event_start = self._event_index_at_frame(previous_frame_index)
            start_time = float(self._frame_timestamps[previous_frame_index])

        events = self.load_events(event_start, event_end)

        imu = None
        if self.has_imu:
            imu = self.load_imu(start_time, frame_time)

        pose = None
        if self.has_gt_pose:
            pose = self.load_nearest_pose(frame_time)

        depth = None
        if self._depth_timestamps is not None and self._depth_timestamps.size > 0:
            depth_index = self.find_nearest_depth_index(frame_time)
            depth = self.load_depth(depth_index)

        return DavisFrameSample(
            events=events,
            timestamp=frame_time,
            image=self.load_image(index),
            imu=imu,
            pose=pose,
            depth=depth,
        )

    # IMU

    @property
    def has_imu(self) -> bool:
        """Whether IMU data were requested and loaded."""
        return self._imu is not None

    @property
    def imu_timestamps(self) -> npt.NDArray[np.float64] | None:
        """IMU timestamps, or None if IMU was not loaded."""
        if self._imu is None:
            return None
        return self._imu["timestamp"]

    @property
    def imu_data(self) -> DavisImuData | None:
        """Loaded IMU arrays, or None if IMU was not loaded."""
        return self._imu

    def load_imu(self, t_start: float, t_end: float) -> DavisImuData | None:
        """Return IMU samples in ``[t_start, t_end)``."""
        if self._imu is None:
            return None

        timestamps = self._imu["timestamp"]
        start, end = np.searchsorted(timestamps, [t_start, t_end], side="left")
        start_index = int(start)
        end_index = int(end)
        imu_slice = slice(start_index, end_index)
        imu_acceleration = self._imu["linear_acceleration"][imu_slice].copy()
        imu_angular_velocity = self._imu["angular_velocity"][imu_slice].copy()
        imu_timestamps = timestamps[imu_slice].copy()
        return DavisImuData(
            timestamp=imu_timestamps,
            linear_acceleration=imu_acceleration,
            angular_velocity=imu_angular_velocity,
        )

    # Ground truth pose

    @property
    def has_gt_pose(self) -> bool:
        """Whether ground-truth pose data were requested and loaded."""
        return self._pose is not None

    @property
    def gt_pose(self) -> DavisPoseData | None:
        """Loaded ground-truth pose trajectory, or None if not loaded."""
        return self._pose

    @property
    def gt_pose_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Ground-truth pose timestamps, or None if not loaded."""
        if self._pose is None:
            return None
        return self._pose["timestamp"]

    def load_nearest_pose(self, t: float) -> npt.NDArray[np.float64] | None:
        """Return nearest raw pose row ``[px, py, pz, qx, qy, qz, qw]``.

        Returns None when no ground-truth trajectory is available, including
        the degenerate case of a present-but-empty ``groundtruth.txt``.
        """
        if self._pose is None:
            return None

        timestamps = self._pose["timestamp"]
        if timestamps.size == 0:
            return None
        index = find_nearest_index(timestamps, t)
        position = self._pose["position"][index]
        quaternion = self._pose["quaternion"][index]
        pose_values = np.concatenate([position, quaternion])
        pose_row = np.asarray(pose_values, dtype=np.float64)
        return pose_row

    # Depth maps

    @property
    def has_depth(self) -> bool:
        """Whether ``depthmaps.txt`` is available."""
        return self._depth_timestamps is not None

    @property
    def depth_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Depth-map timestamps, or None if unavailable."""
        return self._depth_timestamps

    @property
    def num_depth_maps(self) -> int:
        """Number of referenced depth maps."""
        return len(self._depth_paths)

    def normalize_depth_index(self, depth_index: int) -> int:
        """Normalize negative depth-map index and validate bounds."""
        return normalize_index(depth_index, self.num_depth_maps, "depth")

    def find_nearest_depth_index(self, t: float) -> int:
        """Return the depth-map index nearest to ``t``."""
        return self._nearest_timestamp_index(self._depth_timestamps, t, description="Depth-map")

    def load_depth(self, depth_index: int) -> npt.NDArray[np.float32] | None:
        """Load one depth map by index if depth maps are available."""
        if not self.has_depth:
            return None

        index = self.normalize_depth_index(depth_index)
        if self._depth_cached is not None:
            return self._depth_cached[index]  # type: ignore[no-any-return]

        cached = self._lazy_depth_cache.get(index)
        if cached is not None:
            return cached

        depth = _decode_depth(self._depth_paths[index])
        return self._lazy_depth_cache.put(index, depth)
