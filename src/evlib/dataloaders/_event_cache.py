"""Reusable event sidecar cache and backend implementations."""

from __future__ import annotations

import abc
import hashlib
import json
import os
import shutil
import uuid
from typing import Optional
from typing import TypedDict
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
from numpy.lib.format import open_memmap

from evlib.codec.fileformat.hdf5 import open_hdf5
from evlib.types import RawEvents


_EVENT_CACHE_SCHEMA_VERSION = 1
_EVENT_BUILD_BLOCK_ROWS = 1_000_000


class _EventCacheMetadata(TypedDict):
    schema_version: int
    source_path: str
    dataset_key: str
    source_size: int
    source_mtime_ns: int
    num_events: int


def _make_cache_signature(
    source_path: str,
    dataset_key: str,
    source_size: int,
    source_mtime_ns: int,
) -> str:
    """Return content addressed signature for an event source."""
    parts = [
        os.path.abspath(source_path),
        dataset_key,
        str(source_size),
        str(source_mtime_ns),
        str(_EVENT_CACHE_SCHEMA_VERSION),
    ]
    joined = "|".join(parts)
    signature = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return signature


def _make_event_cache_dir(
    cache_root: str,
    source_path: str,
    dataset_key: str,
    cache_name: str,
) -> str:
    stat_result = os.stat(source_path)
    source_size = int(stat_result.st_size)
    source_mtime_ns = int(stat_result.st_mtime_ns)
    signature = _make_cache_signature(
        source_path,
        dataset_key,
        source_size,
        source_mtime_ns,
    )
    directory_name = f"{cache_name}_{signature[:16]}"
    return os.path.join(cache_root, "event_sidecars", directory_name)


def _make_event_cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "x": os.path.join(cache_dir, "events_x.npy"),
        "y": os.path.join(cache_dir, "events_y.npy"),
        "timestamp": os.path.join(cache_dir, "events_t.npy"),
        "polarity": os.path.join(cache_dir, "events_p.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def _write_json(path: str, data: _EventCacheMetadata) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def _load_event_cache_metadata(metadata_path: str) -> Optional[_EventCacheMetadata]:
    if not os.path.isfile(metadata_path):
        return None

    with open(metadata_path, "r", encoding="utf-8") as file_handle:
        metadata = cast(_EventCacheMetadata, json.load(file_handle))
    return metadata


def _event_cache_is_complete(
    cache_dir: str,
    source_path: str,
    dataset_key: str,
) -> Optional[_EventCacheMetadata]:
    """Return metadata if the sidecar exists and still matches the source."""
    paths = _make_event_cache_paths(cache_dir)
    metadata = _load_event_cache_metadata(paths["metadata"])
    if metadata is None:
        return None

    required_keys = ("x", "y", "timestamp", "polarity")
    if not all(os.path.isfile(paths[key]) for key in required_keys):
        return None

    stat_result = os.stat(source_path)
    source_size = int(stat_result.st_size)
    source_mtime_ns = int(stat_result.st_mtime_ns)
    source_path_abs = os.path.abspath(source_path)

    is_stale = (
        metadata["schema_version"] != _EVENT_CACHE_SCHEMA_VERSION
        or metadata["source_path"] != source_path_abs
        or metadata["dataset_key"] != dataset_key
        or metadata["source_size"] != source_size
        or metadata["source_mtime_ns"] != source_mtime_ns
    )
    if is_stale:
        return None

    return metadata


def _copy_event_rows_into_columns(
    event_dataset: h5py.Dataset,
    x_values: npt.NDArray[np.int16],
    y_values: npt.NDArray[np.int16],
    timestamp_values: npt.NDArray[np.float64],
    polarity_values: npt.NDArray[np.bool_],
) -> None:
    """Copy an HDF5 event table into typed column arrays blockwise."""
    num_events = int(event_dataset.shape[0])
    buffer = np.empty((_EVENT_BUILD_BLOCK_ROWS, 4), dtype=np.float64)
    start_index = 0

    while start_index < num_events:
        end_index = min(start_index + _EVENT_BUILD_BLOCK_ROWS, num_events)
        block = buffer[: end_index - start_index]
        source_slice = np.s_[start_index:end_index, :]
        event_dataset.read_direct(block, source_sel=source_slice)

        x_values[start_index:end_index] = block[:, 0]
        y_values[start_index:end_index] = block[:, 1]
        timestamp_values[start_index:end_index] = block[:, 2]
        np.greater(block[:, 3], 0.0, out=polarity_values[start_index:end_index])

        start_index = end_index


def _freeze_event_columns(
    x_values: npt.NDArray[np.int16],
    y_values: npt.NDArray[np.int16],
    timestamp_values: npt.NDArray[np.float64],
    polarity_values: npt.NDArray[np.bool_],
) -> None:
    arrays = (x_values, y_values, timestamp_values, polarity_values)
    for array in arrays:
        array.flags.writeable = False


def _build_event_cache(
    source_path: str,
    dataset_key: str,
    cache_dir: str,
) -> _EventCacheMetadata:
    """Build a typed event sidecar from an HDF5 events table."""
    parent_dir = os.path.dirname(cache_dir)
    os.makedirs(parent_dir, exist_ok=True)

    temp_dir = os.path.join(parent_dir, f".tmp_{uuid.uuid4().hex}")
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    paths = _make_event_cache_paths(temp_dir)

    try:
        with open_hdf5(source_path) as h5_file:
            event_dataset = h5_file[dataset_key]
            num_events = int(event_dataset.shape[0])

            x_values = open_memmap(paths["x"], mode="w+", dtype=np.int16, shape=(num_events,))
            y_values = open_memmap(paths["y"], mode="w+", dtype=np.int16, shape=(num_events,))
            timestamp_values = open_memmap(
                paths["timestamp"],
                mode="w+",
                dtype=np.float64,
                shape=(num_events,),
            )
            polarity_values = open_memmap(
                paths["polarity"],
                mode="w+",
                dtype=np.bool_,
                shape=(num_events,),
            )

            _copy_event_rows_into_columns(
                event_dataset,
                x_values,
                y_values,
                timestamp_values,
                polarity_values,
            )

            x_values.flush()
            y_values.flush()
            timestamp_values.flush()
            polarity_values.flush()

        stat_result = os.stat(source_path)
        metadata: _EventCacheMetadata = {
            "schema_version": _EVENT_CACHE_SCHEMA_VERSION,
            "source_path": os.path.abspath(source_path),
            "dataset_key": dataset_key,
            "source_size": int(stat_result.st_size),
            "source_mtime_ns": int(stat_result.st_mtime_ns),
            "num_events": num_events,
        }
        _write_json(paths["metadata"], metadata)

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(temp_dir, cache_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _prepare_event_cache(
    source_path: str,
    dataset_key: str,
    cache_name: str,
    cache_root: str,
) -> tuple[str, dict[str, str], _EventCacheMetadata]:
    """Ensure an event sidecar exists and return its location and metadata."""
    source_path_abs = os.path.abspath(source_path)
    cache_dir = _make_event_cache_dir(
        cache_root,
        source_path_abs,
        dataset_key,
        cache_name,
    )
    paths = _make_event_cache_paths(cache_dir)
    metadata = _event_cache_is_complete(cache_dir, source_path_abs, dataset_key)
    if metadata is None:
        metadata = _build_event_cache(source_path_abs, dataset_key, cache_dir)
    return cache_dir, paths, metadata


def load_cached_events(
    source_path: str,
    dataset_key: str,
    cache_name: str,
    cache_root: str,
) -> tuple[
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Load event columns from a persistent sidecar into ram."""
    _, paths, _ = _prepare_event_cache(
        source_path,
        dataset_key,
        cache_name,
        cache_root,
    )
    x_values = np.load(paths["x"])
    y_values = np.load(paths["y"])
    timestamp_values = np.load(paths["timestamp"])
    polarity_values = np.load(paths["polarity"])
    _freeze_event_columns(
        x_values,
        y_values,
        timestamp_values,
        polarity_values,
    )
    return x_values, y_values, timestamp_values, polarity_values


class _EventBackend(abc.ABC):
    """Interface for loading slices of a typed event stream."""

    @property
    @abc.abstractmethod
    def num_events(self) -> int: ...

    @abc.abstractmethod
    def load_events(self, start_index: int, end_index: int) -> RawEvents: ...

    @abc.abstractmethod
    def time_to_index(self, t: float) -> int:
        """Return the last event index strictly before ``t``."""

    @abc.abstractmethod
    def index_to_time(self, index: int) -> float: ...

    @abc.abstractmethod
    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]: ...

    @abc.abstractmethod
    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class _CachedEventBackend(_EventBackend):
    """Event backend with all event columns resident in RAM."""

    def __init__(
        self,
        x_values: npt.NDArray[np.int16],
        y_values: npt.NDArray[np.int16],
        timestamp_values: npt.NDArray[np.float64],
        polarity_values: npt.NDArray[np.bool_],
    ) -> None:
        self._x = x_values
        self._y = y_values
        self._timestamp = timestamp_values
        self._polarity = polarity_values

    @classmethod
    def from_event_dataset(cls, event_dataset: h5py.Dataset) -> "_CachedEventBackend":
        """Read directly from HDF5 into frozen typed columns."""
        num_events = int(event_dataset.shape[0])
        x_values = np.empty(num_events, dtype=np.int16)
        y_values = np.empty(num_events, dtype=np.int16)
        timestamp_values = np.empty(num_events, dtype=np.float64)
        polarity_values = np.empty(num_events, dtype=np.bool_)

        _copy_event_rows_into_columns(
            event_dataset,
            x_values,
            y_values,
            timestamp_values,
            polarity_values,
        )
        _freeze_event_columns(
            x_values,
            y_values,
            timestamp_values,
            polarity_values,
        )
        return cls(
            x_values,
            y_values,
            timestamp_values,
            polarity_values,
        )

    @classmethod
    def from_sidecar(
        cls,
        source_path: str,
        dataset_key: str,
        cache_name: str,
        cache_root: str,
    ) -> "_CachedEventBackend":
        """Load typed columns from a persistent sidecar into RAM."""
        x_values, y_values, timestamp_values, polarity_values = load_cached_events(
            source_path,
            dataset_key,
            cache_name,
            cache_root,
        )
        return cls(
            x_values,
            y_values,
            timestamp_values,
            polarity_values,
        )

    @property
    def num_events(self) -> int:
        return len(self._timestamp)

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        event_slice = slice(start_index, end_index)
        return RawEvents(
            x=self._x[event_slice].copy(),
            y=self._y[event_slice].copy(),
            timestamp=self._timestamp[event_slice].copy(),
            polarity=self._polarity[event_slice].copy(),
        )

    def time_to_index(self, t: float) -> int:
        return int(self._timestamp.searchsorted(t, side="left") - 1)

    def index_to_time(self, index: int) -> float:
        return float(self._timestamp[index])

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        indices = self._timestamp.searchsorted(timestamp_array, side="left") - 1
        return np.asarray(indices, dtype=np.int64)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        index_array = np.asarray(indices, dtype=np.int64)
        timestamps = self._timestamp[index_array]
        return np.asarray(timestamps, dtype=np.float64)

    def close(self) -> None:
        return None


class _LazyEventBackend(_EventBackend):
    """Sidecar backed event backend using read only NumPy memmaps."""

    def __init__(
        self,
        source_path: str,
        dataset_key: str,
        cache_name: str,
        cache_root: str,
    ) -> None:
        self._cache_dir, self._cache_paths, metadata = _prepare_event_cache(
            source_path,
            dataset_key,
            cache_name,
            cache_root,
        )
        self._num_events = int(metadata["num_events"])
        self._x: Optional[npt.NDArray[np.int16]] = None
        self._y: Optional[npt.NDArray[np.int16]] = None
        self._timestamp: Optional[npt.NDArray[np.float64]] = None
        self._polarity: Optional[npt.NDArray[np.bool_]] = None
        self._pid: Optional[int] = None

    def __getstate__(self) -> dict:
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

    def _ensure_open(self) -> None:
        pid = os.getpid()
        has_arrays = (
            self._x is not None
            and self._y is not None
            and self._timestamp is not None
            and self._polarity is not None
        )
        if self._pid == pid and has_arrays:
            return

        self._x = np.load(self._cache_paths["x"], mmap_mode="r")
        self._y = np.load(self._cache_paths["y"], mmap_mode="r")
        self._timestamp = np.load(self._cache_paths["timestamp"], mmap_mode="r")
        self._polarity = np.load(self._cache_paths["polarity"], mmap_mode="r")
        self._pid = pid

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        self._ensure_open()

        assert self._x is not None
        assert self._y is not None
        assert self._timestamp is not None
        assert self._polarity is not None

        event_slice = slice(start_index, end_index)
        return RawEvents(
            x=self._x[event_slice].copy(),
            y=self._y[event_slice].copy(),
            timestamp=self._timestamp[event_slice].copy(),
            polarity=self._polarity[event_slice].copy(),
        )

    def time_to_index(self, t: float) -> int:
        self._ensure_open()

        assert self._timestamp is not None
        return int(self._timestamp.searchsorted(t, side="left") - 1)

    def index_to_time(self, index: int) -> float:
        self._ensure_open()

        assert self._timestamp is not None
        return float(self._timestamp[index])

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        self._ensure_open()

        assert self._timestamp is not None
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        indices = self._timestamp.searchsorted(timestamp_array, side="left") - 1
        return np.asarray(indices, dtype=np.int64)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self._ensure_open()

        assert self._timestamp is not None
        index_array = np.asarray(indices, dtype=np.int64)
        timestamps = self._timestamp[index_array]
        return np.asarray(timestamps, dtype=np.float64)

    def close(self) -> None:
        self._x = None
        self._y = None
        self._timestamp = None
        self._polarity = None
        self._pid = None
