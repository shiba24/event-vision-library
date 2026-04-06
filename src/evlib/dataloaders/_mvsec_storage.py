"""Private storage backends for MVSEC loader internals."""

import abc
import hashlib
import json
import os
import shutil
import uuid
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
from numpy.lib.format import open_memmap

from evlib.codec.fileformat.hdf5 import open_hdf5
from evlib.types import RawEvents


ResidentLoadMode = Literal["cached", "lazy"]

# bump these when the on disk sidecar format changes
_EVENT_CACHE_SCHEMA_VERSION = 1
_EVENT_BUILD_BLOCK_ROWS = 1_000_000  # keep peak memory bounded during HDF5 reads
_GT_FLOW_CACHE_SCHEMA_VERSION = 1


class _EventCacheMetadata(TypedDict):
    schema_version: int
    source_path: str
    dataset_key: str
    source_size: int
    source_mtime_ns: int
    num_events: int


class _GTFlowCacheMetadata(TypedDict):
    schema_version: int
    source_path: str
    source_size: int
    source_mtime_ns: int


def normalize_resident_load_mode(name: str, value: str) -> ResidentLoadMode:
    if value not in ("cached", "lazy"):
        raise ValueError(f"{name} must be 'cached' or 'lazy', got {value!r}")
    return cast(ResidentLoadMode, value)


def resolve_mvsec_cache_dir(cache_dir: Optional[str]) -> str:
    """Resolve per user MVSEC cache directory.

    Falls back to ``$XDG_CACHE_HOME/evlib/mvsec`` or ``~/.cache/evlib/mvsec``.
    """
    if cache_dir is not None:
        return os.path.abspath(os.path.expanduser(cache_dir))

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg is not None:
        base = os.path.expanduser(xdg)
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache")

    return os.path.abspath(os.path.join(base, "evlib", "mvsec"))


# Cache signatures and paths


def _make_cache_signature(
    source_path: str,
    dataset_key: str,
    source_size: int,
    source_mtime_ns: int,
) -> str:
    """Content addressed hash so a changed source file gets a fresh sidecar."""
    parts = [
        os.path.abspath(source_path),
        dataset_key,
        str(source_size),
        str(source_mtime_ns),
        str(_EVENT_CACHE_SCHEMA_VERSION),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _make_event_cache_dir(
    cache_root: str,
    source_path: str,
    dataset_key: str,
    sequence: str,
    camera: str,
) -> str:
    stat = os.stat(source_path)
    sig = _make_cache_signature(
        source_path,
        dataset_key,
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    name = f"{sequence}_{camera}_{sig[:16]}"
    return os.path.join(cache_root, "event_sidecars", name)


def _make_event_cache_paths(cache_dir: str) -> dict:
    return {
        "x": os.path.join(cache_dir, "events_x.npy"),
        "y": os.path.join(cache_dir, "events_y.npy"),
        "timestamp": os.path.join(cache_dir, "events_t.npy"),
        "polarity": os.path.join(cache_dir, "events_p.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def _make_gt_flow_cache_dir(
    cache_root: str,
    source_path: str,
    sequence: str,
) -> str:
    stat = os.stat(source_path)
    parts = [
        os.path.abspath(source_path),
        str(int(stat.st_size)),
        str(int(stat.st_mtime_ns)),
        str(_GT_FLOW_CACHE_SCHEMA_VERSION),
    ]
    sig = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return os.path.join(cache_root, "gt_flow_sidecars", f"{sequence}_{sig[:16]}")


def _make_gt_flow_cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "x": os.path.join(cache_dir, "x_flow.npy"),
        "y": os.path.join(cache_dir, "y_flow.npy"),
        "timestamp": os.path.join(cache_dir, "timestamps.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def _write_json(path: str, data: Union[_EventCacheMetadata, _GTFlowCacheMetadata]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# Cache metadata and staleness checks


def _load_event_cache_metadata(metadata_path: str) -> Optional[_EventCacheMetadata]:
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return cast(_EventCacheMetadata, json.load(f))


def _load_gt_flow_cache_metadata(metadata_path: str) -> Optional[_GTFlowCacheMetadata]:
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return cast(_GTFlowCacheMetadata, json.load(f))


def _event_cache_is_complete(
    cache_dir: str,
    source_path: str,
    dataset_key: str,
) -> Optional[_EventCacheMetadata]:
    """Return metadata if the sidecar is complete and still matches the source file."""
    paths = _make_event_cache_paths(cache_dir)
    meta = _load_event_cache_metadata(paths["metadata"])
    if meta is None:
        return None

    if not all(os.path.isfile(paths[k]) for k in ("x", "y", "timestamp", "polarity")):
        return None

    # staleness - size + mtime, not content hash (good enough, and fast)
    stat = os.stat(source_path)
    if (
        meta["schema_version"] != _EVENT_CACHE_SCHEMA_VERSION
        or meta["source_path"] != os.path.abspath(source_path)
        or meta["dataset_key"] != dataset_key
        or meta["source_size"] != int(stat.st_size)
        or meta["source_mtime_ns"] != int(stat.st_mtime_ns)
    ):
        return None

    return meta


def _gt_flow_cache_is_complete(
    cache_dir: str,
    source_path: str,
) -> Optional[_GTFlowCacheMetadata]:
    paths = _make_gt_flow_cache_paths(cache_dir)
    meta = _load_gt_flow_cache_metadata(paths["metadata"])
    if meta is None:
        return None

    if not all(os.path.isfile(paths[k]) for k in ("x", "y", "timestamp")):
        return None

    stat = os.stat(source_path)
    if (
        meta["schema_version"] != _GT_FLOW_CACHE_SCHEMA_VERSION
        or meta["source_path"] != os.path.abspath(source_path)
        or meta["source_size"] != int(stat.st_size)
        or meta["source_mtime_ns"] != int(stat.st_mtime_ns)
    ):
        return None

    return meta


# Building sidecars

# Both builders use the same pattern - write into a uuid temp dir then
# atomic os.replace into the real path, readers never see half written files

# TODO: no file lock, concurrent builds of the same sidecar will race but
# the last os.replace wins and the result is still valid.


def _copy_event_rows_into_columns(
    event_dataset: h5py.Dataset,
    x_values: npt.NDArray[np.int16],
    y_values: npt.NDArray[np.int16],
    timestamp_values: npt.NDArray[np.float64],
    polarity_values: npt.NDArray[np.bool_],
) -> None:
    """Blockwise HDF5 read into pre allocated typed columns.

    MVSEC stores events as (N, 4) float64 rows. Streaming in fixed blocks
    avoids loading the full table into memory for large sequences.
    """
    n = int(event_dataset.shape[0])
    buf = np.empty((_EVENT_BUILD_BLOCK_ROWS, 4), dtype=np.float64)
    start = 0

    while start < n:
        end = min(start + _EVENT_BUILD_BLOCK_ROWS, n)
        block = buf[: end - start]
        event_dataset.read_direct(block, source_sel=np.s_[start:end, :])

        x_values[start:end] = block[:, 0]
        y_values[start:end] = block[:, 1]
        timestamp_values[start:end] = block[:, 2]
        # MVSEC pol is +1/-1 float, convert to bool
        np.greater(block[:, 3], 0.0, out=polarity_values[start:end])

        start = end


def _freeze_event_columns(
    x_values: npt.NDArray[np.int16],
    y_values: npt.NDArray[np.int16],
    timestamp_values: npt.NDArray[np.float64],
    polarity_values: npt.NDArray[np.bool_],
) -> None:
    for arr in (x_values, y_values, timestamp_values, polarity_values):
        arr.flags.writeable = False


def _build_event_cache(
    source_path: str,
    dataset_key: str,
    cache_dir: str,
) -> _EventCacheMetadata:
    """Build typed npy sidecar from the HDF5 events table."""
    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)

    tmp = os.path.join(parent, f".tmp_{uuid.uuid4().hex}")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    paths = _make_event_cache_paths(tmp)

    try:
        with open_hdf5(source_path) as h5:
            ds = h5[dataset_key]
            n = int(ds.shape[0])

            x = open_memmap(paths["x"], mode="w+", dtype=np.int16, shape=(n,))
            y = open_memmap(paths["y"], mode="w+", dtype=np.int16, shape=(n,))
            t = open_memmap(paths["timestamp"], mode="w+", dtype=np.float64, shape=(n,))
            p = open_memmap(paths["polarity"], mode="w+", dtype=np.bool_, shape=(n,))

            _copy_event_rows_into_columns(ds, x, y, t, p)

            x.flush()
            y.flush()
            t.flush()
            p.flush()

        stat = os.stat(source_path)
        meta: _EventCacheMetadata = {
            "schema_version": _EVENT_CACHE_SCHEMA_VERSION,
            "source_path": os.path.abspath(source_path),
            "dataset_key": dataset_key,
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "num_events": n,
        }
        _write_json(paths["metadata"], meta)

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(tmp, cache_dir)
        return meta
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _build_gt_flow_cache(
    source_path: str,
    cache_dir: str,
) -> _GTFlowCacheMetadata:
    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)

    tmp = os.path.join(parent, f".tmp_{uuid.uuid4().hex}")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    paths = _make_gt_flow_cache_paths(tmp)

    try:
        with np.load(source_path) as npz:
            x_flow = np.asarray(npz["x_flow_dist"], dtype=np.float32)
            y_flow = np.asarray(npz["y_flow_dist"], dtype=np.float32)
            timestamps = np.asarray(npz["timestamps"], dtype=np.float64)

        np.save(paths["x"], x_flow)
        np.save(paths["y"], y_flow)
        np.save(paths["timestamp"], timestamps)

        stat = os.stat(source_path)
        meta: _GTFlowCacheMetadata = {
            "schema_version": _GT_FLOW_CACHE_SCHEMA_VERSION,
            "source_path": os.path.abspath(source_path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }
        _write_json(paths["metadata"], meta)

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(tmp, cache_dir)
        return meta
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# Loading from sidecars


def _prepare_event_cache(
    source_path: str,
    dataset_key: str,
    sequence: str,
    camera: str,
    cache_root: str,
) -> tuple[str, dict[str, str], _EventCacheMetadata]:
    """Ensure an event sidecar exists and return (cache_dir, paths, metadata)."""
    src = os.path.abspath(source_path)
    cache_dir = _make_event_cache_dir(cache_root, src, dataset_key, sequence, camera)
    paths = _make_event_cache_paths(cache_dir)
    meta = _event_cache_is_complete(cache_dir, src, dataset_key)
    if meta is None:
        meta = _build_event_cache(src, dataset_key, cache_dir)
    return cache_dir, paths, meta


def load_mvsec_cached_events(
    source_path: str,
    dataset_key: str,
    sequence: str,
    camera: str,
    cache_root: str,
) -> tuple[
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Load event columns from a persistent sidecar into RAM (frozen)."""
    _, paths, _ = _prepare_event_cache(source_path, dataset_key, sequence, camera, cache_root)
    x = np.load(paths["x"])
    y = np.load(paths["y"])
    t = np.load(paths["timestamp"])
    p = np.load(paths["polarity"])
    _freeze_event_columns(x, y, t, p)
    return x, y, t, p


def _load_gt_flow_from_npz_file(
    source_path: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    with np.load(source_path) as npz:
        x = np.asarray(npz["x_flow_dist"], dtype=np.float32)
        y = np.asarray(npz["y_flow_dist"], dtype=np.float32)
        ts = np.asarray(npz["timestamps"], dtype=np.float64)
    return x, y, ts


def load_mvsec_gt_flow(
    source_path: str,
    sequence: str,
    cache_root: str,
    load_mode: ResidentLoadMode,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Load GT flow arrays, building a decompressed sidecar cache if needed."""
    try:
        cache_dir = _make_gt_flow_cache_dir(cache_root, source_path, sequence)
        meta = _gt_flow_cache_is_complete(cache_dir, source_path)
        if meta is None:
            meta = _build_gt_flow_cache(source_path, cache_dir)

        paths = _make_gt_flow_cache_paths(cache_dir)
        mmap: Optional[Literal["r"]] = "r" if load_mode == "lazy" else None

        x_flow = np.load(paths["x"], mmap_mode=mmap)
        y_flow = np.load(paths["y"], mmap_mode=mmap)
        timestamps = np.load(paths["timestamp"], mmap_mode=mmap)

        if load_mode == "cached":
            x_flow = np.asarray(x_flow, dtype=np.float32)
            y_flow = np.asarray(y_flow, dtype=np.float32)
            timestamps = np.asarray(timestamps, dtype=np.float64)

        return x_flow, y_flow, timestamps
    except OSError:
        # cache might be on a read only fs or gone, fall back to raw npz
        return _load_gt_flow_from_npz_file(source_path)


# Lazy HDF5 reader (process safe, pickle safe)


class _LazyH5Dataset:
    """Process local lazy reader for a single HDF5 dataset.

    Reopens the file handle after fork() or unpickle since HDF5
    handles are not safe across process boundaries.
    """

    def __init__(self, file_path: str, dataset_key: str, dtype: type) -> None:
        self._file_path = file_path
        self._dataset_key = dataset_key
        self._dtype = dtype
        self._file: Optional[h5py.File] = None
        self._dataset: Optional[h5py.Dataset] = None
        self._pid: Optional[int] = None
        self._closed = False

    def __getstate__(self) -> dict:
        # HDF5 handles cant cross process boundaries, strip them
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        state["_pid"] = None
        return state

    @property
    def has_open_handle(self) -> bool:
        return self._file is not None

    def _ensure_open(self) -> Optional[h5py.Dataset]:
        if self._closed:
            return None

        pid = os.getpid()
        if self._dataset is not None and self._pid == pid:
            return self._dataset

        self._drop_handles()

        self._file = open_hdf5(self._file_path)
        self._dataset = self._file[self._dataset_key]
        self._pid = pid
        return self._dataset

    def _drop_handles(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._dataset = None
        self._pid = None

    def read(self, index: int) -> "Optional[npt.NDArray[np.generic]]":
        ds = self._ensure_open()
        if ds is None:
            return None
        arr: npt.NDArray[np.generic] = np.asarray(ds[index], dtype=self._dtype)
        arr.flags.writeable = False
        return arr

    def close(self) -> None:
        self._drop_handles()
        self._closed = True


# Event backends


class _EventBackend(abc.ABC):
    """Interface for loading slices of an MVSEC event stream."""

    @property
    @abc.abstractmethod
    def num_events(self) -> int: ...

    @abc.abstractmethod
    def load_events(self, start_index: int, end_index: int) -> RawEvents: ...

    @abc.abstractmethod
    def time_to_index(self, t: float) -> int:
        """Index of the last event strictly before *t*."""

    @abc.abstractmethod
    def index_to_time(self, index: int) -> float: ...

    @abc.abstractmethod
    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]: ...

    @abc.abstractmethod
    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class _CachedEventBackend(_EventBackend):
    """All events resident in RAM as typed frozen columns."""

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
        """Read directly from HDF5 into frozen in memory columns."""
        n = int(event_dataset.shape[0])
        x = np.empty(n, dtype=np.int16)
        y = np.empty(n, dtype=np.int16)
        t = np.empty(n, dtype=np.float64)
        p = np.empty(n, dtype=np.bool_)

        _copy_event_rows_into_columns(event_dataset, x, y, t, p)
        _freeze_event_columns(x, y, t, p)
        return cls(x, y, t, p)

    @classmethod
    def from_sidecar(
        cls,
        source_path: str,
        dataset_key: str,
        sequence: str,
        camera: str,
        cache_root: str,
    ) -> "_CachedEventBackend":
        """Load from persistent sidecar cache."""
        x, y, t, p = load_mvsec_cached_events(
            source_path,
            dataset_key,
            sequence,
            camera,
            cache_root,
        )
        return cls(x, y, t, p)

    @property
    def num_events(self) -> int:
        return len(self._timestamp)

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        s = slice(start_index, end_index)
        return RawEvents(
            x=self._x[s].copy(),
            y=self._y[s].copy(),
            timestamp=self._timestamp[s].copy(),
            polarity=self._polarity[s].copy(),
        )

    def time_to_index(self, t: float) -> int:
        return int(self._timestamp.searchsorted(t, side="left") - 1)

    def index_to_time(self, index: int) -> float:
        return float(self._timestamp[index])

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        ts = np.asarray(timestamps, dtype=np.float64)
        return np.asarray(self._timestamp.searchsorted(ts, side="left") - 1, dtype=np.int64)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        idx = np.asarray(indices, dtype=np.int64)
        return np.asarray(self._timestamp[idx], dtype=np.float64)

    def close(self) -> None:
        pass


class _LazyEventBackend(_EventBackend):
    """Sidecar backed event access via read only memmaps.

    Builds the sidecar once, then mmaps into it. The OS pages data
    in and out so this works for sequences too large to fit in RAM.
    Memmaps are process local so we reopen after fork/unpickle.
    """

    def __init__(
        self,
        source_path: str,
        dataset_key: str,
        sequence: str,
        camera: str,
        cache_root: str,
    ) -> None:
        self._source_path = os.path.abspath(source_path)
        self._dataset_key = dataset_key
        self._sequence = sequence
        self._camera = camera
        self._cache_root = cache_root

        self._cache_dir, self._cache_paths, meta = _prepare_event_cache(
            source_path,
            dataset_key,
            sequence,
            camera,
            cache_root,
        )

        self._num_events = int(meta["num_events"])
        self._x: Optional[npt.NDArray[np.int16]] = None
        self._y: Optional[npt.NDArray[np.int16]] = None
        self._timestamp: Optional[npt.NDArray[np.float64]] = None
        self._polarity: Optional[npt.NDArray[np.bool_]] = None
        self._pid: Optional[int] = None

    def __getstate__(self) -> dict:
        # memmaps are process local, drop and reopen lazily after unpickle
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
        if (
            self._pid == pid
            and self._x is not None
            and self._y is not None
            and self._timestamp is not None
            and self._polarity is not None
        ):
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

        s = slice(start_index, end_index)
        return RawEvents(
            x=self._x[s].copy(),
            y=self._y[s].copy(),
            timestamp=self._timestamp[s].copy(),
            polarity=self._polarity[s].copy(),
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
        ts = np.asarray(timestamps, dtype=np.float64)
        return np.asarray(self._timestamp.searchsorted(ts, side="left") - 1, dtype=np.int64)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self._ensure_open()
        assert self._timestamp is not None
        idx = np.asarray(indices, dtype=np.int64)
        return np.asarray(self._timestamp[idx], dtype=np.float64)

    def close(self) -> None:
        self._x = None
        self._y = None
        self._timestamp = None
        self._polarity = None
        self._pid = None
