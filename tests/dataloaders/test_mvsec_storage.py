"""Tests for MVSEC storage backends and caching utilities."""

import json
import os
from unittest import mock

import h5py
import numpy as np
import pytest

from evlib.dataloaders import LoadingType
from evlib.dataloaders._event_cache import _build_event_cache
from evlib.dataloaders._event_cache import _CachedEventBackend
from evlib.dataloaders._event_cache import _copy_event_rows_into_columns
from evlib.dataloaders._event_cache import _event_cache_is_complete
from evlib.dataloaders._event_cache import _freeze_event_columns
from evlib.dataloaders._event_cache import _LazyEventBackend
from evlib.dataloaders._event_cache import _load_event_cache_metadata
from evlib.dataloaders._event_cache import _make_cache_signature
from evlib.dataloaders._event_cache import _make_event_cache_dir
from evlib.dataloaders._event_cache import _make_event_cache_paths
from evlib.dataloaders._event_cache import _prepare_event_cache
from evlib.dataloaders._event_cache import _write_json as _write_event_json
from evlib.dataloaders._event_cache import load_cached_events
from evlib.dataloaders._mvsec_storage import _build_gt_flow_cache
from evlib.dataloaders._mvsec_storage import _gt_flow_cache_is_complete
from evlib.dataloaders._mvsec_storage import _load_gt_flow_cache_metadata
from evlib.dataloaders._mvsec_storage import _make_gt_flow_cache_dir
from evlib.dataloaders._mvsec_storage import _make_gt_flow_cache_paths
from evlib.dataloaders._mvsec_storage import _write_json as _write_gt_flow_json
from evlib.dataloaders._mvsec_storage import load_mvsec_gt_flow
from evlib.dataloaders._mvsec_storage import resolve_mvsec_cache_dir
from evlib.dataloaders._storage_common import _LazyH5Dataset


N_EVENTS = 50
DATASET_KEY = "davis/left/events"


def _make_hdf5(path: str, n_events: int = N_EVENTS) -> None:
    """Create a minimal MVSEC-like HDF5 file."""
    with h5py.File(path, "w") as f:
        events = np.zeros((n_events, 4), dtype=np.float64)
        events[:, 0] = np.arange(n_events) % 10  # x
        events[:, 1] = np.arange(n_events) % 8  # y
        events[:, 2] = np.linspace(0.0, 1.0, n_events)  # timestamp
        events[:, 3] = np.tile([0.0, 1.0], n_events // 2 + 1)[:n_events]  # polarity
        f.create_dataset(DATASET_KEY, data=events)


def _make_gt_flow_npz(path: str, n_frames: int = 5) -> None:
    """Create a minimal GT flow NPZ file."""
    h, w = 4, 4
    np.savez(
        path,
        x_flow_dist=np.random.randn(n_frames, h, w).astype(np.float32),
        y_flow_dist=np.random.randn(n_frames, h, w).astype(np.float32),
        timestamps=np.linspace(0.0, 1.0, n_frames).astype(np.float64),
    )


class TestLoadingType:
    def test_cached(self) -> None:
        result = LoadingType.from_resident_value("cached", name="events")
        assert result is LoadingType.CACHED

    def test_lazy(self) -> None:
        result = LoadingType.from_resident_value("lazy", name="events")
        assert result is LoadingType.LAZY

    def test_from_resident_enum(self) -> None:
        result = LoadingType.from_resident_value(LoadingType.LAZY, name="events")
        assert result is LoadingType.LAZY

    def test_bool_is_invalid_for_resident_mode(self) -> None:
        with pytest.raises(ValueError, match="events must be"):
            LoadingType.from_resident_value(True, name="events")  # type: ignore[arg-type]

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="events must be"):
            LoadingType.from_resident_value("invalid", name="events")  # type: ignore[arg-type]


class TestResolveCacheDir:
    def test_explicit_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = resolve_mvsec_cache_dir(str(tmp_path / "custom"))
        assert result == str(tmp_path / "custom")

    def test_default_uses_xdg(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/test_xdg")
        result = resolve_mvsec_cache_dir(None)
        assert "evlib" in result
        assert "mvsec" in result

    def test_default_without_xdg(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = resolve_mvsec_cache_dir(None)
        assert ".cache" in result
        assert "evlib" in result


class TestCacheSignature:
    def test_deterministic(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        sig1 = _make_cache_signature(hdf5_path, DATASET_KEY, 100, 999)
        sig2 = _make_cache_signature(hdf5_path, DATASET_KEY, 100, 999)
        assert sig1 == sig2

    def test_different_inputs_differ(self) -> None:
        sig1 = _make_cache_signature("/a", "k", 100, 999)
        sig2 = _make_cache_signature("/b", "k", 100, 999)
        assert sig1 != sig2


class TestEventCachePaths:
    def test_keys(self) -> None:
        paths = _make_event_cache_paths("/tmp/cache")
        assert set(paths.keys()) == {"x", "y", "timestamp", "polarity", "metadata"}

    def test_event_cache_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = _make_event_cache_dir(
            str(tmp_path), hdf5_path, DATASET_KEY, "indoor_flying1_left"
        )
        assert "indoor_flying1_left_" in cache_dir


class TestGTFlowCachePaths:
    def test_keys(self) -> None:
        paths = _make_gt_flow_cache_paths("/tmp/cache")
        assert set(paths.keys()) == {"x", "y", "timestamp", "metadata"}

    def test_gt_flow_cache_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path)
        cache_dir = _make_gt_flow_cache_dir(str(tmp_path), npz_path, "indoor_flying1")
        assert "indoor_flying1_" in cache_dir


class TestEventCacheMetadata:
    def test_load_missing_returns_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _load_event_cache_metadata(str(tmp_path / "missing.json"))
        assert result is None

    def test_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        metadata = {
            "schema_version": 1,
            "source_path": "/test",
            "dataset_key": "k",
            "source_size": 100,
            "source_mtime_ns": 999,
            "num_events": 50,
        }
        path = str(tmp_path / "metadata.json")
        _write_event_json(path, metadata)  # type: ignore[arg-type]
        loaded = _load_event_cache_metadata(path)
        assert loaded == metadata


class TestGTFlowCacheMetadata:
    def test_load_missing_returns_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _load_gt_flow_cache_metadata(str(tmp_path / "missing.json"))
        assert result is None


class TestEventCacheComplete:
    def test_returns_none_when_no_metadata(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        result = _event_cache_is_complete(str(tmp_path / "cache"), hdf5_path, DATASET_KEY)
        assert result is None

    def test_returns_none_when_files_missing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        stat = os.stat(hdf5_path)
        metadata = {
            "schema_version": 1,
            "source_path": os.path.abspath(hdf5_path),
            "dataset_key": DATASET_KEY,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "num_events": N_EVENTS,
        }
        _write_event_json(os.path.join(cache_dir, "metadata.json"), metadata)  # type: ignore[arg-type]
        result = _event_cache_is_complete(cache_dir, hdf5_path, DATASET_KEY)
        assert result is None

    def test_returns_none_when_stale(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        paths = _make_event_cache_paths(cache_dir)
        for key in ("x", "y", "timestamp", "polarity"):
            np.save(paths[key], np.zeros(1))
        metadata = {
            "schema_version": 1,
            "source_path": os.path.abspath(hdf5_path),
            "dataset_key": DATASET_KEY,
            "source_size": 0,
            "source_mtime_ns": 0,
            "num_events": N_EVENTS,
        }
        _write_event_json(paths["metadata"], metadata)  # type: ignore[arg-type]
        result = _event_cache_is_complete(cache_dir, hdf5_path, DATASET_KEY)
        assert result is None


class TestGTFlowCacheComplete:
    def test_returns_none_when_no_metadata(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path)
        result = _gt_flow_cache_is_complete(str(tmp_path / "cache"), npz_path)
        assert result is None

    def test_returns_none_when_files_missing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path)
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        stat = os.stat(npz_path)
        metadata = {
            "schema_version": 1,
            "source_path": os.path.abspath(npz_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }
        _write_gt_flow_json(os.path.join(cache_dir, "metadata.json"), metadata)  # type: ignore[arg-type]
        result = _gt_flow_cache_is_complete(cache_dir, npz_path)
        assert result is None

    def test_returns_none_when_stale(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path)
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        paths = _make_gt_flow_cache_paths(cache_dir)
        for key in ("x", "y", "timestamp"):
            np.save(paths[key], np.zeros(1))
        metadata = {
            "schema_version": 1,
            "source_path": os.path.abspath(npz_path),
            "source_size": 0,
            "source_mtime_ns": 0,
        }
        _write_gt_flow_json(paths["metadata"], metadata)  # type: ignore[arg-type]
        result = _gt_flow_cache_is_complete(cache_dir, npz_path)
        assert result is None


class TestBuildEventCache:
    def test_build_and_verify(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "event_cache")

        metadata = _build_event_cache(hdf5_path, DATASET_KEY, cache_dir)
        assert metadata["num_events"] == N_EVENTS
        assert metadata["schema_version"] == 1

        paths = _make_event_cache_paths(cache_dir)
        x = np.load(paths["x"])
        assert len(x) == N_EVENTS

        result = _event_cache_is_complete(cache_dir, hdf5_path, DATASET_KEY)
        assert result is not None

    def test_replaces_existing_cache(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "event_cache")

        _build_event_cache(hdf5_path, DATASET_KEY, cache_dir)
        metadata = _build_event_cache(hdf5_path, DATASET_KEY, cache_dir)
        assert metadata["num_events"] == N_EVENTS


class TestBuildGTFlowCache:
    def test_build_and_verify(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=5)
        cache_dir = str(tmp_path / "gt_flow_cache")

        metadata = _build_gt_flow_cache(npz_path, cache_dir)
        assert metadata["schema_version"] == 1

        paths = _make_gt_flow_cache_paths(cache_dir)
        timestamps = np.load(paths["timestamp"])
        assert len(timestamps) == 5

        result = _gt_flow_cache_is_complete(cache_dir, npz_path)
        assert result is not None


class TestBuildGTFlowCacheEdgeCases:
    def test_replaces_existing_cache_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_dir = str(tmp_path / "gt_cache")

        _build_gt_flow_cache(npz_path, cache_dir)
        _build_gt_flow_cache(npz_path, cache_dir)
        paths = _make_gt_flow_cache_paths(cache_dir)
        assert os.path.isfile(paths["metadata"])

    def test_cleanup_on_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_dir = str(tmp_path / "gt_cache_err")

        with mock.patch("evlib.dataloaders._mvsec_storage.np.save", side_effect=IOError("boom")):
            with pytest.raises(IOError):
                _build_gt_flow_cache(npz_path, cache_dir)
        assert not os.path.isdir(cache_dir)


class TestBuildGTFlowCacheTempDirExists:
    def test_removes_pre_existing_temp_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_dir = str(tmp_path / "gt_cache")
        parent_dir = os.path.dirname(cache_dir)
        os.makedirs(parent_dir, exist_ok=True)

        fake_hex = "a" * 32
        temp_dir = os.path.join(parent_dir, f".tmp_{fake_hex}")
        os.makedirs(temp_dir)

        fake_uuid = mock.MagicMock()
        fake_uuid.hex = fake_hex
        with mock.patch("evlib.dataloaders._mvsec_storage.uuid.uuid4", return_value=fake_uuid):
            _build_gt_flow_cache(npz_path, cache_dir)

        assert not os.path.isdir(temp_dir)
        assert os.path.isdir(cache_dir)


class TestBuildEventCacheTempDirExists:
    def test_removes_pre_existing_temp_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "event_cache")
        parent_dir = os.path.dirname(cache_dir)
        os.makedirs(parent_dir, exist_ok=True)

        fake_hex = "b" * 32
        temp_dir = os.path.join(parent_dir, f".tmp_{fake_hex}")
        os.makedirs(temp_dir)

        fake_uuid = mock.MagicMock()
        fake_uuid.hex = fake_hex
        with mock.patch("evlib.dataloaders._event_cache.uuid.uuid4", return_value=fake_uuid):
            _build_event_cache(hdf5_path, DATASET_KEY, cache_dir)

        assert not os.path.isdir(temp_dir)
        assert os.path.isdir(cache_dir)


class TestBuildEventCacheEdgeCases:
    def test_cleanup_on_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_dir = str(tmp_path / "event_cache_err")

        with mock.patch(
            "evlib.dataloaders._event_cache._copy_event_rows_into_columns",
            side_effect=IOError("boom"),
        ):
            with pytest.raises(IOError):
                _build_event_cache(hdf5_path, DATASET_KEY, cache_dir)
        assert not os.path.isdir(cache_dir)


class TestLoadMvsecGTFlow:
    def test_reuses_existing_cache(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_root = str(tmp_path / "cache")

        load_mvsec_gt_flow(npz_path, "seq1", cache_root, LoadingType.CACHED)
        x, y, t = load_mvsec_gt_flow(npz_path, "seq1", cache_root, LoadingType.CACHED)
        assert len(t) == 3

    def test_cached_mode(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_root = str(tmp_path / "cache")

        x, y, t = load_mvsec_gt_flow(npz_path, "seq1", cache_root, LoadingType.CACHED)
        assert x.dtype == np.float32
        assert t.dtype == np.float64
        assert len(t) == 3
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert not isinstance(x, np.memmap)
        assert not isinstance(y, np.memmap)
        assert not isinstance(t, np.memmap)

    def test_lazy_mode(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_root = str(tmp_path / "cache")

        x, y, t = load_mvsec_gt_flow(npz_path, "seq1", cache_root, LoadingType.LAZY)
        assert len(t) == 3
        assert isinstance(x, np.memmap)
        assert isinstance(y, np.memmap)
        assert isinstance(t, np.memmap)

    def test_rejects_non_enum_mode(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_root = str(tmp_path / "cache")

        with pytest.raises(ValueError, match="LoadingType.CACHED or LoadingType.LAZY"):
            load_mvsec_gt_flow(npz_path, "seq1", cache_root, "cached")  # type: ignore[arg-type]

    def test_falls_back_on_os_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        npz_path = str(tmp_path / "flow.npz")
        _make_gt_flow_npz(npz_path, n_frames=3)
        cache_root = str(tmp_path / "cache")

        with mock.patch(
            "evlib.dataloaders._mvsec_storage._make_gt_flow_cache_dir",
            side_effect=OSError("disk full"),
        ):
            x, y, t = load_mvsec_gt_flow(npz_path, "seq1", cache_root, LoadingType.CACHED)
        assert len(t) == 3


class TestCopyEventRows:
    def test_columns_match_rows(self) -> None:
        n = 10
        rows = np.zeros((n, 4), dtype=np.float64)
        rows[:, 0] = np.arange(n)  # x
        rows[:, 1] = np.arange(n) + 10  # y
        rows[:, 2] = np.linspace(0, 1, n)  # timestamp
        rows[:, 3] = np.tile([0.0, 1.0], n // 2)  # polarity

        x = np.empty(n, dtype=np.int16)
        y = np.empty(n, dtype=np.int16)
        ts = np.empty(n, dtype=np.float64)
        pol = np.empty(n, dtype=np.bool_)

        with h5py.File("dummy.hdf5", "w", driver="core", backing_store=False) as f:
            ds = f.create_dataset("events", data=rows)
            _copy_event_rows_into_columns(ds, x, y, ts, pol)

        np.testing.assert_array_equal(x, rows[:, 0].astype(np.int16))
        np.testing.assert_array_equal(y, rows[:, 1].astype(np.int16))
        expected_pol = rows[:, 3] > 0.0
        np.testing.assert_array_equal(pol, expected_pol)


class TestFreezeEventColumns:
    def test_marks_readonly(self) -> None:
        x = np.zeros(5, dtype=np.int16)
        y = np.zeros(5, dtype=np.int16)
        ts = np.zeros(5, dtype=np.float64)
        pol = np.zeros(5, dtype=np.bool_)
        _freeze_event_columns(x, y, ts, pol)
        assert not x.flags.writeable
        assert not y.flags.writeable
        assert not ts.flags.writeable
        assert not pol.flags.writeable


class TestPrepareEventCache:
    def test_creates_and_reuses(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")

        cache_name = "seq1_left"
        _, paths1, meta1 = _prepare_event_cache(hdf5_path, DATASET_KEY, cache_name, cache_root)
        _, paths2, meta2 = _prepare_event_cache(hdf5_path, DATASET_KEY, cache_name, cache_root)
        assert meta1["num_events"] == meta2["num_events"]
        assert paths1 == paths2


class TestLoadMvsecCachedEvents:
    def test_loads_frozen_columns(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")

        cache_name = "seq1_left"
        x, y, ts, pol = load_cached_events(hdf5_path, DATASET_KEY, cache_name, cache_root)
        assert len(x) == N_EVENTS
        assert not x.flags.writeable
        assert not ts.flags.writeable


class TestLazyH5Dataset:
    def test_read_and_close(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        reader = _LazyH5Dataset(hdf5_path, DATASET_KEY, np.float64)

        assert not reader.has_open_handle
        row = reader.read(0)
        assert reader.has_open_handle
        assert row is not None  # type: ignore[unreachable]
        assert not row.flags.writeable

        reader.close()
        assert not reader.has_open_handle
        result = reader.read(0)
        assert result is None

    def test_pickle_drops_handles(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        reader = _LazyH5Dataset(hdf5_path, DATASET_KEY, np.float64)
        reader.read(0)

        state = reader.__getstate__()
        assert state["_file"] is None
        assert state["_dataset"] is None
        assert state["_pid"] is None

    def test_reuses_open_handle(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        reader = _LazyH5Dataset(hdf5_path, DATASET_KEY, np.float64)
        reader.read(0)
        reader.read(1)
        assert reader.has_open_handle


class TestCachedEventBackend:
    def test_from_event_dataset(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            backend = _CachedEventBackend.from_event_dataset(f[DATASET_KEY])

        assert backend.num_events == N_EVENTS
        events = backend.load_events(0, 5)
        assert len(events) == 5

    def test_from_sidecar(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")
        cache_name = "seq1_left"

        backend = _CachedEventBackend.from_sidecar(hdf5_path, DATASET_KEY, cache_name, cache_root)
        assert backend.num_events == N_EVENTS

    def test_time_and_index_methods(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            backend = _CachedEventBackend.from_event_dataset(f[DATASET_KEY])

        idx = backend.time_to_index(0.5)
        t = backend.index_to_time(0)
        assert isinstance(idx, int)
        assert isinstance(t, float)

        indices = backend.times_to_indices(np.array([0.0, 0.5, 1.0]))
        assert indices.dtype == np.int64

        times = backend.indices_to_times(np.array([0, 1, 2]))
        assert times.dtype == np.float64

    def test_close_is_noop(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            backend = _CachedEventBackend.from_event_dataset(f[DATASET_KEY])
        backend.close()
        assert backend.num_events == N_EVENTS


class TestLazyEventBackend:
    def test_load_events(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")
        cache_name = "seq1_left"

        backend = _LazyEventBackend(hdf5_path, DATASET_KEY, cache_name, cache_root)
        assert backend.num_events == N_EVENTS
        events = backend.load_events(0, 5)
        assert len(events) == 5

    def test_time_and_index_methods(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")
        cache_name = "seq1_left"

        backend = _LazyEventBackend(hdf5_path, DATASET_KEY, cache_name, cache_root)
        idx = backend.time_to_index(0.5)
        t = backend.index_to_time(0)
        assert isinstance(idx, int)
        assert isinstance(t, float)

        indices = backend.times_to_indices(np.array([0.0, 0.5]))
        assert indices.dtype == np.int64

        times = backend.indices_to_times(np.array([0, 1]))
        assert times.dtype == np.float64

    def test_close_releases_memmaps(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")
        cache_name = "seq1_left"

        backend = _LazyEventBackend(hdf5_path, DATASET_KEY, cache_name, cache_root)
        backend.load_events(0, 1)
        backend.close()
        assert backend._x is None
        assert backend._timestamp is None

    def test_pickle_drops_memmaps(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_hdf5(hdf5_path)
        cache_root = str(tmp_path / "cache")
        cache_name = "seq1_left"

        backend = _LazyEventBackend(hdf5_path, DATASET_KEY, cache_name, cache_root)
        backend.load_events(0, 1)
        state = backend.__getstate__()
        assert state["_x"] is None
        assert state["_pid"] is None
