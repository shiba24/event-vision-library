"""MVSEC storage helpers and GT flow sidecar cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import cast

import numpy as np
import numpy.typing as npt

from ._storage_common import ResidentLoadMode


_GT_FLOW_CACHE_SCHEMA_VERSION = 1


class _GTFlowCacheMetadata(TypedDict):
    schema_version: int
    source_path: str
    source_size: int
    source_mtime_ns: int


def resolve_mvsec_cache_dir(cache_dir: Optional[str]) -> str:
    """MVSEC cache root directory."""
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

    cache_dir_path = os.path.join(cache_base, "evlib", "mvsec")
    return os.path.abspath(cache_dir_path)


def _make_gt_flow_cache_dir(
    cache_root: str,
    source_path: str,
    sequence: str,
) -> str:
    stat_result = os.stat(source_path)
    parts = [
        os.path.abspath(source_path),
        str(int(stat_result.st_size)),
        str(int(stat_result.st_mtime_ns)),
        str(_GT_FLOW_CACHE_SCHEMA_VERSION),
    ]
    joined = "|".join(parts)
    signature = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    directory_name = f"{sequence}_{signature[:16]}"
    return os.path.join(cache_root, "gt_flow_sidecars", directory_name)


def _make_gt_flow_cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "x": os.path.join(cache_dir, "x_flow.npy"),
        "y": os.path.join(cache_dir, "y_flow.npy"),
        "timestamp": os.path.join(cache_dir, "timestamps.npy"),
        "metadata": os.path.join(cache_dir, "metadata.json"),
    }


def _write_json(path: str, data: _GTFlowCacheMetadata) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def _load_gt_flow_cache_metadata(metadata_path: str) -> Optional[_GTFlowCacheMetadata]:
    if not os.path.isfile(metadata_path):
        return None

    with open(metadata_path, "r", encoding="utf-8") as file_handle:
        metadata = cast(_GTFlowCacheMetadata, json.load(file_handle))
    return metadata


def _gt_flow_cache_is_complete(
    cache_dir: str,
    source_path: str,
) -> Optional[_GTFlowCacheMetadata]:
    paths = _make_gt_flow_cache_paths(cache_dir)
    metadata = _load_gt_flow_cache_metadata(paths["metadata"])
    if metadata is None:
        return None

    required_keys = ("x", "y", "timestamp")
    if not all(os.path.isfile(paths[key]) for key in required_keys):
        return None

    stat_result = os.stat(source_path)
    source_size = int(stat_result.st_size)
    source_mtime_ns = int(stat_result.st_mtime_ns)
    source_path_abs = os.path.abspath(source_path)
    is_stale = (
        metadata["schema_version"] != _GT_FLOW_CACHE_SCHEMA_VERSION
        or metadata["source_path"] != source_path_abs
        or metadata["source_size"] != source_size
        or metadata["source_mtime_ns"] != source_mtime_ns
    )
    if is_stale:
        return None

    return metadata


def _build_gt_flow_cache(
    source_path: str,
    cache_dir: str,
) -> _GTFlowCacheMetadata:
    parent_dir = os.path.dirname(cache_dir)
    os.makedirs(parent_dir, exist_ok=True)

    temp_dir = os.path.join(parent_dir, f".tmp_{uuid.uuid4().hex}")
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    paths = _make_gt_flow_cache_paths(temp_dir)

    try:
        with np.load(source_path) as npz_file:
            x_flow = np.asarray(npz_file["x_flow_dist"], dtype=np.float32)
            y_flow = np.asarray(npz_file["y_flow_dist"], dtype=np.float32)
            timestamps = np.asarray(npz_file["timestamps"], dtype=np.float64)

        np.save(paths["x"], x_flow)
        np.save(paths["y"], y_flow)
        np.save(paths["timestamp"], timestamps)

        stat_result = os.stat(source_path)
        metadata: _GTFlowCacheMetadata = {
            "schema_version": _GT_FLOW_CACHE_SCHEMA_VERSION,
            "source_path": os.path.abspath(source_path),
            "source_size": int(stat_result.st_size),
            "source_mtime_ns": int(stat_result.st_mtime_ns),
        }
        _write_json(paths["metadata"], metadata)

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.replace(temp_dir, cache_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _load_gt_flow_from_npz_file(
    source_path: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    with np.load(source_path) as npz_file:
        x_flow = np.asarray(npz_file["x_flow_dist"], dtype=np.float32)
        y_flow = np.asarray(npz_file["y_flow_dist"], dtype=np.float32)
        timestamps = np.asarray(npz_file["timestamps"], dtype=np.float64)
    return x_flow, y_flow, timestamps


def load_mvsec_gt_flow(
    source_path: str,
    sequence: str,
    cache_root: str,
    load_mode: ResidentLoadMode,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Load MVSEC GT flow arrays, build a decompressed sidecar if needed."""
    try:
        cache_dir = _make_gt_flow_cache_dir(cache_root, source_path, sequence)
        metadata = _gt_flow_cache_is_complete(cache_dir, source_path)
        if metadata is None:
            metadata = _build_gt_flow_cache(source_path, cache_dir)

        paths = _make_gt_flow_cache_paths(cache_dir)
        mmap_mode: Optional[Literal["r"]]
        if load_mode == "lazy":
            mmap_mode = "r"
        else:
            mmap_mode = None

        x_flow = np.load(paths["x"], mmap_mode=mmap_mode)
        y_flow = np.load(paths["y"], mmap_mode=mmap_mode)
        timestamps = np.load(paths["timestamp"], mmap_mode=mmap_mode)

        if load_mode == "cached":
            x_flow = np.asarray(x_flow, dtype=np.float32)
            y_flow = np.asarray(y_flow, dtype=np.float32)
            timestamps = np.asarray(timestamps, dtype=np.float64)

        return x_flow, y_flow, timestamps
    except OSError:
        return _load_gt_flow_from_npz_file(source_path)
