"""Low level I/O dataloader for the MVSEC dataset.

Extracts all file access (HDF5, NPZ, calibration) from the dataset layer
so can be use for flexible access patterns

Supports all MVSEC data modalities (events, images, IMU, depth, poses,
odometry, optical flow, velodyne lidar) with per modality lazy/cached
loading control.

Reference: https://daniilidis-group.github.io/mvsec/

Zhu, A. Z., Thakur, D., Ozaslan, T., Pfrommer, B., Kumar, V., & Daniilidis, K. (2018).
The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception.
IEEE Robotics and Automation Letters, 3(3), 2032-2039.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import warnings
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from evlib.codec.fileformat.hdf5 import open_hdf5
from evlib.types import RawEvents

from ._base import DataLoaderBase
from ._event_cache import _CachedEventBackend
from ._event_cache import _EventBackend
from ._event_cache import _LazyEventBackend
from ._mvsec_storage import load_mvsec_gt_flow
from ._mvsec_storage import resolve_mvsec_cache_dir
from ._mvsec_types import MVSECOdometryData
from ._storage_common import ResidentLoadMode
from ._storage_common import _LazyH5Dataset
from ._storage_common import normalize_resident_load_mode
from .utils import find_nearest_index
from .utils import get_flow_coordinate_grid
from .utils import propagate_flow_step


logger = logging.getLogger(__name__)

# False = off, True/"lazy" = on demand, "cached" = in memory
LoadMode = Union[bool, Literal["lazy", "cached"]]


# Module lvl helpers


def _resolve_load_mode(value: LoadMode) -> Tuple[bool, bool]:
    """Parse a LoadMode into (should_load, should_cache)."""
    if value is False:
        return (False, False)
    if value is True or value == "lazy":
        return (True, False)
    if value == "cached":
        return (True, True)
    raise ValueError(f"Invalid load mode: {value!r}. Expected False, True, 'lazy', or 'cached'.")


def _resolve_depth_load_modes(
    load_gt_depth_raw: LoadMode,
    load_gt_depth_rect: LoadMode,
) -> Tuple[Tuple[bool, bool], Tuple[bool, bool]]:
    return _resolve_load_mode(load_gt_depth_raw), _resolve_load_mode(load_gt_depth_rect)


def _load_calibration(
    root: str, category: str, camera: str
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Load calibration rectification maps from text files."""
    name_x = f"{category}_{camera}_x_map.txt"
    name_y = f"{category}_{camera}_y_map.txt"
    for d in [root, os.path.join(root, f"{category}_calib")]:
        xp = os.path.join(d, name_x)
        yp = os.path.join(d, name_y)
        if os.path.isfile(xp) and os.path.isfile(yp):
            return np.loadtxt(xp), np.loadtxt(yp)
    return None


def _load_odometry_npz(npz_path: str) -> Optional[MVSECOdometryData]:
    npz = np.load(npz_path)
    return MVSECOdometryData(
        timestamps=np.asarray(npz["timestamps"], dtype=np.float64),
        linear_velocity=np.asarray(npz["lin_vel"], dtype=np.float64),
        position=np.asarray(npz["pos"], dtype=np.float64),
        quaternion=np.asarray(npz["quat"], dtype=np.float64),
        angular_velocity=np.asarray(npz["ang_vel"], dtype=np.float64),
    )


def _estimate_interval_flow(
    x_flow_all: npt.NDArray[np.float32],
    y_flow_all: npt.NDArray[np.float32],
    gt_timestamps: npt.NDArray[np.float64],
    t_start: float,
    t_end: float,
) -> npt.NDArray[np.float32]:
    """Estimate pixel displacement over [t_start, t_end) from GT flow frames.

    GT flow is given at discrete timestamps. For arbitrary intervals
    either scale a single frame (fast path) or chain propagate across
    multiple frames tracking where each pixel ends up.
    """
    if t_end <= t_start:
        raise ValueError(f"Expected t_end > t_start, got [{t_start}, {t_end}).")
    if len(gt_timestamps) < 2:
        raise ValueError("At least two GT flow timestamps are required.")
    if t_start < float(gt_timestamps[0]) or t_end > float(gt_timestamps[-1]):
        raise ValueError(
            f"Interval [{t_start}, {t_end}) falls outside GT support "
            f"[{gt_timestamps[0]}, {gt_timestamps[-1]}]."
        )

    last_flow_idx = len(gt_timestamps) - 2
    start_idx = int(np.searchsorted(gt_timestamps, t_start, side="right")) - 1
    if start_idx < 0 or start_idx > last_flow_idx:
        raise ValueError(
            f"Could not resolve a GT flow interval for start time {t_start}. "
            f"GT timestamps range: [{gt_timestamps[0]}, {gt_timestamps[-1]}]"
        )

    gt_t0 = float(gt_timestamps[start_idx])
    gt_t1 = float(gt_timestamps[start_idx + 1])
    gt_dt = gt_t1 - gt_t0
    req_dt = t_end - t_start
    remaining = gt_t1 - t_start

    # fast path - requested window fits inside one GT interval
    if gt_dt >= req_dt and remaining >= req_dt:
        scale = np.float32(req_dt / gt_dt)
        fx = x_flow_all[start_idx].copy() * scale
        fy = y_flow_all[start_idx].copy() * scale
        h, w = fx.shape
        out = np.empty((h, w, 2), dtype=np.float32)
        out[..., 0] = fx
        out[..., 1] = fy
        return out

    # slow path - walk pixel coords through multiple GT intervals
    # leading partial -> full middle intervals -> trailing partial
    h, w = x_flow_all.shape[1], x_flow_all.shape[2]
    origin_x, origin_y = get_flow_coordinate_grid(h, w)
    xc = origin_x.copy()
    yc = origin_y.copy()
    xm = np.ones((h, w), dtype=np.bool_)
    ym = np.ones((h, w), dtype=np.bool_)

    # leading partial
    propagate_flow_step(
        x_flow_all[start_idx],
        y_flow_all[start_idx],
        xc,
        yc,
        xm,
        ym,
        remaining / gt_dt,
    )

    # full middle intervals
    i = start_idx + 1
    while i < last_flow_idx:
        if float(gt_timestamps[i + 1]) >= t_end:
            break
        propagate_flow_step(
            x_flow_all[i],
            y_flow_all[i],
            xc,
            yc,
            xm,
            ym,
            1.0,
        )
        i += 1

    # trailing partial
    trail_t0 = float(gt_timestamps[i])
    trail_t1 = float(gt_timestamps[i + 1])
    trail_scale = (t_end - trail_t0) / (trail_t1 - trail_t0)
    propagate_flow_step(
        x_flow_all[i],
        y_flow_all[i],
        xc,
        yc,
        xm,
        ym,
        trail_scale,
    )

    # displacement = where pixels ended up minus where they started
    dx = xc - origin_x
    dy = yc - origin_y
    dx[~xm] = 0.0
    dy[~ym] = 0.0

    out = np.empty((h, w, 2), dtype=np.float32)
    out[..., 0] = dx
    out[..., 1] = dy
    return out


def _freeze_array(arr: "Optional[npt.NDArray[np.generic]]") -> None:
    if arr is not None:
        arr.flags.writeable = False


# MVSECDataLoader


class MVSECDataLoader(DataLoaderBase):
    """Low level I/O for a single MVSEC sequence.

    Handles HDF5 reading, NPZ decompression, calibration loading, and
    typed column caching.  Use directly for custom access patterns
    (overlapping windows, multiscale, etc.) or let MVSECDataset wrap it
    for PyTorch style frame indexed sampling.

    Large HDF5 backed modalities (depth, blended images, HDF5 flow, and
    velodyne) support a per modality loading strategy via the ``LoadMode``
    tristate:

    - ``False`` -- do not load this modality.
    - ``True`` or ``"lazy"`` -- lazy load on demand (HDF5 handle stays open).
    - ``"cached"`` -- read entirely into memory during ``__init__``.

    Small modalities (IMU, odometry NPZ, GT odometry, GT poses) are always
    cached in memory when enabled because they are tiny.

    The default profile is intentionally lean:

    - events are cached in RAM
    - grayscale images are cached in RAM
    - optional GT and calibration modalities stay disabled until requested

    Args:
        root: Directory containing the MVSEC dataset files.
        sequence: Sequence name (e.g. ``"indoor_flying1"``).
        camera: ``"left"`` or ``"right"``.
        load_gt_flow: LoadMode for GT optical flow from NPZ.
        load_calibration: If True, load calibration rectification maps.
        load_imu: If True, cache IMU data from data HDF5.
        load_odometry_npz: If True, cache odometry from ``*_odom.npz``.
        load_gt_odometry: If True, cache LOAM odometry SE(3) from gt HDF5.
        load_gt_poses: If True, cache Cartographer poses SE(3) from gt HDF5.
        load_gt_depth_raw: LoadMode for raw depth maps from gt HDF5.
        load_gt_depth_rect: LoadMode for rectified depth maps from gt HDF5.
        load_gt_flow_hdf5: LoadMode for optical flow from gt HDF5.
        load_gt_blended: LoadMode for blended images from gt HDF5.
        load_velodyne: LoadMode for velodyne lidar from data HDF5.
        event_load_mode: ``"cached"`` to keep all events in RAM or ``"lazy"``
            to build and use a typed read only sidecar cache.
        image_load_mode: ``"cached"`` to keep all images in RAM or ``"lazy"``
            to read frames from HDF5 on demand.
        cache_dir: Optional root directory for MVSEC sidecar caches.
    """

    IMAGE_SHAPE: Tuple[int, int] = (260, 346)  # (h, w)

    # these are GT flow specific , may likely be better to move to a config in future
    VALID_FRAMES: Dict[str, Tuple[int, Optional[int]]] = {
        "indoor_flying1": (60, 1340),
        "indoor_flying2": (140, 1500),
        "indoor_flying3": (100, 1711),
        "indoor_flying4": (104, 380),
        "outdoor_day1": (0, 5020),
        "outdoor_day2": (30, None),
    }

    def __init__(
        self,
        root: str,
        sequence: str,
        camera: str = "left",
        load_gt_flow: LoadMode = False,
        load_calibration: bool = False,
        load_imu: bool = False,
        load_odometry_npz: bool = False,
        load_gt_odometry: bool = False,
        load_gt_poses: bool = False,
        load_gt_depth_raw: LoadMode = False,
        load_gt_depth_rect: LoadMode = False,
        load_gt_flow_hdf5: LoadMode = False,
        load_gt_blended: LoadMode = False,
        load_velodyne: LoadMode = False,
        event_load_mode: ResidentLoadMode = "cached",
        image_load_mode: ResidentLoadMode = "cached",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.root = root
        self.sequence = sequence
        if camera not in ("left", "right"):
            raise ValueError(f"camera must be 'left' or 'right', got '{camera}'")
        self.camera = camera
        has_explicit_cache = cache_dir is not None
        self._event_load_mode = normalize_resident_load_mode("event_load_mode", event_load_mode)
        self._image_load_mode = normalize_resident_load_mode("image_load_mode", image_load_mode)
        self._cache_dir = resolve_mvsec_cache_dir(cache_dir)

        raw_depth_modes, rect_depth_modes = _resolve_depth_load_modes(
            load_gt_depth_raw,
            load_gt_depth_rect,
        )
        do_load_gt_flow, do_cache_gt_flow = _resolve_load_mode(load_gt_flow)
        do_load_depth_raw, do_cache_depth_raw = raw_depth_modes
        do_load_depth_rect, do_cache_depth_rect = rect_depth_modes
        do_load_flow_h5, do_cache_flow_h5 = _resolve_load_mode(load_gt_flow_hdf5)
        do_load_blended, do_cache_blended = _resolve_load_mode(load_gt_blended)
        do_load_velodyne, do_cache_velodyne = _resolve_load_mode(load_velodyne)

        hdf5_path = os.path.join(root, f"{sequence}_data.hdf5")
        self._data_hdf5_path = hdf5_path
        davis_key = f"davis/{camera}"
        self._davis_key = davis_key
        event_cache_name = f"{sequence}_{camera}"

        npz_path = os.path.join(root, f"{sequence}_gt_flow_dist.npz")
        category = sequence.rstrip("0123456789")

        has_gt_npz = do_load_gt_flow and os.path.isfile(npz_path)
        odom_npz_path = os.path.join(root, f"{sequence}_odom.npz")
        has_odom_npz = load_odometry_npz and os.path.isfile(odom_npz_path)

        # calibration and odometry are independent file reads so kick
        # them off in threads while the main thread handles HDF5
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        if load_calibration or has_odom_npz:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        calib_future = (
            executor.submit(_load_calibration, root, category, camera)
            if executor and load_calibration
            else None
        )
        odom_future = (
            executor.submit(_load_odometry_npz, odom_npz_path)
            if executor and has_odom_npz
            else None
        )

        self._images: Optional[npt.NDArray[np.uint8]] = None
        self._images_lazy: Optional[_LazyH5Dataset] = None
        self._has_images = False
        self._frame_ts: Optional[npt.NDArray[np.float64]] = None
        self._frame_event_inds: Optional[npt.NDArray[np.int64]] = None
        self._imu: Optional[npt.NDArray[np.float64]] = None
        self._imu_ts: Optional[npt.NDArray[np.float64]] = None
        self._velodyne_cached: Optional[npt.NDArray[np.float32]] = None
        self._velodyne_lazy: Optional[_LazyH5Dataset] = None
        self._velodyne_ts: Optional[npt.NDArray[np.float64]] = None

        with open_hdf5(hdf5_path) as data_h5:
            davis = data_h5[davis_key]

            # pick event backend - cached = all in RAM, lazy = sidecar mmap
            self._event_backend: _EventBackend
            if self._event_load_mode == "cached":
                if has_explicit_cache:
                    # try sidecar first, fallback direct HDF5 if broken
                    try:
                        self._event_backend = _CachedEventBackend.from_sidecar(
                            source_path=hdf5_path,
                            dataset_key=f"{davis_key}/events",
                            cache_name=event_cache_name,
                            cache_root=self._cache_dir,
                        )
                    except OSError:
                        self._event_backend = _CachedEventBackend.from_event_dataset(
                            davis["events"]
                        )
                else:
                    self._event_backend = _CachedEventBackend.from_event_dataset(davis["events"])
            else:
                self._event_backend = _LazyEventBackend(
                    source_path=hdf5_path,
                    dataset_key=f"{davis_key}/events",
                    cache_name=event_cache_name,
                    cache_root=self._cache_dir,
                )

            if "image_raw_ts" in davis:
                self._frame_ts = np.array(davis["image_raw_ts"], dtype=np.float64)

            if "image_raw_event_inds" in davis:
                self._frame_event_inds = np.array(davis["image_raw_event_inds"], dtype=np.int64)

            self._has_images = "image_raw" in davis
            if self._has_images:
                if self._image_load_mode == "cached":
                    self._images = np.array(davis["image_raw"], dtype=np.uint8)
                else:
                    self._images_lazy = _LazyH5Dataset(
                        hdf5_path,
                        f"{davis_key}/image_raw",
                        np.uint8,
                    )

            if load_imu:
                if "imu" in davis:
                    self._imu = np.array(davis["imu"], dtype=np.float64)
                    self._imu_ts = np.array(davis["imu_ts"], dtype=np.float64)
                else:
                    logger.warning("IMU data not found in %s", hdf5_path)

            if do_load_velodyne:
                if "velodyne" in data_h5 and "scans" in data_h5["velodyne"]:
                    vel = data_h5["velodyne"]
                    self._velodyne_ts = np.array(vel["scans_ts"], dtype=np.float64)
                    if do_cache_velodyne:
                        self._velodyne_cached = np.array(vel["scans"], dtype=np.float32)
                    else:
                        self._velodyne_lazy = _LazyH5Dataset(
                            hdf5_path,
                            "velodyne/scans",
                            np.float32,
                        )
                else:
                    logger.warning("Velodyne data not found in %s", hdf5_path)

        gt_hdf5_path = os.path.join(root, f"{sequence}_gt.hdf5")
        self._gt_hdf5_path = gt_hdf5_path

        self._gt_odometry: Optional[npt.NDArray[np.float64]] = None
        self._gt_odometry_ts: Optional[npt.NDArray[np.float64]] = None
        self._gt_pose: Optional[npt.NDArray[np.float64]] = None
        self._gt_pose_ts: Optional[npt.NDArray[np.float64]] = None
        self._depth_raw_cached: Optional[npt.NDArray[np.float32]] = None
        self._depth_raw_lazy: Optional[_LazyH5Dataset] = None
        self._depth_raw_ts: Optional[npt.NDArray[np.float64]] = None
        self._depth_rect_cached: Optional[npt.NDArray[np.float32]] = None
        self._depth_rect_lazy: Optional[_LazyH5Dataset] = None
        self._depth_rect_ts: Optional[npt.NDArray[np.float64]] = None
        self._blended_cached: Optional[npt.NDArray[np.uint8]] = None
        self._blended_lazy: Optional[_LazyH5Dataset] = None
        self._blended_ts: Optional[npt.NDArray[np.float64]] = None
        self._flow_h5_cached: Optional[npt.NDArray[np.float64]] = None
        self._flow_h5_lazy: Optional[_LazyH5Dataset] = None
        self._flow_h5_ts: Optional[npt.NDArray[np.float64]] = None

        needs_gt_h5 = (
            do_load_depth_raw
            or do_load_depth_rect
            or load_gt_odometry
            or load_gt_poses
            or do_load_flow_h5
            or do_load_blended
        )
        if needs_gt_h5 and os.path.isfile(gt_hdf5_path):
            with open_hdf5(gt_hdf5_path) as gt_h5:
                gt_key = f"davis/{camera}"
                gt_group = gt_h5[gt_key] if gt_key in gt_h5 else None

                if gt_group is None:
                    logger.warning("Camera '%s' not found in %s", camera, gt_hdf5_path)
                else:
                    self._load_gt_small_data(gt_group, load_gt_odometry, load_gt_poses)
                    self._load_gt_large_data(
                        gt_group,
                        gt_hdf5_path,
                        gt_key,
                        do_load_depth_raw,
                        do_cache_depth_raw,
                        do_load_depth_rect,
                        do_cache_depth_rect,
                        do_load_flow_h5,
                        do_cache_flow_h5,
                        do_load_blended,
                        do_cache_blended,
                    )
        elif needs_gt_h5:
            logger.warning("GT HDF5 file not found: %s", gt_hdf5_path)

        self._gt_x_flow: Optional[npt.NDArray[np.float32]] = None
        self._gt_y_flow: Optional[npt.NDArray[np.float32]] = None
        self._gt_ts: Optional[npt.NDArray[np.float64]] = None
        if has_gt_npz:
            mode: ResidentLoadMode = "cached" if do_cache_gt_flow else "lazy"
            self._gt_x_flow, self._gt_y_flow, self._gt_ts = load_mvsec_gt_flow(
                npz_path,
                sequence,
                self._cache_dir,
                mode,
            )
        elif do_load_gt_flow:
            logger.warning("GT flow file not found: %s", npz_path)

        self._x_map: Optional[npt.NDArray[np.float64]] = None
        self._y_map: Optional[npt.NDArray[np.float64]] = None
        if calib_future is not None:
            calib_result = calib_future.result()
            if calib_result is not None:
                self._x_map, self._y_map = calib_result
            else:
                dirs = [root, os.path.join(root, f"{category}_calib")]
                logger.warning("Calibration map(s) not found for %s in %s", camera, dirs)

        self._odometry_npz: Optional[MVSECOdometryData] = None
        if odom_future is not None:
            odom_result = odom_future.result()
            if odom_result is not None:
                self._odometry_npz = odom_result
        elif load_odometry_npz:
            logger.warning("Odometry NPZ file not found: %s", odom_npz_path)

        if executor is not None:
            executor.shutdown(wait=False)

        # freeze everything we loaded into memory
        for arr in [
            self._frame_ts,
            self._frame_event_inds,
            self._images,
            self._gt_x_flow,
            self._gt_y_flow,
            self._gt_ts,
            self._x_map,
            self._y_map,
            self._imu,
            self._imu_ts,
            self._velodyne_cached,
            self._velodyne_ts,
            self._gt_odometry,
            self._gt_odometry_ts,
            self._gt_pose,
            self._gt_pose_ts,
            self._depth_raw_cached,
            self._depth_raw_ts,
            self._depth_rect_cached,
            self._depth_rect_ts,
            self._blended_cached,
            self._blended_ts,
            self._flow_h5_cached,
            self._flow_h5_ts,
        ]:
            _freeze_array(arr)

        if self._odometry_npz is not None:
            for arr in [
                self._odometry_npz.timestamps,
                self._odometry_npz.linear_velocity,
                self._odometry_npz.position,
                self._odometry_npz.quaternion,
                self._odometry_npz.angular_velocity,
            ]:
                _freeze_array(arr)

    # Private init helpers

    def _load_gt_small_data(
        self,
        gt_group: h5py.Group,
        load_gt_odometry: bool,
        load_gt_poses: bool,
    ) -> None:
        if load_gt_odometry:
            if "odometry" in gt_group:
                self._gt_odometry = np.array(gt_group["odometry"], dtype=np.float64)
                self._gt_odometry_ts = np.array(gt_group["odometry_ts"], dtype=np.float64)
            else:
                logger.warning("GT odometry not found in gt HDF5.")

        if load_gt_poses:
            if "pose" in gt_group:
                self._gt_pose = np.array(gt_group["pose"], dtype=np.float64)
                self._gt_pose_ts = np.array(gt_group["pose_ts"], dtype=np.float64)
            else:
                logger.warning("GT pose not found in gt HDF5.")

    def _load_gt_large_data(
        self,
        gt_group: h5py.Group,
        gt_hdf5_path: str,
        gt_davis_key: str,
        do_load_depth_raw: bool,
        do_cache_depth_raw: bool,
        do_load_depth_rect: bool,
        do_cache_depth_rect: bool,
        do_load_flow_h5: bool,
        do_cache_flow_h5: bool,
        do_load_blended: bool,
        do_cache_blended: bool,
    ) -> None:
        """Load or set up lazy refs for large GT datasets."""
        if do_load_depth_raw:
            if "depth_image_raw" in gt_group:
                self._depth_raw_ts = np.array(gt_group["depth_image_raw_ts"], dtype=np.float64)
                if do_cache_depth_raw:
                    self._depth_raw_cached = np.array(gt_group["depth_image_raw"], dtype=np.float32)
                else:
                    self._depth_raw_lazy = _LazyH5Dataset(
                        gt_hdf5_path,
                        f"{gt_davis_key}/depth_image_raw",
                        np.float32,
                    )
            else:
                logger.warning("depth_image_raw not found in gt HDF5.")

        if do_load_depth_rect:
            if "depth_image_rect" in gt_group:
                self._depth_rect_ts = np.array(gt_group["depth_image_rect_ts"], dtype=np.float64)
                if do_cache_depth_rect:
                    self._depth_rect_cached = np.array(
                        gt_group["depth_image_rect"],
                        dtype=np.float32,
                    )
                else:
                    self._depth_rect_lazy = _LazyH5Dataset(
                        gt_hdf5_path,
                        f"{gt_davis_key}/depth_image_rect",
                        np.float32,
                    )
            else:
                logger.warning("depth_image_rect not found in gt HDF5.")

        if do_load_flow_h5:
            if "flow_dist" in gt_group:
                self._flow_h5_ts = np.array(gt_group["flow_dist_ts"], dtype=np.float64)
                if do_cache_flow_h5:
                    self._flow_h5_cached = np.array(gt_group["flow_dist"], dtype=np.float64)
                else:
                    self._flow_h5_lazy = _LazyH5Dataset(
                        gt_hdf5_path,
                        f"{gt_davis_key}/flow_dist",
                        np.float64,
                    )
            else:
                logger.warning("flow_dist not found in gt HDF5.")

        if do_load_blended:
            if "blended_image_rect" in gt_group:
                self._blended_ts = np.array(gt_group["blended_image_rect_ts"], dtype=np.float64)
                if do_cache_blended:
                    self._blended_cached = np.array(
                        gt_group["blended_image_rect"],
                        dtype=np.uint8,
                    )
                else:
                    self._blended_lazy = _LazyH5Dataset(
                        gt_hdf5_path,
                        f"{gt_davis_key}/blended_image_rect",
                        np.uint8,
                    )
            else:
                logger.warning("blended_image_rect not found in gt HDF5.")

    # DataLoaderBase interface

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in [start_index, end_index). Returns mutable copies."""
        return self._event_backend.load_events(start_index, end_index)

    @property
    def num_events(self) -> int:
        return self._event_backend.num_events

    def time_to_index(self, t: float) -> int:
        return self._event_backend.time_to_index(t)

    def index_to_time(self, index: int) -> float:
        return self._event_backend.index_to_time(index)

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return self._event_backend.times_to_indices(timestamps)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._event_backend.indices_to_times(indices)

    def close(self) -> None:
        self._event_backend.close()
        for reader in [
            self._images_lazy,
            self._velodyne_lazy,
            self._depth_raw_lazy,
            self._depth_rect_lazy,
            self._flow_h5_lazy,
            self._blended_lazy,
        ]:
            if reader is not None:
                reader.close()

    def __del__(self) -> None:
        lazy_readers = [
            getattr(self, attr, None)
            for attr in (
                "_images_lazy",
                "_velodyne_lazy",
                "_depth_raw_lazy",
                "_depth_rect_lazy",
                "_flow_h5_lazy",
                "_blended_lazy",
            )
        ]
        if any(r is not None and r.has_open_handle for r in lazy_readers):
            warnings.warn(
                f"MVSECDataLoader for '{self.sequence}' was not closed. "
                "Call .close() or use a context manager to release HDF5 handles.",
                ResourceWarning,
                stacklevel=2,
            )
            self.close()

    # GT optical flow (NPZ)

    @property
    def has_gt_flow(self) -> bool:
        return self._gt_x_flow is not None

    def load_optical_flow(self, t1: float, t2: float) -> npt.NDArray[np.float32]:
        """Load GT optical flow between two timestamps.

        Returns (H, W, 2) float32 where channels are [flow_x, flow_y].
        Estimates displacement over [t1, t2) by propagating through GT frames.
        """
        if self._gt_x_flow is None or self._gt_y_flow is None or self._gt_ts is None:
            raise RuntimeError("Ground truth flow not loaded.")

        return _estimate_interval_flow(
            self._gt_x_flow,
            self._gt_y_flow,
            self._gt_ts,
            t1,
            t2,
        )

    def get_gt_timestamps(self, event_index: int) -> Tuple[Optional[float], Optional[float]]:
        """Return (t_before, t_after) GT timestamps bracketing event_index."""
        if self._gt_ts is None:
            return (None, None)

        event_t = self.index_to_time(event_index)
        ip = int(np.searchsorted(self._gt_ts, event_t, side="right"))

        t_before = float(self._gt_ts[ip - 1]) if ip > 0 else None
        t_after = float(self._gt_ts[ip]) if ip < len(self._gt_ts) else None

        return (t_before, t_after)

    def gt_time_list(self) -> npt.NDArray[np.float64]:
        """Return all GT timestamps. Raises if GT flow not loaded."""
        if self._gt_ts is None:
            raise RuntimeError("Ground truth flow not loaded.")
        return self._gt_ts

    # Frames / images

    @property
    def frame_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._frame_ts

    @property
    def frame_event_indices(self) -> Optional[npt.NDArray[np.int64]]:
        return self._frame_event_inds

    @property
    def event_load_mode(self) -> ResidentLoadMode:
        return self._event_load_mode

    @property
    def image_load_mode(self) -> ResidentLoadMode:
        return self._image_load_mode

    @property
    def has_images(self) -> bool:
        return self._has_images

    @property
    def images(self) -> Optional[npt.NDArray[np.uint8]]:
        """Precached images, or None if lazy/unavailable."""
        return self._images

    @property
    def num_frames(self) -> int:
        return len(self._frame_ts) if self._frame_ts is not None else 0

    def normalize_frame_index(self, frame_index: int) -> int:
        """Normalize negative index and validate bounds."""
        n = self.num_frames
        idx = frame_index
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Frame index {frame_index} out of range for dataset with {n} frames")
        return idx

    def find_nearest_frame_index(self, t: float) -> int:
        if self._frame_ts is None or len(self._frame_ts) == 0:
            raise RuntimeError("Frame timestamps are not available for this sequence.")
        return find_nearest_index(self._frame_ts, t)

    def load_frame_sample(self, frame_index: int) -> dict[str, object]:
        """Load a synchronized single camera MVSEC sample for one frame."""
        idx = self.normalize_frame_index(frame_index)
        if self._frame_ts is None:
            raise RuntimeError("Frame timestamps are not available for this sequence.")

        frame_t = float(self._frame_ts[idx])
        if idx == 0:
            start_t = float(self.index_to_time(0))
        else:
            start_t = float(self._frame_ts[idx - 1])

        # use precomputed event indices when available
        if self._frame_event_inds is not None:
            ev_end = int(self._frame_event_inds[idx])
            ev_start = 0 if idx == 0 else int(self._frame_event_inds[idx - 1])
            events = self.load_events(ev_start, ev_end)
        else:
            events = self.get_events_by_time(start_t, frame_t)

        image = self.load_image(idx)

        flow = None
        if self.has_gt_flow:
            try:
                flow = self.load_optical_flow(start_t, frame_t)
            except ValueError:
                flow = None

        imu = None
        if self.has_imu:
            imu = self.load_imu(start_t, frame_t)

        depth = None
        if self.has_gt_depth_raw and self._depth_raw_ts is not None and len(self._depth_raw_ts) > 0:
            depth = self.load_depth_raw(find_nearest_index(self._depth_raw_ts, frame_t))

        depth_rect = None
        if (
            self.has_gt_depth_rect
            and self._depth_rect_ts is not None
            and len(self._depth_rect_ts) > 0
        ):
            depth_rect = self.load_depth_rect(find_nearest_index(self._depth_rect_ts, frame_t))

        blended = None
        if self.has_gt_blended and self._blended_ts is not None and len(self._blended_ts) > 0:
            blended = self.load_blended_image(find_nearest_index(self._blended_ts, frame_t))

        velodyne = None
        if self.has_velodyne and self._velodyne_ts is not None and len(self._velodyne_ts) > 0:
            velodyne = self.load_velodyne_scan(find_nearest_index(self._velodyne_ts, frame_t))

        pose = None
        if self.has_gt_pose:
            pose = self.load_nearest_pose(frame_t, source="pose")

        return {
            "events": events,
            "timestamp": frame_t,
            "image": image,
            "flow": flow,
            "imu": imu,
            "depth": depth,
            "depth_rect": depth_rect,
            "blended": blended,
            "velodyne": velodyne,
            "pose": pose,
        }

    def load_image(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """Load a single grayscale frame by index."""
        if self._images is not None:
            return cast(npt.NDArray[np.uint8], self._images[frame_index])
        if self._images_lazy is not None:
            raw = self._images_lazy.read(frame_index)
            if raw is None:
                return None
            arr = np.asarray(raw, dtype=np.uint8)
            arr.flags.writeable = False
            return arr
        return None

    # Calibration

    @property
    def valid_frame_range(self) -> Optional[Tuple[int, Optional[int]]]:
        return self.VALID_FRAMES.get(self.sequence)

    @property
    def has_calibration(self) -> bool:
        return self._x_map is not None and self._y_map is not None

    def undistort_events(self, events: RawEvents) -> RawEvents:
        """Apply calibration rectification maps to events.

        Returns new RawEvents with rectified coordinates.
        Raises RuntimeError if calibration maps are not loaded.
        """
        if self._x_map is None or self._y_map is None:
            raise RuntimeError("Calibration maps not loaded.")

        h, w = self.IMAGE_SHAPE
        cy = np.clip(events.y, 0, h - 1)
        cx = np.clip(events.x, 0, w - 1)

        return RawEvents(
            x=np.round(self._x_map[cy, cx]).astype(np.int16),
            y=np.round(self._y_map[cy, cx]).astype(np.int16),
            timestamp=events.timestamp.copy(),
            polarity=events.polarity.copy(),
        )

    # IMU

    @property
    def has_imu(self) -> bool:
        return self._imu is not None

    @property
    def imu_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._imu_ts

    @property
    def imu_data(self) -> Optional[npt.NDArray[np.float64]]:
        """Full IMU array (N, 6) [ax, ay, az, gx, gy, gz], or None."""
        return self._imu

    def load_imu(
        self, t_start: float, t_end: float
    ) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Return IMU readings and timestamps in [t_start, t_end), or None."""
        if self._imu is None or self._imu_ts is None:
            return None

        i0 = int(np.searchsorted(self._imu_ts, t_start, side="left"))
        i1 = int(np.searchsorted(self._imu_ts, t_end, side="left"))
        return (self._imu[i0:i1].copy(), self._imu_ts[i0:i1].copy())

    # Odometry NPZ

    @property
    def has_odometry_npz(self) -> bool:
        return self._odometry_npz is not None

    @property
    def odometry_npz(self) -> Optional[MVSECOdometryData]:
        return self._odometry_npz

    # GT odometry (LOAM SE(3))

    @property
    def has_gt_odometry(self) -> bool:
        return self._gt_odometry is not None

    @property
    def gt_odometry(self) -> Optional[npt.NDArray[np.float64]]:
        """LOAM odometry SE(3) poses (N, 4, 4), or None."""
        return self._gt_odometry

    @property
    def gt_odometry_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._gt_odometry_ts

    # GT pose (Cartographer SE(3))

    @property
    def has_gt_pose(self) -> bool:
        return self._gt_pose is not None

    @property
    def gt_pose(self) -> Optional[npt.NDArray[np.float64]]:
        """Cartographer SE(3) poses (N, 4, 4), or None."""
        return self._gt_pose

    @property
    def gt_pose_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._gt_pose_ts

    def load_nearest_pose(
        self, t: float, source: str = "pose"
    ) -> Optional[npt.NDArray[np.float64]]:
        """Return the nearest SE(3) pose (4, 4) to time *t*.

        Args:
            t: Query timestamp.
            source: ``"pose"`` for Cartographer or ``"odometry"`` for LOAM.
        """
        if source == "pose":
            poses = self._gt_pose
            ts = self._gt_pose_ts
        elif source == "odometry":
            poses = self._gt_odometry
            ts = self._gt_odometry_ts
        else:
            raise ValueError(f"source must be 'pose' or 'odometry', got '{source}'")

        if poses is None or ts is None:
            return None
        pose: npt.NDArray[np.float64] = poses[find_nearest_index(ts, t)].copy()
        return pose

    # GT depth

    @property
    def has_gt_depth(self) -> bool:
        return self.has_gt_depth_raw or self.has_gt_depth_rect

    @property
    def has_gt_depth_raw(self) -> bool:
        return self._depth_raw_cached is not None or self._depth_raw_lazy is not None

    @property
    def has_gt_depth_rect(self) -> bool:
        return self._depth_rect_cached is not None or self._depth_rect_lazy is not None

    @property
    def depth_raw_images(self) -> Optional[npt.NDArray[np.float32]]:
        return self._depth_raw_cached

    @property
    def depth_rect_images(self) -> Optional[npt.NDArray[np.float32]]:
        return self._depth_rect_cached

    @property
    def gt_depth_raw_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._depth_raw_ts

    @property
    def gt_depth_rect_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._depth_rect_ts

    @property
    def gt_depth_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self.gt_depth_raw_timestamps

    @property
    def num_gt_depth_raw_frames(self) -> int:
        return len(self._depth_raw_ts) if self._depth_raw_ts is not None else 0

    @property
    def num_gt_depth_rect_frames(self) -> int:
        return len(self._depth_rect_ts) if self._depth_rect_ts is not None else 0

    @property
    def num_gt_depth_frames(self) -> int:
        if self._depth_raw_ts is not None:
            return len(self._depth_raw_ts)
        if self._depth_rect_ts is not None:
            return len(self._depth_rect_ts)
        return 0

    def load_depth_raw(self, frame_index: int) -> Optional[npt.NDArray[np.float32]]:
        return cast(
            Optional[npt.NDArray[np.float32]],
            self._read_large_frame(
                self._depth_raw_cached, self._depth_raw_lazy, frame_index, np.float32
            ),
        )

    def load_depth_rect(self, frame_index: int) -> Optional[npt.NDArray[np.float32]]:
        return cast(
            Optional[npt.NDArray[np.float32]],
            self._read_large_frame(
                self._depth_rect_cached, self._depth_rect_lazy, frame_index, np.float32
            ),
        )

    def load_depth(
        self, frame_index: int, rectified: bool = False
    ) -> Optional[npt.NDArray[np.float32]]:
        return self.load_depth_rect(frame_index) if rectified else self.load_depth_raw(frame_index)

    # GT blended images

    @property
    def has_gt_blended(self) -> bool:
        return self._blended_cached is not None or self._blended_lazy is not None

    @property
    def gt_blended_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._blended_ts

    @property
    def blended_images(self) -> Optional[npt.NDArray[np.uint8]]:
        return self._blended_cached

    def load_blended_image(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        return cast(
            Optional[npt.NDArray[np.uint8]],
            self._read_large_frame(self._blended_cached, self._blended_lazy, frame_index, np.uint8),
        )

    # GT flow from HDF5

    @property
    def has_gt_flow_hdf5(self) -> bool:
        return self._flow_h5_cached is not None or self._flow_h5_lazy is not None

    @property
    def flow_hdf5_frames(self) -> Optional[npt.NDArray[np.float64]]:
        return self._flow_h5_cached

    @property
    def gt_flow_hdf5_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._flow_h5_ts

    def load_flow_hdf5(self, frame_index: int) -> Optional[npt.NDArray[np.float64]]:
        """Load a single flow field (2, H, W) float64 from gt HDF5."""
        return cast(
            Optional[npt.NDArray[np.float64]],
            self._read_large_frame(
                self._flow_h5_cached, self._flow_h5_lazy, frame_index, np.float64
            ),
        )

    # Velodyne lidar

    @property
    def has_velodyne(self) -> bool:
        return self._velodyne_cached is not None or self._velodyne_lazy is not None

    @property
    def velodyne_scans(self) -> Optional[npt.NDArray[np.float32]]:
        return self._velodyne_cached

    @property
    def velodyne_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        return self._velodyne_ts

    @property
    def num_velodyne_scans(self) -> int:
        return len(self._velodyne_ts) if self._velodyne_ts is not None else 0

    def load_velodyne_scan(self, scan_index: int) -> Optional[npt.NDArray[np.float32]]:
        """Load a single velodyne scan (N_points, 4) [x, y, z, intensity]."""
        return cast(
            Optional[npt.NDArray[np.float32]],
            self._read_large_frame(
                self._velodyne_cached, self._velodyne_lazy, scan_index, np.float32
            ),
        )

    # Shared lazy/cached dual path reader

    @staticmethod
    def _read_large_frame(
        cached: "Optional[npt.NDArray[np.generic]]",
        lazy: Optional[_LazyH5Dataset],
        index: int,
        dtype: type,
    ) -> "Optional[npt.NDArray[np.generic]]":
        if cached is not None:
            frame: npt.NDArray[np.generic] = cached[index].copy()
            return frame
        if lazy is not None:
            raw = lazy.read(index)
            if raw is None:
                return None
            return np.asarray(raw, dtype=dtype)
        return None
