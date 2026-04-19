"""MVSEC dataset loader.

Expected file structure:
    {root}/{sequence}_data.hdf5
    {root}/{sequence}_gt_flow_dist.npz                      (optional)
    {root}/{sequence}_odom.npz                              (optional)
    {root}/{sequence}_gt.hdf5                               (optional)
    {root}/{category}_{camera}_x_map.txt                    (optional)
    {root}/{category}_calib/{category}_{camera}_x_map.txt   (optional, alternative)

category is the sequence name without its trailing digit
(e.g. "indoor_flying" for "indoor_flying1")
Calibration maps are searched in root first, then in {root}/{category}_calib/

Reference: https://daniilidis-group.github.io/mvsec/
Zhu, A. Z., Thakur, D., Ozaslan, T., Pfrommer, B., Kumar, V., & Daniilidis, K. (2018).
The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception.
IEEE Robotics and Automation Letters, 3(3), 2032-2039.
"""

from __future__ import annotations

from typing import Any
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt

from evlib.dataloaders import LoadingType
from evlib.dataloaders import LoadMode
from evlib.dataloaders import MVSECDataLoader
from evlib.dataloaders import MVSECOdometryData
from evlib.dataloaders import ResidentLoadMode
from evlib.types import RawEvents

from ._base import BlockAccessDataset
from ._base import IteratorAccessDataset


def mvsec_collate_fn(batch: List[dict]) -> dict:
    """Collate MVSEC samples with variable length events.

    Uses batch[0] keys, stacks ``timestamp``, and leaves other fields as lists.
    Preserves variable length events and ``None`` values.
    """
    if not batch:
        raise ValueError("batch must not be empty")

    result: dict = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key == "timestamp":
            result[key] = np.asarray(values, dtype=np.float64)
        else:
            result[key] = values
    return result


class MVSECDataset(BlockAccessDataset):
    """MVSEC dataset (block access / map style).

    Thin wrapper around class MVSECDataLoader that adds a frame indexed __getitem__ / __len__ contract suitable for PyTorch DataLoader integration.

    For custom access patterns (overlapping windows, multi-scale pyramids, arbitrary time slicing), use the attr loader directly

        ds = MVSECDataset(root, "indoor_flying1")
        loader = ds.loader
        events = loader.load_events(0, 10000)
        events = loader.get_events_by_time(t_start, t_end)

    Or instantiate class MVSECDataLoader standalone, no Dataset needed.

    Args:
        root: Directory containing the MVSEC files.
        sequence: Sequence name, e.g. "indoor_flying1".
        camera: "left" or "right".
        load_gt_flow: LoadMode for ground truth optical flow.
        load_calibration: If True, load calibration maps.
        load_imu: If True, cache IMU data.
        load_odometry_npz: If True, cache odometry from NPZ.
        load_gt_odometry: If True, cache LOAM odometry from gt HDF5.
        load_gt_poses: If True, cache Cartographer poses from gt HDF5.
        load_gt_depth_raw: LoadMode for raw depth maps from gt HDF5.
        load_gt_depth_rect: LoadMode for rectified depth maps from gt HDF5.
        load_gt_flow_hdf5: LoadMode for optical flow from gt HDF5.
        load_gt_blended: LoadMode for blended images from gt HDF5.
        load_velodyne: LoadMode for velodyne lidar from data HDF5.
        event_load_mode: ``"cached"`` or ``"lazy"`` for events.
        image_load_mode: ``"cached"`` or ``"lazy"`` for images.
        cache_dir: Optional root directory for MVSEC sidecar caches.
    """

    IMAGE_SHAPE: Tuple[int, int] = MVSECDataLoader.IMAGE_SHAPE

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
        self._loader = MVSECDataLoader(
            root,
            sequence,
            camera,
            load_gt_flow=load_gt_flow,
            load_calibration=load_calibration,
            load_imu=load_imu,
            load_odometry_npz=load_odometry_npz,
            load_gt_odometry=load_gt_odometry,
            load_gt_poses=load_gt_poses,
            load_gt_depth_raw=load_gt_depth_raw,
            load_gt_depth_rect=load_gt_depth_rect,
            load_gt_flow_hdf5=load_gt_flow_hdf5,
            load_gt_blended=load_gt_blended,
            load_velodyne=load_velodyne,
            event_load_mode=event_load_mode,
            image_load_mode=image_load_mode,
            cache_dir=cache_dir,
        )

    # Underlying DataLoader

    @property
    def loader(self) -> MVSECDataLoader:
        """Underlying MVSECDataLoader."""
        return self._loader

    @property
    def root(self) -> str:
        return self._loader.root

    @property
    def sequence(self) -> str:
        return self._loader.sequence

    @property
    def camera(self) -> str:
        return self._loader.camera

    # BlockAccessDataset contract (PyTorch style)

    def __getitem__(self, index: int) -> dict:
        """Return a synchronized single-camera sample for the given frame index."""
        return self._loader.load_frame_sample(index)

    def __len__(self) -> int:
        """Number of frames."""
        return self.num_frames

    def close(self) -> None:
        self._loader.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"root={self.root!r}, "
            f"sequence={self.sequence!r}, "
            f"camera={self.camera!r})"
        )

    # Convenience delegations to loader.
    # sections below expose selected loader APIs directly on the dataset so callers can use ds.load_events(...) without reaching through ds.loader.

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in [start_index, end_index)."""
        return self._loader.load_events(start_index, end_index)

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return self._loader.num_events

    def time_to_index(self, t: float) -> int:
        """Find the last event strictly before time t."""
        return self._loader.time_to_index(t)

    def index_to_time(self, index: int) -> float:
        """Return the timestamp of the event at index."""
        return self._loader.index_to_time(index)

    def times_to_indices(
        self,
        timestamps: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        """Vectorized form of :meth:`time_to_index`."""
        return self._loader.times_to_indices(timestamps)

    def indices_to_times(
        self,
        indices: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Vectorized form of :meth:`index_to_time`."""
        return self._loader.indices_to_times(indices)

    def get_events_by_time(self, t_start: float, t_end: float) -> RawEvents:
        """Load events in [t_start, t_end)."""
        return self._loader.get_events_by_time(t_start, t_end)

    def iter_events(
        self,
        num_events: Optional[int] = None,
        time_window: Optional[float] = None,
    ) -> Iterator[RawEvents]:
        """Yield RawEvents chunks."""
        return self._loader.iter_events(num_events=num_events, time_window=time_window)

    # GT optical flow

    @property
    def has_gt_flow(self) -> bool:
        """Whether ground truth optical flow is available."""
        return self._loader.has_gt_flow

    def load_optical_flow(self, t1: float, t2: float) -> npt.NDArray[np.float32]:
        """Load ground truth optical flow between two timestamps."""
        return self._loader.load_optical_flow(t1, t2)

    def get_gt_timestamps(self, event_index: int) -> Tuple[Optional[float], Optional[float]]:
        """Return the floor and ceil GT timestamps bracketing *event_index*."""
        return self._loader.get_gt_timestamps(event_index)

    def gt_time_list(self) -> npt.NDArray[np.float64]:
        """Return all GT timestamps."""
        return self._loader.gt_time_list()

    # Frames / images

    @property
    def frame_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Grayscale frame timestamps from HDF5, or None if unavailable."""
        return self._loader.frame_timestamps

    @property
    def frame_event_indices(self) -> Optional[npt.NDArray[np.int64]]:
        """Event indices aligned to grayscale frames, or None if unavailable."""
        return self._loader.frame_event_indices

    @property
    def event_load_mode(self) -> LoadingType:
        """Configured event loading mode."""
        return self._loader.event_load_mode

    @property
    def image_load_mode(self) -> LoadingType:
        """Configured image loading mode."""
        return self._loader.image_load_mode

    @property
    def has_images(self) -> bool:
        """Whether grayscale images are available."""
        return self._loader.has_images

    def load_image(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """Load a single grayscale frame by index."""
        return self._loader.load_image(frame_index)

    @property
    def images(self) -> Optional[npt.NDArray[np.uint8]]:
        """Cached grayscale image stack, or None if lazy/unavailable."""
        return self._loader.images

    @property
    def num_frames(self) -> int:
        """Number of grayscale frames."""
        return self._loader.num_frames

    def find_nearest_frame_index(self, t: float) -> int:
        """Find the nearest grayscale frame to time *t*."""
        return self._loader.find_nearest_frame_index(t)

    # Calibration

    @property
    def has_calibration(self) -> bool:
        """Whether calibration maps are available."""
        return self._loader.has_calibration

    def undistort_events(self, events: RawEvents) -> RawEvents:
        """Apply calibration rectification maps to events."""
        return self._loader.undistort_events(events)

    # IMU

    @property
    def has_imu(self) -> bool:
        """Whether IMU data is available."""
        return self._loader.has_imu

    @property
    def imu_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """IMU timestamps, or None."""
        return self._loader.imu_timestamps

    @property
    def imu_data(self) -> Optional[npt.NDArray[np.float64]]:
        """Full IMU array (N, 6), or None."""
        return self._loader.imu_data

    def load_imu(
        self, t_start: float, t_end: float
    ) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Return IMU readings and timestamps in [t_start, t_end)."""
        return self._loader.load_imu(t_start, t_end)

    # Odometry NPZ

    @property
    def has_odometry_npz(self) -> bool:
        """Whether odometry NPZ data is available."""
        return self._loader.has_odometry_npz

    @property
    def odometry_npz(self) -> Optional[MVSECOdometryData]:
        """MVSEC odometry data from NPZ, or None."""
        return self._loader.odometry_npz

    # GT odometry

    @property
    def has_gt_odometry(self) -> bool:
        """Whether GT odometry (LOAM) is available."""
        return self._loader.has_gt_odometry

    @property
    def gt_odometry(self) -> Optional[npt.NDArray[np.float64]]:
        """LOAM odometry SE(3) poses (N, 4, 4), or None."""
        return self._loader.gt_odometry

    @property
    def gt_odometry_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """LOAM odometry timestamps, or None."""
        return self._loader.gt_odometry_timestamps

    # GT pose

    @property
    def has_gt_pose(self) -> bool:
        """Whether GT pose (Cartographer) is available."""
        return self._loader.has_gt_pose

    @property
    def gt_pose(self) -> Optional[npt.NDArray[np.float64]]:
        """Cartographer SE(3) poses (N, 4, 4), or None."""
        return self._loader.gt_pose

    @property
    def gt_pose_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Cartographer pose timestamps, or None."""
        return self._loader.gt_pose_timestamps

    def load_nearest_pose(
        self, t: float, source: str = "pose"
    ) -> Optional[npt.NDArray[np.float64]]:
        """Return the nearest SE(3) pose (4, 4) to time *t*."""
        return self._loader.load_nearest_pose(t, source=source)

    # GT depth

    @property
    def has_gt_depth(self) -> bool:
        """Whether GT depth maps are available."""
        return self._loader.has_gt_depth

    @property
    def has_gt_depth_raw(self) -> bool:
        """Whether raw GT depth maps are available."""
        return self._loader.has_gt_depth_raw

    @property
    def has_gt_depth_rect(self) -> bool:
        """Whether rectified GT depth maps are available."""
        return self._loader.has_gt_depth_rect

    @property
    def gt_depth_raw_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Raw depth timestamps, or None."""
        return self._loader.gt_depth_raw_timestamps

    @property
    def gt_depth_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Alias for raw depth timestamps."""
        return self._loader.gt_depth_timestamps

    @property
    def gt_depth_rect_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Rectified depth timestamps, or None."""
        return self._loader.gt_depth_rect_timestamps

    @property
    def depth_raw_images(self) -> Optional[npt.NDArray[np.float32]]:
        """Cached raw depth image stack, or None if lazy/unavailable."""
        return self._loader.depth_raw_images

    @property
    def num_gt_depth_raw_frames(self) -> int:
        """Number of raw GT depth frames."""
        return self._loader.num_gt_depth_raw_frames

    @property
    def num_gt_depth_rect_frames(self) -> int:
        """Number of rectified GT depth frames."""
        return self._loader.num_gt_depth_rect_frames

    @property
    def num_gt_depth_frames(self) -> int:
        """Number of GT depth frames from the available depth source."""
        return self._loader.num_gt_depth_frames

    @property
    def depth_rect_images(self) -> Optional[npt.NDArray[np.float32]]:
        """Cached rectified depth image stack, or None if lazy/unavailable."""
        return self._loader.depth_rect_images

    def load_depth(
        self, frame_index: int, rectified: bool = False
    ) -> Optional[npt.NDArray[np.float32]]:
        """Load a single depth frame."""
        return self._loader.load_depth(frame_index, rectified=rectified)

    def load_depth_raw(self, frame_index: int) -> Optional[npt.NDArray[np.float32]]:
        """Load a single raw depth frame."""
        return self._loader.load_depth_raw(frame_index)

    def load_depth_rect(self, frame_index: int) -> Optional[npt.NDArray[np.float32]]:
        """Load a single rectified depth frame."""
        return self._loader.load_depth_rect(frame_index)

    # GT blended images

    @property
    def has_gt_blended(self) -> bool:
        """Whether GT blended images are available."""
        return self._loader.has_gt_blended

    @property
    def gt_blended_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Blended image timestamps, or None."""
        return self._loader.gt_blended_timestamps

    @property
    def blended_images(self) -> Optional[npt.NDArray[np.uint8]]:
        """Cached blended image stack, or None if lazy/unavailable."""
        return self._loader.blended_images

    def load_blended_image(self, frame_index: int) -> Optional[npt.NDArray[np.uint8]]:
        """Load a single blended image."""
        return self._loader.load_blended_image(frame_index)

    # GT flow from HDF5

    @property
    def has_gt_flow_hdf5(self) -> bool:
        """Whether GT flow from HDF5 is available."""
        return self._loader.has_gt_flow_hdf5

    @property
    def flow_hdf5_frames(self) -> Optional[npt.NDArray[np.float64]]:
        """Cached GT HDF5 flow stack, or None if lazy/unavailable."""
        return self._loader.flow_hdf5_frames

    @property
    def gt_flow_hdf5_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """GT HDF5 flow timestamps, or None."""
        return self._loader.gt_flow_hdf5_timestamps

    def load_flow_hdf5(self, frame_index: int) -> Optional[npt.NDArray[np.float64]]:
        """Load a single flow field from gt HDF5."""
        return self._loader.load_flow_hdf5(frame_index)

    # Velodyne lidar

    @property
    def has_velodyne(self) -> bool:
        """Whether velodyne lidar data is available."""
        return self._loader.has_velodyne

    @property
    def velodyne_timestamps(self) -> Optional[npt.NDArray[np.float64]]:
        """Velodyne scan timestamps, or None."""
        return self._loader.velodyne_timestamps

    @property
    def velodyne_scans(self) -> Optional[npt.NDArray[np.float32]]:
        """Cached velodyne scan stack, or None if lazy/unavailable."""
        return self._loader.velodyne_scans

    @property
    def num_velodyne_scans(self) -> int:
        """Number of velodyne scans."""
        return self._loader.num_velodyne_scans

    def load_velodyne_scan(self, scan_index: int) -> Optional[npt.NDArray[np.float32]]:
        """Load a single velodyne scan."""
        return self._loader.load_velodyne_scan(scan_index)


class MVSECIterator(IteratorAccessDataset):
    """Streaming iterator over MVSEC frames.

    Yields the same dicts as :meth:`MVSECDataset.__getitem__`, frame by frame.

    Args:
        root: Directory containing the MVSEC files.
        sequence: Sequence name.
        **kwargs: Forwarded to class MVSECDataset.
    """

    def __init__(self, root: str, sequence: str, **kwargs: Any) -> None:
        self._dataset = MVSECDataset(root, sequence, **kwargs)
        self._current = 0

    @property
    def root(self) -> str:
        return self._dataset.root

    @property
    def sequence(self) -> str:
        return self._dataset.sequence

    @property
    def camera(self) -> str:
        return self._dataset.camera

    def __iter__(self) -> "MVSECIterator":
        self._current = 0
        return self

    def __next__(self) -> dict:
        dataset_length = len(self._dataset)
        exhausted = self._current >= dataset_length
        if exhausted:
            raise StopIteration
        current_index = self._current
        sample = self._dataset[current_index]
        self._current += 1
        return sample

    def reset(self) -> None:
        """Reset iteration cursor to the beginning."""
        self._current = 0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"root={self.root!r}, "
            f"sequence={self.sequence!r}, "
            f"camera={self.camera!r})"
        )

    def close(self) -> None:
        self._dataset.close()
