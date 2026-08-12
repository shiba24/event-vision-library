"""DSEC dataset wrappers."""

from __future__ import annotations

from typing import Any
from typing import Iterator
from typing import cast

import numpy as np
import numpy.typing as npt

from evlib.dataloaders import DSECCamera
from evlib.dataloaders import DSECDataLoader
from evlib.dataloaders import DSECImuData
from evlib.dataloaders import DSECLidarScan
from evlib.dataloaders import DSECSplit
from evlib.dataloaders import LoadingType
from evlib.dataloaders import LoadMode
from evlib.dataloaders import ResidentLoadMode
from evlib.types import RawEvents

from ._base import BlockAccessDataset
from ._base import BlockDatasetIterator
from ._base import event_sample_collate


dsec_collate_fn = event_sample_collate


class DSECDataset(BlockAccessDataset):
    """Map style dataset for one DSEC sequence.

    The wrapper adds an indexed dataset contract to :class:`evlib.dataloaders.DSECDataLoader`.

    Args:
        root: Directory containing the DSEC dataset.
        sequence: DSEC sequence name.
        split: Dataset split containing the sequence.
        camera: Event-camera stream to use, either ``"left"`` or ``"right"``.
        load_images: Loading mode for rectified images.
        load_flow_forward: Loading mode for forward optical flow.
        load_flow_backward: Loading mode for backward optical flow.
        load_disparity: Loading mode for disparity maps.
        load_imu: Loading mode for IMU samples.
        load_lidar: Loading mode for lidar point clouds.
        load_calibration: Whether to load camera calibration.
        load_rectify_map: Whether to load the event rectification map.
        event_load_mode: Loading mode for event arrays.
        prerectify_events: Whether to rectify cached events during initialization.
    """

    EVENT_SHAPE: tuple[int, int] = DSECDataLoader.EVENT_SHAPE
    IMAGE_SHAPE: tuple[int, int] = DSECDataLoader.IMAGE_SHAPE

    def __init__(
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
        """Initialize a map style dataset for one DSEC sequence."""
        self._loader = DSECDataLoader(
            root,
            sequence,
            split=split,
            camera=camera,
            load_images=load_images,
            load_flow_forward=load_flow_forward,
            load_flow_backward=load_flow_backward,
            load_disparity=load_disparity,
            load_imu=load_imu,
            load_lidar=load_lidar,
            load_calibration=load_calibration,
            load_rectify_map=load_rectify_map,
            event_load_mode=event_load_mode,
            prerectify_events=prerectify_events,
        )

    @property
    def loader(self) -> DSECDataLoader:
        """Return the underlying DSEC dataloader."""
        return self._loader

    @property
    def root(self) -> str:
        """Return the dataset root passed to the loader."""
        return self._loader.root

    @property
    def sequence(self) -> str:
        """Return the DSEC sequence name."""
        return self._loader.sequence

    @property
    def split(self) -> DSECSplit:
        """Return the dataset split."""
        return self._loader.split

    @property
    def camera(self) -> DSECCamera:
        """Return the selected event camera stream: ``"left"`` or ``"right"``."""
        return self._loader.camera

    def __getitem__(self, index: int) -> dict:
        """Return the synchronized DSEC sample at an index."""
        return cast(dict, self._loader.load_frame_sample(index))

    def __len__(self) -> int:
        """Return the number of synchronized samples."""
        return self._loader.num_samples

    def close(self) -> None:
        """Release loader resources."""
        self._loader.close()

    def __repr__(self) -> str:
        """Return a concise dataset representation."""
        return (
            f"{type(self).__name__}("
            f"root={self.root!r}, "
            f"sequence={self.sequence!r}, "
            f"split={self.split!r}, "
            f"camera={self.camera!r})"
        )

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in ``[start_index, end_index)``."""
        return self._loader.load_events(start_index, end_index)

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return self._loader.num_events

    def time_to_index(self, t: float) -> int:
        """Find the last event strictly before time ``t``."""
        return self._loader.time_to_index(t)

    def index_to_time(self, index: int) -> float:
        """Return the timestamp of one event by index."""
        return self._loader.index_to_time(index)

    def times_to_indices(self, timestamps: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Vectorized form of :meth:`time_to_index`."""
        return self._loader.times_to_indices(timestamps)

    def indices_to_times(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Vectorized form of :meth:`index_to_time`."""
        return self._loader.indices_to_times(indices)

    def get_events_by_time(self, t_start: float, t_end: float) -> RawEvents:
        """Load events in ``[t_start, t_end)``."""
        return self._loader.get_events_by_time(t_start, t_end)

    def iter_events(
        self,
        num_events: int | None = None,
        time_window: float | None = None,
    ) -> Iterator[RawEvents]:
        """Yield event chunks by count or by time window."""
        return self._loader.iter_events(num_events=num_events, time_window=time_window)

    @property
    def event_load_mode(self) -> LoadingType:
        """Loading mode used for event arrays."""
        return self._loader.event_load_mode

    @property
    def t_offset(self) -> np.int64:
        """Sequence time offset in microseconds."""
        return self._loader.t_offset

    @property
    def ms_to_idx(self) -> npt.NDArray[np.int64]:
        """Millisecond to event index lookup table."""
        return self._loader.ms_to_idx

    @property
    def has_rectify_map(self) -> bool:
        """Whether an event rectification map is loaded."""
        return self._loader.has_rectify_map

    @property
    def events_prerectified(self) -> bool:
        """Whether cached events were rectified during initialization."""
        return self._loader.events_prerectified

    @property
    def rectify_map(self) -> npt.NDArray[np.float32] | None:
        """Event rectification map, or None when not loaded."""
        return self._loader.rectify_map

    def rectify_events(self, events: RawEvents) -> RawEvents:
        """Map raw event coordinates onto the rectified image plane."""
        return self._loader.rectify_events(events)

    @property
    def has_images(self) -> bool:
        """Whether rectified images are available."""
        return self._loader.has_images

    @property
    def image_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Image timestamps, or None when images are not loaded."""
        return self._loader.image_timestamps

    @property
    def image_exposure_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Image exposure timestamps, or None when unavailable."""
        return self._loader.image_exposure_timestamps

    @property
    def num_images(self) -> int:
        """Number of rectified images."""
        return self._loader.num_images

    def load_image(self, index: int) -> npt.NDArray[np.uint8]:
        """Load one rectified image by index."""
        return self._loader.load_image(index)

    def find_nearest_image_index(self, t: float) -> int:
        """Find the image index nearest to time ``t``."""
        return self._loader.find_nearest_image_index(t)

    @property
    def has_flow_forward(self) -> bool:
        """Whether forward optical flow is available."""
        return self._loader.has_flow_forward

    @property
    def has_flow_backward(self) -> bool:
        """Whether backward optical flow is available."""
        return self._loader.has_flow_backward

    @property
    def flow_forward_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Forward flow timestamps, or None when not loaded."""
        return self._loader.flow_forward_timestamps

    @property
    def flow_backward_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Backward flow timestamps, or None when not loaded."""
        return self._loader.flow_backward_timestamps

    @property
    def num_flow_forward(self) -> int:
        """Number of forward optical flow frames."""
        return self._loader.num_flow_forward

    @property
    def num_flow_backward(self) -> int:
        """Number of backward optical flow frames."""
        return self._loader.num_flow_backward

    def load_flow_forward(
        self, index: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load one forward flow frame and its validity mask."""
        return self._loader.load_flow_forward(index)

    def load_flow_backward(
        self, index: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load one backward flow frame and its validity mask."""
        return self._loader.load_flow_backward(index)

    @property
    def has_disparity(self) -> bool:
        """Whether disparity maps are available."""
        return self._loader.has_disparity

    @property
    def disparity_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Disparity timestamps, or None when not loaded."""
        return self._loader.disparity_timestamps

    @property
    def num_disparity_frames(self) -> int:
        """Number of disparity frames."""
        return self._loader.num_disparity_frames

    def load_disparity(self, index: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Load one disparity map and its validity mask."""
        return self._loader.load_disparity(index)

    @property
    def has_imu(self) -> bool:
        """Whether IMU samples are available."""
        return self._loader.has_imu

    @property
    def imu_timestamps(self) -> npt.NDArray[np.float64] | None:
        """IMU timestamps, or None when not loaded."""
        return self._loader.imu_timestamps

    @property
    def imu_data(self) -> DSECImuData | None:
        """Resident IMU measurements, or None when not loaded."""
        return self._loader.imu_data

    @property
    def num_imu_samples(self) -> int:
        """Number of IMU samples."""
        return self._loader.num_imu_samples

    def load_imu(self, t_start: float, t_end: float) -> DSECImuData:
        """Load IMU measurements in ``[t_start, t_end)``."""
        return self._loader.load_imu(t_start, t_end)

    @property
    def has_lidar(self) -> bool:
        """Whether lidar scans are available."""
        return self._loader.has_lidar

    @property
    def lidar_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Lidar scan timestamps, or None when not loaded."""
        return self._loader.lidar_timestamps

    @property
    def num_lidar_scans(self) -> int:
        """Number of lidar scans."""
        return self._loader.num_lidar_scans

    @property
    def lidar_frame_id(self) -> str | None:
        """Lidar coordinate frame identifier, or None when unavailable."""
        return self._loader.lidar_frame_id

    def load_lidar(self, index: int) -> DSECLidarScan:
        """Load one lidar scan by index."""
        return self._loader.load_lidar(index)

    def load_lidar_by_time(self, t_start: float, t_end: float) -> list[DSECLidarScan]:
        """Load lidar scans in ``[t_start, t_end)``."""
        return self._loader.load_lidar_by_time(t_start, t_end)

    @property
    def has_calibration(self) -> bool:
        """Whether camera calibration is loaded."""
        return self._loader.has_calibration

    @property
    def calibration(self) -> dict[str, Any] | None:
        """Camera calibration, or None when not loaded."""
        return self._loader.calibration

    @property
    def cam_to_lidar_calibration(self) -> dict[str, Any] | None:
        """Camera to lidar extrinsics, or None when not loaded."""
        return self._loader.cam_to_lidar_calibration

    @property
    def cam_to_imu_calibration(self) -> dict[str, Any] | None:
        """Camera to IMU extrinsics, or None when not loaded."""
        return self._loader.cam_to_imu_calibration

    @property
    def imu_calibration(self) -> dict[str, Any] | None:
        """IMU intrinsics, or None when not loaded."""
        return self._loader.imu_calibration


class DSECIterator(BlockDatasetIterator[DSECDataset]):
    """Streaming iterator over one DSEC sequence.

    Args:
        root: Directory containing the DSEC dataset.
        sequence: DSEC sequence name.
        ``**kwargs``: Forwarded to :class:`DSECDataset`.
    """

    def __init__(self, root: str, sequence: str, **kwargs: Any) -> None:
        """Initialize an iterator over one DSEC sequence."""
        super().__init__(DSECDataset(root, sequence, **kwargs))

    @property
    def root(self) -> str:
        """Return the dataset root passed to the loader."""
        return self._dataset.root

    @property
    def sequence(self) -> str:
        """Return the DSEC sequence name."""
        return self._dataset.sequence

    @property
    def split(self) -> DSECSplit:
        """Return the dataset split."""
        return self._dataset.split

    @property
    def camera(self) -> DSECCamera:
        """Return the selected event camera stream: ``"left"`` or ``"right"``."""
        return self._dataset.camera

    def __repr__(self) -> str:
        """Return a concise iterator representation."""
        return (
            f"{type(self).__name__}("
            f"root={self.root!r}, "
            f"sequence={self.sequence!r}, "
            f"split={self.split!r}, "
            f"camera={self.camera!r})"
        )
