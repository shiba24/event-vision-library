"""Event-Camera Dataset (ECD) wrappers.

Expected text-file structure::
    {root}/{sequence}/events.txt
    {root}/{sequence}/images.txt
    {root}/{sequence}/images/frame_00000000.png
    {root}/{sequence}/imu.txt               (optional)
    {root}/{sequence}/groundtruth.txt       (optional)
    {root}/{sequence}/calib.txt             (optional)
    {root}/{sequence}/depthmaps.txt         (synthetic sequences only, optional)

This dataset wrapper delegates all DAVIS/RPG text recording I/O to
class DavisRecordingLoader and adds the ECD sequence contract.

Reference: https://rpg.ifi.uzh.ch/davis_data.html
Mueggler, E., Rebecq, H., Gallego, G., Delbruck, T., & Scaramuzza, D. (2017).
The Event-Camera Dataset and Simulator: Event-based Data for Pose Estimation,
Visual Odometry, and SLAM. International Journal of Robotics Research, 36(2), 142-149.
"""

from __future__ import annotations

import os
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt

from evlib.dataloaders import DavisCameraCalibration
from evlib.dataloaders import DavisFrameSample
from evlib.dataloaders import DavisImuData
from evlib.dataloaders import DavisPoseData
from evlib.dataloaders import DavisRecordingLoader
from evlib.dataloaders import LoadingType
from evlib.dataloaders import LoadMode
from evlib.dataloaders import ResidentLoadMode
from evlib.types import RawEvents

from ._base import BlockAccessDataset
from ._base import IteratorAccessDataset
from ._base import event_sample_collate


ECD_CAMERA_MODEL = "DAVIS240C"
ECD_DAVIS240C_SENSOR_RESOLUTION: tuple[int, int] = (180, 240)

ECD_REAL_SEQUENCES: tuple[str, ...] = (
    "shapes_rotation",
    "shapes_translation",
    "shapes_6dof",
    "poster_rotation",
    "poster_translation",
    "poster_6dof",
    "boxes_rotation",
    "boxes_translation",
    "boxes_6dof",
    "hdr_poster",
    "hdr_boxes",
    "outdoors_walking",
    "outdoors_running",
    "dynamic_rotation",
    "dynamic_translation",
    "dynamic_6dof",
    "calibration",
    "office_zigzag",
    "office_spiral",
    "urban",
    "slider_close",
    "slider_far",
    "slider_hdr_close",
    "slider_hdr_far",
    "slider_depth",
)
ECD_SYNTHETIC_SEQUENCES: tuple[str, ...] = (
    "simulation_3planes",
    "simulation_3walls",
)
ECD_SEQUENCES: tuple[str, ...] = ECD_REAL_SEQUENCES + ECD_SYNTHETIC_SEQUENCES

ecd_collate_fn = event_sample_collate


def _validate_sequence_name(sequence: str) -> str:
    if not sequence:
        raise ValueError("ECD sequence name must not be empty.")

    sequence_is_path = os.path.basename(sequence) != sequence
    if sequence_is_path:
        raise ValueError(f"ECD sequence must be a sequence name, not a path: {sequence!r}.")

    if sequence not in ECD_SEQUENCES:
        sequence_names = ", ".join(ECD_SEQUENCES)
        raise ValueError(f"Unknown ECD sequence {sequence!r}. Expected one of: {sequence_names}.")

    return sequence


def _has_recording_file(sequence_dir: str) -> bool:
    events_path = os.path.join(sequence_dir, "events.txt")
    return os.path.isfile(events_path)


def _has_nested_recording_file(sequence_dir: str) -> bool:
    if not os.path.isdir(sequence_dir):
        return False

    for child_name in sorted(os.listdir(sequence_dir)):
        child_dir = os.path.join(sequence_dir, child_name)
        if not os.path.isdir(child_dir):
            continue
        if _has_recording_file(child_dir):
            return True

    return False


def _sequence_is_extracted(root_path: str, sequence: str) -> bool:
    sequence_dir = os.path.join(root_path, sequence)
    return _has_recording_file(sequence_dir) or _has_nested_recording_file(sequence_dir)


def _resolve_sequence_dir(root: str, sequence: str) -> str:
    root_path = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"ECD dataset root does not exist: {root}")

    root_is_sequence_dir = os.path.basename(root_path) == sequence
    if root_is_sequence_dir and _has_recording_file(root_path):
        return root_path

    sequence_dir = os.path.join(root_path, sequence)
    if os.path.isdir(sequence_dir):
        return sequence_dir

    zip_path = f"{sequence_dir}.zip"
    if os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"ECD sequence {sequence!r} is available only as {zip_path}. "
            "Extract the text archive before constructing ECDDataset."
        )

    raise FileNotFoundError(
        f"ECD sequence directory does not exist: {sequence_dir}. "
        "Pass the dataset root that contains extracted sequence directories."
    )


class ECDDataset(BlockAccessDataset):
    """Event-Camera Dataset (ECD) sequence.

    The dataset is a wrapper around class DavisRecordingLoader.
    It provides frame indexed samples while keeping low level event slicing and modality access
    available through the ``loader`` property.

    Indexing returns one sample dict per grayscale frame,
    ``len(dataset)`` is the frame count,
    and class ecd_collate_fn collates samples into batches.

    Note:
        Sample timestamps are in each recording's native clock and are not guaranteed to start at zero.
        ECD sequences exported from rosbags carry absolute (Unix epoch) timestamps, for example ``boxes_6dof``,
        others such as ``slider_depth`` start near zero.
        All modalities within one recording share the same clock,
        so time differences and time based queries are always consistent within a sequence.

    Args:
        root: Directory containing extracted ECD text file sequence directories.
        sequence: Official ECD sequence name, for example ``"shapes_rotation"``.
        load_imu: If True, load ``imu.txt`` when present.
        load_gt_pose: If True, load ``groundtruth.txt`` when present.
        event_load_mode: ``"lazy"`` uses read only event sidecars and ``"cached"`` loads event sidecars into memory.
        image_load_mode: ``"lazy"`` decodes frames on demand and ``"cached"`` decodes all frames during initialization.
        depth_load_mode: Load mode for optional synthetic ``depthmaps.txt`` files.
        precompute_frame_event_indices: Whether to precompute frame-to-event index alignment at initialization.
        By default this follows the DAVIS loader.
        cache_dir: Optional root directory for DAVIS event sidecar caches.
    """

    CAMERA_MODEL = ECD_CAMERA_MODEL
    SENSOR_RESOLUTION = ECD_DAVIS240C_SENSOR_RESOLUTION
    SEQUENCES = ECD_SEQUENCES
    REAL_SEQUENCES = ECD_REAL_SEQUENCES
    SYNTHETIC_SEQUENCES = ECD_SYNTHETIC_SEQUENCES

    def __init__(
        self,
        root: str,
        sequence: str,
        *,
        load_imu: bool = True,
        load_gt_pose: bool = True,
        event_load_mode: ResidentLoadMode = "lazy",
        image_load_mode: ResidentLoadMode = "lazy",
        depth_load_mode: LoadMode = True,
        precompute_frame_event_indices: bool | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize an ECD sequence dataset."""
        self._root = os.path.abspath(os.path.expanduser(root))
        self._sequence = _validate_sequence_name(sequence)
        sequence_dir = _resolve_sequence_dir(root, self._sequence)
        self._loader = DavisRecordingLoader(
            sequence_dir,
            load_imu=load_imu,
            load_gt_pose=load_gt_pose,
            event_load_mode=event_load_mode,
            image_load_mode=image_load_mode,
            depth_load_mode=depth_load_mode,
            precompute_frame_event_indices=precompute_frame_event_indices,
            cache_dir=cache_dir,
            sensor_resolution=self.SENSOR_RESOLUTION,
        )

    @classmethod
    def available_sequences(cls, root: str | None = None) -> tuple[str, ...]:
        """Return official sequence names, optionally filtered by extracted files on disk."""
        if root is None:
            return cls.SEQUENCES

        root_path = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"ECD dataset root does not exist: {root}")

        extracted_sequences = [
            sequence for sequence in cls.SEQUENCES if _sequence_is_extracted(root_path, sequence)
        ]
        return tuple(extracted_sequences)

    @property
    def loader(self) -> DavisRecordingLoader:
        """Underlying DAVIS/RPG text recording loader."""
        return self._loader

    @property
    def root(self) -> str:
        """Dataset root passed to this wrapper."""
        return self._root

    @property
    def sequence(self) -> str:
        """Official ECD sequence name."""
        return self._sequence

    @property
    def recording_dir(self) -> str:
        """Resolved extracted recording directory used by the underlying loader."""
        return self._loader.recording_dir

    @property
    def camera_model(self) -> str:
        """Camera model for real ECD DAVIS sequences."""
        return self.CAMERA_MODEL

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """ECD DAVIS240C sensor resolution as ``(height, width)``."""
        return self.SENSOR_RESOLUTION

    def __getitem__(self, index: int) -> dict:
        """Return a synchronized frame indexed sample.

        The sample is a dict with keys:
            ``events``,
            ``timestamp``,
            ``image``,
            ``imu``,
            ``pose``,
            ``depth``
        optional modalities are None when the sequence does not provide them.
        ``timestamp`` is in the recording's native clock - see the class note on timestamps.
        """
        sample = self._loader.load_frame_sample(index)
        return dict(sample)

    def __len__(self) -> int:
        """Number of grayscale frames in the sequence."""
        return self.num_frames

    def close(self) -> None:
        """Release loader resources."""
        self._loader.close()

    def __repr__(self) -> str:
        """Return a concise dataset representation."""
        return f"{type(self).__name__}(root={self.root!r}, sequence={self.sequence!r})"

    # Events

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
    def t_start(self) -> float | None:
        """Timestamp of the first event, or None for empty recordings."""
        return self._loader.t_start

    @property
    def t_end(self) -> float | None:
        """Timestamp of the last event, or None for empty recordings."""
        return self._loader.t_end

    @property
    def duration(self) -> float | None:
        """Event timestamp span in seconds, or None for empty recordings."""
        return self._loader.duration

    @property
    def event_load_mode(self) -> LoadingType:
        """Configured event loading mode."""
        return self._loader.event_load_mode

    @property
    def cache_dir(self) -> str:
        """Root directory for DAVIS event sidecar caches."""
        return self._loader.cache_dir

    # Frames / images

    @property
    def has_images(self) -> bool:
        """Whether image timestamps and frame references are available."""
        return self._loader.has_images

    @property
    def frame_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Frame timestamps from ``images.txt``, or None if unavailable."""
        return self._loader.frame_timestamps

    @property
    def frame_event_indices(self) -> npt.NDArray[np.int64] | None:
        """First event index at or after each frame timestamp, if precomputed."""
        return self._loader.frame_event_indices

    @property
    def image_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Alias for ``frame_timestamps``."""
        return self._loader.image_timestamps

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """Decoded frame shape as ``(height, width)``, if frames are available."""
        return self._loader.image_shape

    @property
    def num_frames(self) -> int:
        """Number of referenced grayscale frames."""
        return self._loader.num_frames

    @property
    def num_images(self) -> int:
        """Alias of ``num_frames``."""
        return self._loader.num_images

    @property
    def image_load_mode(self) -> LoadingType:
        """Configured image loading mode."""
        return self._loader.image_load_mode

    def find_nearest_frame_index(self, t: float) -> int:
        """Return the frame index nearest to ``t``."""
        return self._loader.find_nearest_frame_index(t)

    def find_nearest_image_index(self, t: float) -> int:
        """Alias of :meth:`find_nearest_frame_index`."""
        return self._loader.find_nearest_image_index(t)

    def load_image(self, frame_index: int) -> npt.NDArray[np.uint8] | None:
        """Load one grayscale frame by index."""
        return self._loader.load_image(frame_index)

    def load_frame_sample(self, frame_index: int) -> DavisFrameSample:
        """Load events and optional modalities associated with one frame."""
        return self._loader.load_frame_sample(frame_index)

    # Calibration / undistortion

    @property
    def has_calibration(self) -> bool:
        """Whether ``calib.txt`` is available."""
        return self._loader.calibration is not None

    @property
    def calibration(self) -> DavisCameraCalibration | None:
        """Parsed OpenCV pinhole calibration, if available."""
        return self._loader.calibration

    def undistort_events(self, events: RawEvents) -> RawEvents:
        """Apply DAVIS calibration to event coordinates."""
        return self._loader.undistort_events(events)

    def undistort_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply DAVIS calibration to one grayscale frame."""
        return self._loader.undistort_image(image)

    # IMU

    @property
    def has_imu(self) -> bool:
        """Whether IMU data were requested and loaded."""
        return self._loader.has_imu

    @property
    def imu_timestamps(self) -> npt.NDArray[np.float64] | None:
        """IMU timestamps, or None if IMU is unavailable."""
        return self._loader.imu_timestamps

    @property
    def imu_data(self) -> DavisImuData | None:
        """Loaded IMU arrays, or None if IMU is unavailable."""
        return self._loader.imu_data

    def load_imu(self, t_start: float, t_end: float) -> DavisImuData | None:
        """Return IMU samples in ``[t_start, t_end)``."""
        return self._loader.load_imu(t_start, t_end)

    # Ground-truth pose

    @property
    def has_gt_pose(self) -> bool:
        """Whether ground truth pose data were requested and loaded."""
        return self._loader.has_gt_pose

    @property
    def gt_pose(self) -> DavisPoseData | None:
        """Loaded ground truth pose trajectory, or None if unavailable."""
        return self._loader.gt_pose

    @property
    def gt_pose_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Ground truth pose timestamps, or None if unavailable."""
        return self._loader.gt_pose_timestamps

    def load_nearest_pose(self, t: float) -> npt.NDArray[np.float64] | None:
        """Return nearest raw pose row ``[px, py, pz, qx, qy, qz, qw]``."""
        return self._loader.load_nearest_pose(t)

    # Depth maps

    @property
    def has_depth(self) -> bool:
        """Whether optional ``depthmaps.txt`` is available."""
        return self._loader.has_depth

    @property
    def depth_load_mode(self) -> LoadingType:
        """Configured depth map loading mode."""
        return self._loader.depth_load_mode

    @property
    def depth_timestamps(self) -> npt.NDArray[np.float64] | None:
        """Depth map timestamps, or None if unavailable."""
        return self._loader.depth_timestamps

    @property
    def num_depth_maps(self) -> int:
        """Number of referenced depth maps."""
        return self._loader.num_depth_maps

    def find_nearest_depth_index(self, t: float) -> int:
        """Return the depth map index nearest to ``t``."""
        return self._loader.find_nearest_depth_index(t)

    def load_depth(self, depth_index: int) -> npt.NDArray[np.float32] | None:
        """Load one depth map by index if depth maps are available."""
        return self._loader.load_depth(depth_index)


class ECDIterator(IteratorAccessDataset):
    """Streaming iterator over an ECD sequence.

    Yields the same dicts as :meth:`ECDDataset.__getitem__`, frame by frame.

    Args:
        root: Directory containing extracted ECD text file sequence directories.
        sequence: Official ECD sequence name.
        ``**kwargs``: Forwarded to class ECDDataset.
    """

    def __init__(self, root: str, sequence: str, **kwargs: Any) -> None:
        """Initialize an iterator over one ECD sequence."""
        self._dataset = ECDDataset(root, sequence, **kwargs)
        self._current = 0

    @property
    def dataset(self) -> ECDDataset:
        """Underlying map style ECD dataset."""
        return self._dataset

    @property
    def root(self) -> str:
        """Dataset root passed to the wrapped dataset."""
        return self._dataset.root

    @property
    def sequence(self) -> str:
        """Official ECD sequence name."""
        return self._dataset.sequence

    @property
    def camera_model(self) -> str:
        """Camera model for real ECD DAVIS sequences."""
        return self._dataset.camera_model

    def __iter__(self) -> ECDIterator:
        """Return the iterator reset to the first frame."""
        self._current = 0
        return self

    def __next__(self) -> dict:
        """Return the next frame indexed sample."""
        dataset_length = len(self._dataset)
        if self._current >= dataset_length:
            raise StopIteration

        current_index = self._current
        sample = self._dataset[current_index]
        self._current += 1
        return sample

    def reset(self) -> None:
        """Reset iteration cursor to the beginning."""
        self._current = 0

    def close(self) -> None:
        """Release wrapped dataset resources."""
        self._dataset.close()

    def __repr__(self) -> str:
        """Return a concise iterator representation."""
        return f"{type(self).__name__}(root={self.root!r}, sequence={self.sequence!r})"
