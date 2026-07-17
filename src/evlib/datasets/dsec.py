"""DSEC dataset wrappers."""

from __future__ import annotations

from typing import Any
from typing import cast

from evlib.dataloaders import DSECCamera
from evlib.dataloaders import DSECDataLoader
from evlib.dataloaders import DSECSplit
from evlib.dataloaders import LoadMode
from evlib.dataloaders import ResidentLoadMode

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
