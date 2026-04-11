"""MVSEC specific lightweight container types."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MVSECOdometryData:
    """Container for MVSEC odometry samples loaded from ``*_odom.npz``."""

    timestamps: npt.NDArray[np.float64]
    linear_velocity: npt.NDArray[np.float64]
    position: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]

    def __len__(self) -> int:
        """Return the number of odometry samples."""
        return len(self.timestamps)
