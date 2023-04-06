"""Voxel-grid representation.
"""
from typing import Any
from typing import Tuple

import numpy as np


class VoxelGrid:
    """Create a voxel grid from events.
    Implementation inspired by https://github.com/uzh-rpg/rpg_e2vid.

    Args:
        image_shape: (height, width)
        num_bins: number of bins in the temporal axis of the voxel grid
    """
    def __init__(self, image_shape: Tuple[int, int], num_bins: int) -> None:
        assert image_shape[0] > 0
        assert image_shape[1] > 0
        assert num_bins > 0
        self.image_shape = image_shape
        self.num_bins = num_bins

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Create voxel grid.

        Args:
            events: events: a NumPy array of size [n x d], where n is the number of events and d = 4.
                    Every event is encoded with 4 values (y, x, t, p).

        Returns:
            A voxelized representation of the event data.
        """
        assert events.shape[1] == 4
        height, width = self.image_shape
        voxel_grid = np.zeros((self.num_bins, height, width), np.float32).ravel()
        y = events[:, 0].astype(int)
        x = events[:, 1].astype(int)
        t = events[:, 2]
        p = events[:, 3]
        p[p == 0] = -1

        # Normalize the event timestamps so that they lie between 0 and num_bins
        t_min, t_max = np.amin(t), np.amax(t)
        delta_t = t_max - t_min

        if delta_t == 0:
            delta_t = 1

        t = (self.num_bins - 1) * (t - t_min) / delta_t
        tis = t.astype(int)
        dts = t - tis
        vals_left = p * (1.0 - dts)
        vals_right = p * dts
        valid_indices = tis < self.num_bins

        np.add.at(
            voxel_grid,
            x[valid_indices] + y[valid_indices] * width + tis[valid_indices] * width * height,
            vals_left[valid_indices],
        )

        valid_indices = (tis + 1) < self.num_bins
        np.add.at(
            voxel_grid,
            x[valid_indices] + y[valid_indices] * width + (tis[valid_indices] + 1) * width * height,
            vals_right[valid_indices],
        )

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, height, width))

        return voxel_grid
