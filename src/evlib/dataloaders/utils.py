from functools import lru_cache
from typing import Any
from typing import Tuple

import numpy as np
import numpy.typing as npt


_cv2: Any = None

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

cv2: Any = _cv2


def find_nearest_index(timestamps: npt.NDArray[np.float64], t: float) -> int:
    """Return the index of the timestamp nearest to ``t``."""
    index = int(np.searchsorted(timestamps, t, side="left"))

    if index >= len(timestamps):
        return len(timestamps) - 1

    if index == 0:
        return 0

    left_distance = abs(t - timestamps[index - 1])
    right_distance = abs(t - timestamps[index])
    return index - 1 if left_distance <= right_distance else index


@lru_cache(maxsize=8)
def get_flow_coordinate_grid(
    height: int,
    width: int,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Return cached base coordinate grids for dense flow propagation."""
    x_coords, y_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    x_coords.flags.writeable = False
    y_coords.flags.writeable = False
    return x_coords, y_coords


def sample_flow_nearest_numpy(
    x_flow: npt.NDArray[np.float32],
    y_flow: npt.NDArray[np.float32],
    x_coords: npt.NDArray[np.float32],
    y_coords: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Sample dense flow at floating coordinates with NumPy nearest neighbor."""
    x_indices = np.rint(x_coords).astype(np.int32)
    y_indices = np.rint(y_coords).astype(np.int32)

    height, width = x_flow.shape
    valid = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)

    sampled_x = np.zeros_like(x_flow, dtype=np.float32)
    sampled_y = np.zeros_like(y_flow, dtype=np.float32)

    if np.any(valid):
        sampled_x[valid] = x_flow[y_indices[valid], x_indices[valid]]
        sampled_y[valid] = y_flow[y_indices[valid], x_indices[valid]]

    return sampled_x, sampled_y, valid


def sample_flow_nearest(
    x_flow: npt.NDArray[np.float32],
    y_flow: npt.NDArray[np.float32],
    x_coords: npt.NDArray[np.float32],
    y_coords: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Sample dense flow at floating coordinates with nearest-neighbor."""
    x_indices = np.rint(x_coords).astype(np.int32)
    y_indices = np.rint(y_coords).astype(np.int32)

    height, width = x_flow.shape
    valid = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)

    if cv2 is None:
        return sample_flow_nearest_numpy(x_flow, y_flow, x_coords, y_coords)

    sampled_x = np.asarray(
        cv2.remap(
            x_flow,
            x_coords,
            y_coords,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
        dtype=np.float32,
    )
    sampled_y = np.asarray(
        cv2.remap(
            y_flow,
            x_coords,
            y_coords,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
        dtype=np.float32,
    )
    return sampled_x, sampled_y, valid


def propagate_flow_step(
    x_flow: npt.NDArray[np.float32],
    y_flow: npt.NDArray[np.float32],
    x_coords: npt.NDArray[np.float32],
    y_coords: npt.NDArray[np.float32],
    x_mask: npt.NDArray[np.bool_],
    y_mask: npt.NDArray[np.bool_],
    scale: float,
) -> None:
    """Propagate pixel coordinates through one GT flow field in place."""
    sampled_x, sampled_y, valid = sample_flow_nearest(
        x_flow,
        y_flow,
        x_coords,
        y_coords,
    )

    x_mask &= valid
    y_mask &= valid
    x_mask[sampled_x == 0.0] = False
    y_mask[sampled_y == 0.0] = False

    x_coords += sampled_x * scale
    y_coords += sampled_y * scale
