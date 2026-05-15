"""Shared helpers for dataloader indexing, caching, and optional decoding."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt


_DecodedValueT = TypeVar("_DecodedValueT")

_cv2: Any = None

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

cv2: Any = _cv2


def freeze_array(arr: "Optional[npt.NDArray[np.generic]]") -> None:
    """Make an array read only if it exists."""
    if arr is not None:
        arr.flags.writeable = False


class LazyDecodeCache(Generic[_DecodedValueT]):
    """Small process local LRU cache for decoded file payloads."""

    def __init__(self, max_items: int) -> None:
        """Create a cache that keeps at most ``max_items`` decoded values."""
        if max_items < 0:
            raise ValueError(f"max_items must be >= 0, got {max_items}")
        self._max_items = max_items
        self._items: OrderedDict[int, _DecodedValueT] = OrderedDict()

    def get(self, index: int) -> Optional[_DecodedValueT]:
        """Return a cached decoded value by index, or None on a miss."""
        if self._max_items == 0:
            return None

        cached_value = self._items.get(index)
        if cached_value is None:
            return None

        self._items.move_to_end(index)
        return cached_value

    def put(self, index: int, decoded_value: _DecodedValueT) -> _DecodedValueT:
        """Store and return a decoded value."""
        if self._max_items == 0:
            return decoded_value

        self._items[index] = decoded_value
        self._items.move_to_end(index)

        while len(self._items) > self._max_items:
            self._items.popitem(last=False)
        return decoded_value

    def clear(self) -> None:
        """Remove all cached values."""
        self._items.clear()


def decode_in_parallel(
    paths: List[str],
    decoder: Callable[[str], Any],
    *,
    max_workers: int,
) -> List[Any]:
    """Decode files with a bounded thread pool."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    if not paths:
        return []
    if len(paths) == 1:
        return [decoder(paths[0])]

    workers = min(max_workers, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(decoder, paths))


def normalize_index(index: int, sequence_length: int, item_name: str) -> int:
    """Normalize one Python style sequence index and validate bounds."""
    normalized_index = index
    if normalized_index < 0:
        normalized_index += sequence_length

    index_is_out_of_range = normalized_index < 0 or normalized_index >= sequence_length
    if index_is_out_of_range:
        raise IndexError(f"{item_name} index {index} out of range")
    return normalized_index


def normalize_indices(
    indices: npt.ArrayLike,
    sequence_length: int,
    item_name: str,
) -> npt.NDArray[np.int64]:
    """Normalize many Python style sequence indices and validate bounds."""
    index_array = np.asarray(indices, dtype=np.int64)
    normalized = np.where(index_array < 0, index_array + sequence_length, index_array)

    index_is_out_of_range = (normalized < 0) | (normalized >= sequence_length)
    if bool(np.any(index_is_out_of_range)):
        raise IndexError(f"{item_name} index out of range")
    return np.asarray(normalized, dtype=np.int64)


def validate_index_interval(
    start_index: int,
    end_index: int,
    sequence_length: int,
    item_name: str,
) -> Tuple[int, int]:
    """Validate a half open ``[start_index, end_index)`` interval."""
    if start_index < 0:
        raise IndexError(f"{item_name} start index {start_index} out of range")
    if end_index < 0:
        raise IndexError(f"{item_name} end index {end_index} out of range")
    if end_index < start_index:
        raise ValueError(
            f"{item_name} end index must be >= start index, "
            f"got start={start_index}, end={end_index}"
        )
    if start_index > sequence_length:
        raise IndexError(f"{item_name} start index {start_index} out of range")
    if end_index > sequence_length:
        raise IndexError(f"{item_name} end index {end_index} out of range")
    return start_index, end_index


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
