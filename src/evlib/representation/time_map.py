"""Time-map representation.
"""
from typing import Any
from typing import Tuple

import numpy as np


class TimeMap:
    """Create a time map of events. Adapted from:
    Lagorce, Xavier, et al. "Hots: a hierarchy of event-based time-surfaces for pattern recognition."
    IEEE transactions on pattern analysis and machine intelligence 39.7 (2016): 1346-1359.

    Note that this implementation is a "random access implementation" and does not hold a state.

    Args:
        image_shape: (height, width)
        decay: the factor in the exponential. A higher value leads to a stronger decay (sharper edges).
    """
    def __init__(self, image_shape: Tuple[int, int], decay: float) -> None:
        assert image_shape[0] > 0
        assert image_shape[1] > 0
        self.image_shape = image_shape
        self.decay = decay

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Create time map.

        Args:
            events: a NumPy array of size [n x d], where n is the number of events and d = 4.
                    Every event is encoded with 4 values (y, x, t, p).

        Returns:
            A 2D representation of the input events
        """
        assert events.shape[1] == 4
        time_map = np.zeros(self.image_shape)
        y = events[:, 0].astype(int)
        x = events[:, 1].astype(int)
        t = events[:, 2]
        time_map[y, x] = t
        return np.exp(-self.decay * (np.amax(t) - time_map))  # type: ignore
