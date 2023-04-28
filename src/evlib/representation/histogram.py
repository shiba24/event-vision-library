"""Histogram representation.
"""
from typing import Any
from typing import Tuple

import numpy as np


class Histogram:
    """Create a 2D histogram from event camera data.

    Args:
        image_shape: (height, width)
        use_polarity: if True, counts every positive events as +1
                        and every negative events as -1, if False,
                        counts every events as +1
    """
    def __init__(self, image_shape: Tuple[int, int], use_polarity: bool = True) -> None:
        assert image_shape[0] > 0
        assert image_shape[1] > 0
        self.image_shape = image_shape
        self.use_polarity = use_polarity

    def _make_histogram(self, events: np.ndarray) -> np.ndarray:
        y = events[:, 0].astype(int)
        x = events[:, 1].astype(int)
        histogram, _, _ = np.histogram2d(
            y,
            x,
            bins=[self.image_shape[0], self.image_shape[1]],
            range=[[0, self.image_shape[0]], [0, self.image_shape[1]]],
        )
        return histogram  # type: ignore

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Create histogram.

        Args:
            events: a NumPy array of size [n x d], where n is the number of events and d = 4.
                    Every event is encoded with 4 values (y, x, t, p).

        Returns:
             A 2D histogram where each pixel holds the number of events in it.
        """
        assert events.shape[1] == 4
        if self.use_polarity:
            p = events[:, 3].astype(int)
            pos_events = events[(p == 1)]
            histogram = self._make_histogram(pos_events)
            neg_events = events[(p == 0)]
            histogram -= self._make_histogram(neg_events)
        else:
            histogram = self._make_histogram(events)
        return histogram
