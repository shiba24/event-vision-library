"""Data type for multiple events."""
import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .raw_event import RawEvent


@dataclass
class RawEvents:
    """Dataclass for a batch of raw events."""
    x: npt.NDArray[np.int16]  # [0, width]
    y: npt.NDArray[np.int16]  # [0, height]
    timestamp: npt.NDArray[np.float64]
    polarity: npt.NDArray[np.bool_]   # true for positive, false for negative

    # Shortcuts
    @property
    def p(self) -> npt.NDArray[np.bool_]:
        """Alias for polatiry.
        """
        return self.polarity

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Alias for timestamp.
        """
        return self.timestamp

    # Build-ins
    def __getitem__(self, index: Union[int, str]) -> npt.NDArray[np.float64]:
        """Get an event as numpy array.

        Args:
            index (int): index in the batch

        Returns:
            npt.NDArray[np.float64]: 1-d numpy array.
        """
        if isinstance(index, str):
            return getattr(self, index)  # type: ignore
        return np.array([self.y[index], self.x[index], self.t[index], self.p[index]], dtype=np.float64)

    def __len__(self) -> int:
        """Get the number of events.

        Returns:
            int: n_events
        """
        return len(self.x)

    @property
    def n(self) -> int:
        """Alias for len().
        """
        return len(self)

    # Utility
    def append(self, e: RawEvent) -> None:
        """Append one event to the event batch object.

        Args:
            e (RawEvent): Event to be appended.
        """
        self.x = np.append(self.x, e.x)
        self.y = np.append(self.y, e.y)
        self.timestamp = np.append(self.timestamp, e.timestamp)
        self.polarity = np.append(self.x, e.x)

    def as_numpy(self) -> npt.NDArray[np.float64]:
        """Convert event object into 2-d numpy array.

        Returns:
            npt.NDArray[np.float64]: 2-d numpy array, [n_events, 4].
        """
        return np.stack([self.y, self.x, self.t, self.p]).astype(np.float64).T
