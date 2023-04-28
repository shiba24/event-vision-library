"""Data type for a single event."""
import copy
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class RawEvent:
    """Dataclass for raw events"""

    x: np.int16  # [0, width]
    y: np.int16  # [0, height]
    timestamp: np.float64
    polarity: bool  # true for positive, false for negative

    # Shortcuts
    @property
    def p(self) -> bool:
        """Alias for polatiry.
        """
        return self.polarity

    @property
    def t(self) -> np.float64:
        """Alias for timestamp.
        """
        return self.timestamp

    @property
    def color(self) -> tuple:
        if self.p:
            return (255, 0, 0)  # Red
        else:
            return (0, 0, 255)  # Blue

    def copy(self) -> Any:
        return copy.deepcopy(self)

    def as_numpy(self) -> npt.NDArray[np.float64]:
        """Convert event object into 1-d numpy array.

        Returns:
            npt.NDArray[np.float64]: 1-d numpy array.
        """
        return np.array([self.y, self.x, self.t, self.p], dtype=np.float64)
