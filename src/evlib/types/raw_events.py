"""Data type for multiple events."""

from dataclasses import dataclass
from typing import Any
from typing import Union
from typing import overload

import numpy as np
import numpy.typing as npt

from .raw_event import RawEvent


@dataclass
class RawEvents:
    """Dataclass for a batch of raw events.

    Dense rows returned by integer indexing and :meth:`as_numpy` follow
    the ``[y, x, t, p]`` convention.
    """

    x: Union[npt.NDArray[np.int16], npt.NDArray[np.int32]]  # [0, width]
    y: Union[npt.NDArray[np.int16], npt.NDArray[np.int32]]  # [0, height]
    timestamp: npt.NDArray[np.float64]
    polarity: npt.NDArray[np.bool_]  # true for positive, false for negative

    # Shortcuts
    @property
    def p(self) -> npt.NDArray[np.bool_]:
        """Alias for polarity."""
        return self.polarity

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Alias for timestamp."""
        return self.timestamp

    # Build-ins
    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.float64]:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> "RawEvents":  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: str) -> Any:  # noqa: D105
        ...

    def __getitem__(
        self, index: Union[int, slice, str]
    ) -> Union[Any, "RawEvents", npt.NDArray[np.float64]]:
        """Get an event, event slice, or named column.

        Integer indexes return dense ``[y, x, t, p]`` rows, while slice indexes
        return a ``RawEvents`` batch.

        Args:
            index: Integer index, slice, or column name.

        Returns:
            The selected event, event batch, or named column.
        """
        if isinstance(index, str):
            return getattr(self, index)
        if isinstance(index, slice):
            return RawEvents(
                x=self.x[index],
                y=self.y[index],
                timestamp=self.timestamp[index],
                polarity=self.polarity[index],
            )
        event = np.array(
            [self.y[index], self.x[index], self.t[index], self.p[index]],
            dtype=np.float64,
        )
        return event

    def __len__(self) -> int:
        """Get the number of events.

        Returns:
            int: n_events
        """
        return len(self.x)

    @property
    def n(self) -> int:
        """Return the event count."""
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
        self.polarity = np.append(self.polarity, e.polarity)

    def as_numpy(self) -> npt.NDArray[np.float64]:
        """Convert event object into 2-d numpy array.

        Rows follow the ``[y, x, t, p]`` convention.

        Returns:
            npt.NDArray[np.float64]: 2-d numpy array, [n_events, 4].
        """
        events = np.column_stack((self.y, self.x, self.t, self.p))
        return events.astype(np.float64)
