"""Abstract base classes for event camera datasets.

    EventDataset              - resource management (close, context manager)
    ├── BlockAccessDataset    - PyTorch map-style (__getitem__, __len__)
    └── IteratorAccessDataset - PyTorch iterable-style (__iter__, __next__)

A DataLoader is not a Dataset. Dataset uses a DataLoader via composition.
DataLoaders live in evlib.dataloaders and provide flexible I/O
(load_events, time_to_index, etc.) for researchers with custom access
patterns. The Dataset adds a sampling contract (__getitem__, __len__)
for PyTorch-like DataLoader integration.
"""

import abc
from typing import Any


class EventDataset(abc.ABC):
    """ABC for any event data source.

    Only resource management, no root/sequence attributes,
    since not all sources have them.
    """

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (file handles, etc.)."""

    def __enter__(self) -> "EventDataset":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class BlockAccessDataset(EventDataset):
    """Map style dataset supporting random access by frame index.

    PyTorch compatible contract:
    __getitem__ returns a sample dict for a given frame index,
    __len__ returns the frame count.

    For low level event I/O (load_events, time_to_index, etc.),
    use the underlying class DataLoaderBase directly via the concrete
    dataset's .loader property.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a sample dict for the given frame index.

        The dict must contain at least an 'events' key with a
        class evlib.types.RawEvents value.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of frames."""


class IteratorAccessDataset(EventDataset):
    """Iterable style dataset for streaming/online sources.

    Subclasses must implement __iter__ and __next__, each yielding
    a dict with at least an 'events' key.
    """

    @abc.abstractmethod
    def __iter__(self) -> "IteratorAccessDataset":
        """Return the iterator (usually self after resetting cursor)."""

    @abc.abstractmethod
    def __next__(self) -> dict:
        """Return the next sample dict, or raise class StopIteration."""

    def reset(self) -> None:
        """Reset iteration to the beginning

        The default implementation raises class NotImplementedError
        subclasses that support rewinding should override this.
        """
        raise NotImplementedError("This iterator does not support reset()")
