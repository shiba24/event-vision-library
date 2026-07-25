"""Abstract base classes for event camera datasets.

    EventDataset              - resource management (close, context manager)
    ├── BlockAccessDataset    - PyTorch map-style (__getitem__, __len__)
    └── IteratorAccessDataset - PyTorch iterable-style (__iter__, __next__)

Each branch also derives from its torch.utils.data counterpart, so concrete
datasets can be handed to a torch DataLoader directly.

A DataLoader is not a Dataset. Dataset uses a DataLoader via composition.
DataLoaders live in evlib.dataloaders and provide flexible I/O
(load_events, time_to_index, etc.) for researchers with custom access
patterns. The Dataset adds a sampling contract (__getitem__, __len__)
for PyTorch-like DataLoader integration.
"""

import abc
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import TypeVar

import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info


EventDatasetT = TypeVar("EventDatasetT", bound="EventDataset")
BlockDatasetT = TypeVar("BlockDatasetT", bound="BlockAccessDataset")
BlockDatasetIteratorT = TypeVar(
    "BlockDatasetIteratorT",
    bound="BlockDatasetIterator[Any]",
)


def event_sample_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate event dataset samples.

    Stacks ``"timestamp"`` and keeps other fields as Python lists.
    Preserves variable length events and ``None`` values.
    """
    if not batch:
        raise ValueError("batch must not be empty")

    result: Dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key == "timestamp":
            result[key] = np.asarray(values, dtype=np.float64)
        else:
            result[key] = values
    return result


class EventDataset(abc.ABC):
    """ABC for any event data source.

    Only resource management, no root/sequence attributes,
    since not all sources have them.
    """

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (file handles, etc.)."""

    def __enter__(self: EventDatasetT) -> EventDatasetT:
        """Return this dataset when entering a context manager."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close this dataset when leaving a context manager."""
        self.close()


class BlockAccessDataset(EventDataset, TorchDataset):
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


class IteratorAccessDataset(EventDataset, TorchIterableDataset):
    """Iterable style dataset for streaming/online sources.

    Subclasses must implement __iter__, which returns an iterator, and
    __next__, which returns a dict with at least an 'events' key.

    Subclasses iterate themselves: __iter__ returns the dataset itself.
    A torch DataLoader calls iter() once per epoch, so __iter__ must
    start a fresh pass for the dataset to be readable more than once.
    """

    @abc.abstractmethod
    def __iter__(self) -> "IteratorAccessDataset":
        """Return the iterator (usually self after resetting cursor)."""

    @abc.abstractmethod
    def __next__(self) -> dict:
        """Return the next sample dict, or raise class StopIteration."""

    def reset(self) -> None:
        """Reset iteration to the beginning.

        The default implementation raises :class:`NotImplementedError`;
        subclasses that support rewinding should override it.
        """
        raise NotImplementedError("This iterator does not support reset()")


class BlockDatasetIterator(IteratorAccessDataset, Generic[BlockDatasetT]):
    """Sequential cursor over a finite block access dataset.

    Each call to :func:`iter` starts a new pass from the first sample, which is
    what lets a torch ``DataLoader`` read the sequence again on every epoch. A
    partly consumed pass therefore cannot be resumed with a second :func:`iter`
    call; keep the iterator itself and call :func:`next`, or call :meth:`reset`
    to rewind deliberately. Closing the iterator closes the wrapped dataset.

    Use the wrapped map style dataset directly with a torch ``DataLoader``
    when worker processes, shuffling, or sampling are needed. This iterator
    rejects worker-process iteration because otherwise every worker would
    repeat the full dataset.

    Args:
        dataset: Map style dataset to iterate over.
    """

    def __init__(self, dataset: BlockDatasetT) -> None:
        """Create an iterator over one map style dataset."""
        self._dataset = dataset
        self._current = 0

    @property
    def dataset(self) -> BlockDatasetT:
        """Return the wrapped map style dataset."""
        return self._dataset

    def __iter__(self: BlockDatasetIteratorT) -> BlockDatasetIteratorT:
        """Return this cursor rewound to the first sample."""
        if get_worker_info() is not None:
            raise RuntimeError(
                "BlockDatasetIterator does not support DataLoader worker processes because "
                "each worker would repeat the full dataset. Use num_workers=0 or pass the "
                "wrapped map style dataset to DataLoader."
            )
        self._current = 0
        return self

    def __next__(self) -> dict:
        """Return the next indexed sample."""
        num_samples = len(self._dataset)
        exhausted = self._current >= num_samples
        if exhausted:
            raise StopIteration

        current_index = self._current
        sample = self._dataset[current_index]
        self._current += 1
        return sample

    def reset(self) -> None:
        """Reset the iteration cursor to the first sample."""
        self._current = 0

    def close(self) -> None:
        """Release wrapped dataset resources."""
        self._dataset.close()
