"""Abstract base class for low level event data I/O.

A DataLoader handles raw file access (e.g. HDF5, NPZ, calibration files, etc.)
and provides primitive operations for reading events by index or time.

It is a separate hierarchy from class EventDataset
a Dataset uses a DataLoader via composition, but a DataLoader is not a Dataset.
"""

import abc
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from evlib.types import RawEvents


_DataLoaderBaseT = TypeVar("_DataLoaderBaseT", bound="DataLoaderBase")


class DataLoaderBase(abc.ABC):
    """ABC for low level event data I/O

    Subclasses implement the abstract methods for a specific file format
    or data source.
    The concrete convenience methods are built on top
    of the abstract primitives.
    """

    # abstract interface

    @abc.abstractmethod
    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        """Load events in [start_index, end_index)."""

    @property
    @abc.abstractmethod
    def num_events(self) -> int:
        """Total number of events."""

    @abc.abstractmethod
    def time_to_index(self, t: float) -> int:
        """Find the index of the last event strictly before time t.

        May return -1 if no event precedes t.
        """

    @abc.abstractmethod
    def index_to_time(self, index: int) -> float:
        """Return the timestamp of the event at index."""

    def times_to_indices(
        self,
        timestamps: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        """Vectorized form of :meth:`time_to_index`.

        Subclasses may override this for a more efficient implementation.
        """
        timestamp_array = self._as_timestamp_array(timestamps)
        flat_timestamps = self._flatten_timestamp_array(timestamp_array)
        resolved_indices = self._resolve_indices_for_timestamps(flat_timestamps)
        reshaped_indices = self._reshape_index_array(
            resolved_indices,
            timestamp_array.shape,
        )
        return reshaped_indices

    def indices_to_times(
        self,
        indices: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Vectorized form of :meth:`index_to_time`.

        Subclasses may override this for a more efficient implementation.
        """
        index_array = self._as_index_array(indices)
        flat_indices = self._flatten_index_array(index_array)
        resolved_timestamps = self._resolve_timestamps_for_indices(flat_indices)
        reshaped_timestamps = self._reshape_timestamp_array(
            resolved_timestamps,
            index_array.shape,
        )
        return reshaped_timestamps

    def _as_timestamp_array(
        self,
        timestamps: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Return timestamps as a float64 ndarray."""
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        return timestamp_array

    def _flatten_timestamp_array(
        self,
        timestamp_array: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return a 1D view of a timestamp array."""
        flat_timestamps = timestamp_array.reshape(-1)
        return flat_timestamps

    def _resolve_indices_for_timestamps(
        self,
        flat_timestamps: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64]:
        """Resolve many timestamps to event indices."""
        resolved_indices = np.fromiter(
            (self.time_to_index(float(timestamp)) for timestamp in flat_timestamps),
            dtype=np.int64,
            count=flat_timestamps.size,
        )
        return resolved_indices

    def _reshape_index_array(
        self,
        flat_indices: npt.NDArray[np.int64],
        shape: Tuple[int, ...],
    ) -> npt.NDArray[np.int64]:
        """Reshape a flat index array to the requested shape."""
        reshaped_indices = flat_indices.reshape(shape)
        return reshaped_indices

    def _as_index_array(
        self,
        indices: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        """Return indices as an int64 ndarray."""
        index_array = np.asarray(indices, dtype=np.int64)
        return index_array

    def _flatten_index_array(
        self,
        index_array: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Return a 1D view of an index array."""
        flat_indices = index_array.reshape(-1)
        return flat_indices

    def _resolve_timestamps_for_indices(
        self,
        flat_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """Resolve many indices to event timestamps."""
        resolved_timestamps = np.fromiter(
            (self.index_to_time(int(index)) for index in flat_indices),
            dtype=np.float64,
            count=flat_indices.size,
        )
        return resolved_timestamps

    def _reshape_timestamp_array(
        self,
        flat_timestamps: npt.NDArray[np.float64],
        shape: Tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Reshape a flat timestamp array to the requested shape."""
        reshaped_timestamps = flat_timestamps.reshape(shape)
        return reshaped_timestamps

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (e.g. file handles, etc.)."""

    # concrete convenience methods

    def get_events_by_time(self, t_start: float, t_end: float) -> RawEvents:
        """Load events whose timestamps fall in [t_start, t_end)."""
        first_at_or_after_start = self._first_event_at_or_after_time(t_start)
        first_at_or_after_end = self._first_event_at_or_after_time(t_end)
        return self.load_events(first_at_or_after_start, first_at_or_after_end)

    def _first_event_at_or_after_time(self, timestamp: float) -> int:
        """Return the first event index whose time is >= ``timestamp``."""
        last_before_timestamp = self.time_to_index(timestamp)
        first_at_or_after_timestamp = last_before_timestamp + 1
        return first_at_or_after_timestamp

    def iter_events(
        self,
        num_events: Optional[int] = None,
        time_window: Optional[float] = None,
    ) -> Iterator[RawEvents]:
        """Yield class RawEvents chunks.

        Exactly one of num_events or time_window must be given.

        Args:
            num_events: Chunk size in number of events (positive).
            time_window: Chunk duration in seconds (positive).

        Raises:
            ValueError: If neither or both arguments are given, or if the
                given value is not positive.
        """
        self._validate_iter_arguments(num_events, time_window)
        if num_events is not None:
            yield from self._iter_events_by_count(num_events)
            return

        assert time_window is not None
        yield from self._iter_events_by_time(time_window)

    def _validate_iter_arguments(
        self,
        num_events: Optional[int],
        time_window: Optional[float],
    ) -> None:
        """Validate iter_events argument exclusivity."""
        both_or_neither_specified = (num_events is None) == (time_window is None)
        if both_or_neither_specified:
            raise ValueError("Specify exactly one of num_events or time_window")

    def _iter_events_by_count(
        self,
        num_events: int,
    ) -> Iterator[RawEvents]:
        """Yield event chunks with a fixed number of events."""
        self._validate_num_events(num_events)
        total_events = self.num_events
        for start_index in range(0, total_events, num_events):
            end_index = min(start_index + num_events, total_events)
            yield self.load_events(start_index, end_index)

    def _validate_num_events(
        self,
        num_events: int,
    ) -> None:
        """Validate the requested event chunk size."""
        if num_events <= 0:
            raise ValueError(f"num_events must be positive, got {num_events}")

    def _iter_events_by_time(
        self,
        time_window: float,
    ) -> Iterator[RawEvents]:
        """Yield event chunks for a fixed time window."""
        self._validate_time_window(time_window)
        is_empty = self._has_no_events()
        if is_empty:
            return

        current_time = self._first_event_time()
        last_time = self._last_event_time()
        current_index = 0

        while current_time < last_time:
            next_time = current_time + time_window
            next_index = self._first_event_at_or_after_time(next_time)
            chunk_is_nonempty = next_index > current_index
            if chunk_is_nonempty:
                yield self.load_events(current_index, next_index)
            current_index = next_index
            current_time = next_time

    def _validate_time_window(
        self,
        time_window: float,
    ) -> None:
        """Validate the requested time window."""
        if time_window <= 0.0:
            raise ValueError(f"time_window must be positive, got {time_window}")

    def _has_no_events(self) -> bool:
        """Return whether the loader has zero events."""
        has_no_events = self.num_events == 0
        return has_no_events

    def _first_event_time(self) -> float:
        """Return the first event timestamp."""
        first_event_time = self.index_to_time(0)
        return first_event_time

    def _last_event_time(self) -> float:
        """Return the last event timestamp."""
        last_event_index = self.num_events - 1
        last_event_time = self.index_to_time(last_event_index)
        return last_event_time

    def __enter__(self: _DataLoaderBaseT) -> _DataLoaderBaseT:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
