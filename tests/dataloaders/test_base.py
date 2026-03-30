"""Tests for DataLoaderBase."""

import numpy as np
import pytest

from evlib.dataloaders._base import DataLoaderBase
from evlib.types import RawEvents


class _DummyLoader(DataLoaderBase):
    def __init__(self) -> None:
        self.closed = False
        self._events = RawEvents(
            x=np.array([0, 1, 2], dtype=np.int16),
            y=np.array([1, 2, 3], dtype=np.int16),
            timestamp=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            polarity=np.array([True, False, True], dtype=np.bool_),
        )

    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        return RawEvents(
            x=self._events.x[start_index:end_index],
            y=self._events.y[start_index:end_index],
            timestamp=self._events.timestamp[start_index:end_index],
            polarity=self._events.polarity[start_index:end_index],
        )

    @property
    def num_events(self) -> int:
        return len(self._events)

    def time_to_index(self, t: float) -> int:
        return int(np.searchsorted(self._events.timestamp, t, side="left")) - 1

    def index_to_time(self, index: int) -> float:
        return float(self._events.timestamp[index])

    def close(self) -> None:
        self.closed = True


class _EmptyLoader(DataLoaderBase):
    def load_events(self, start_index: int, end_index: int) -> RawEvents:
        return RawEvents(
            x=np.array([], dtype=np.int16),
            y=np.array([], dtype=np.int16),
            timestamp=np.array([], dtype=np.float64),
            polarity=np.array([], dtype=np.bool_),
        )

    @property
    def num_events(self) -> int:
        return 0

    def time_to_index(self, t: float) -> int:
        return -1

    def index_to_time(self, index: int) -> float:
        return 0.0

    def close(self) -> None:
        pass


class TestDataLoaderBase:
    def test_abstract_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            DataLoaderBase()  # type: ignore[abstract]

    def test_context_manager_closes(self) -> None:
        loader = _DummyLoader()
        with loader:
            assert loader.closed is False
        assert loader.closed is True

    def test_get_events_by_time(self) -> None:
        loader = _DummyLoader()
        events = loader.get_events_by_time(0.15, 0.31)
        expected_timestamps = np.array([0.2, 0.3], dtype=np.float64)
        assert len(events) == 2
        np.testing.assert_array_equal(events.timestamp, expected_timestamps)

    def test_vectorized_helpers_match_scalar_methods(self) -> None:
        loader = _DummyLoader()
        timestamps = np.array([0.05, 0.2, 0.31], dtype=np.float64)
        indices = loader.times_to_indices(timestamps)
        expected_indices = np.array([-1, 0, 2], dtype=np.int64)
        np.testing.assert_array_equal(indices, expected_indices)

        event_indices = np.array([0, 2], dtype=np.int64)
        times = loader.indices_to_times(event_indices)
        expected_times = np.array([0.1, 0.3], dtype=np.float64)
        np.testing.assert_array_equal(times, expected_times)

    def test_iter_events_by_count(self) -> None:
        loader = _DummyLoader()
        chunks = list(loader.iter_events(num_events=2))
        chunk_lengths = [len(chunk) for chunk in chunks]
        expected_lengths = [2, 1]
        assert chunk_lengths == expected_lengths

    def test_iter_events_by_time_window(self) -> None:
        loader = _DummyLoader()
        chunks = list(loader.iter_events(time_window=0.15))
        chunk_lengths = [len(chunk) for chunk in chunks]
        assert len(chunks) >= 1
        assert sum(chunk_lengths) <= loader.num_events

    def test_iter_events_by_time_window_large_window(self) -> None:
        loader = _DummyLoader()
        chunks = list(loader.iter_events(time_window=1.0))
        chunk_lengths = [len(chunk) for chunk in chunks]
        total_events = sum(chunk_lengths)
        assert total_events == loader.num_events

    def test_iter_events_by_time_window_empty_loader(self) -> None:
        loader = _EmptyLoader()
        chunks = list(loader.iter_events(time_window=0.1))
        assert chunks == []

    def test_iter_events_neither_argument_raises(self) -> None:
        loader = _DummyLoader()
        with pytest.raises(ValueError, match="Specify exactly one"):
            list(loader.iter_events())

    def test_iter_events_both_arguments_raises(self) -> None:
        loader = _DummyLoader()
        with pytest.raises(ValueError, match="Specify exactly one"):
            list(loader.iter_events(num_events=2, time_window=0.1))

    def test_iter_events_negative_num_events_raises(self) -> None:
        loader = _DummyLoader()
        with pytest.raises(ValueError, match="num_events must be positive"):
            list(loader.iter_events(num_events=-1))

    def test_iter_events_negative_time_window_raises(self) -> None:
        loader = _DummyLoader()
        with pytest.raises(ValueError, match="time_window must be positive"):
            list(loader.iter_events(time_window=-1.0))

    def test_iter_events_by_time_window_skips_empty_chunks(self) -> None:
        loader = _DummyLoader()
        # Window so small that some iterations produce no new events
        chunks = list(loader.iter_events(time_window=0.01))
        total_events = sum(len(c) for c in chunks)
        assert total_events == loader.num_events
