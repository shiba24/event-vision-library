"""Tests for dataset base classes."""

import pytest

from evlib.datasets._base import BlockAccessDataset
from evlib.datasets._base import EventDataset
from evlib.datasets._base import IteratorAccessDataset


class _DummyEventDataset(EventDataset):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyBlockDataset(BlockAccessDataset):
    def __init__(self) -> None:
        self.closed = False

    def __getitem__(self, index: int) -> dict:
        return {"events": index}

    def __len__(self) -> int:
        return 3

    def close(self) -> None:
        self.closed = True


class _DummyIteratorDataset(IteratorAccessDataset):
    def __init__(self) -> None:
        self.closed = False
        self._cursor = 0

    def __iter__(self) -> "_DummyIteratorDataset":
        self._cursor = 0
        return self

    def __next__(self) -> dict:
        if self._cursor >= 2:
            raise StopIteration
        value = {"events": self._cursor}
        self._cursor += 1
        return value

    def close(self) -> None:
        self.closed = True


class TestDatasetBaseClasses:
    def test_event_dataset_context_manager_closes(self) -> None:
        ds = _DummyEventDataset()
        with ds:
            assert ds.closed is False
        assert ds.closed is True

    def test_block_access_dataset_contract(self) -> None:
        ds = _DummyBlockDataset()
        sample = ds[0]
        expected_sample = {"events": 0}
        assert len(ds) == 3
        assert sample == expected_sample

    def test_iterator_access_dataset_contract(self) -> None:
        ds = _DummyIteratorDataset()
        samples = list(ds)
        expected_samples = [{"events": 0}, {"events": 1}]
        assert samples == expected_samples
        with pytest.raises(StopIteration):
            next(ds)

    def test_reset_default_raises(self) -> None:
        ds = _DummyIteratorDataset()
        with pytest.raises(NotImplementedError):
            ds.reset()

    def test_abstract_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            EventDataset()
        with pytest.raises(TypeError):
            BlockAccessDataset()
        with pytest.raises(TypeError):
            IteratorAccessDataset()
