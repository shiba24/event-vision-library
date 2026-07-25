"""Tests for dataset base classes."""

import pytest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from evlib.datasets._base import BlockAccessDataset
from evlib.datasets._base import BlockDatasetIterator
from evlib.datasets._base import EventDataset
from evlib.datasets._base import IteratorAccessDataset


def _events_only(batch: list) -> list:
    return [sample["events"] for sample in batch]


class _DummyEventDataset(EventDataset):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DummyBlockDataset(BlockAccessDataset):
    def __init__(self, num_samples: int = 3) -> None:
        self.closed = False
        self._num_samples = num_samples

    def __getitem__(self, index: int) -> dict:
        return {"events": index}

    def __len__(self) -> int:
        return self._num_samples

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


class TestDatasetBaseClasses:  # noqa: D101
    def test_event_dataset_context_manager_closes(self) -> None:  # noqa: D102
        ds = _DummyEventDataset()
        with ds:
            assert ds.closed is False
        assert ds.closed is True

    def test_block_access_dataset_contract(self) -> None:  # noqa: D102
        ds = _DummyBlockDataset()
        sample = ds[0]
        expected_sample = {"events": 0}
        assert len(ds) == 3
        assert sample == expected_sample

    def test_iterator_access_dataset_contract(self) -> None:  # noqa: D102
        ds = _DummyIteratorDataset()
        samples = list(ds)
        expected_samples = [{"events": 0}, {"events": 1}]
        assert samples == expected_samples
        with pytest.raises(StopIteration):
            next(ds)

    def test_base_classes_have_expected_torch_types(self) -> None:  # noqa: D102
        assert isinstance(_DummyBlockDataset(), TorchDataset)
        assert isinstance(_DummyIteratorDataset(), TorchIterableDataset)

    def test_reset_default_raises(self) -> None:  # noqa: D102
        ds = _DummyIteratorDataset()
        with pytest.raises(NotImplementedError):
            ds.reset()

    def test_abstract_base_cannot_be_instantiated(self) -> None:  # noqa: D102
        with pytest.raises(TypeError):
            EventDataset()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            BlockAccessDataset()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            IteratorAccessDataset()  # type: ignore[abstract]

    def test_block_access_dataset_feeds_torch_dataloader(self) -> None:  # noqa: D102
        loader = DataLoader(_DummyBlockDataset(), batch_size=2, collate_fn=_events_only)
        expected_batches = [[0, 1], [2]]
        assert list(loader) == expected_batches


class TestBlockDatasetIterator:  # noqa: D101
    def test_yields_every_sample_in_order(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset())
        expected_samples = [{"events": 0}, {"events": 1}, {"events": 2}]
        assert list(iterator) == expected_samples

    def test_exhausted_iterator_raises_stop_iteration(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset())
        for _ in iterator:
            pass
        with pytest.raises(StopIteration):
            next(iterator)

    def test_empty_dataset_stops_immediately(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset(num_samples=0))
        assert list(iterator) == []

    def test_reset_replays_from_first_sample(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset())
        first_sample = next(iterator)
        next(iterator)

        iterator.reset()
        replayed_sample = next(iterator)

        assert first_sample == replayed_sample == {"events": 0}

    def test_iter_starts_a_new_pass(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset())
        next(iterator)
        next(iterator)

        restarted_sample = next(iter(iterator))

        assert restarted_sample == {"events": 0}

    def test_worker_process_iteration_raises(  # noqa: D102
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evlib.datasets._base.get_worker_info", object)
        iterator = BlockDatasetIterator(_DummyBlockDataset())

        with pytest.raises(RuntimeError, match="each worker would repeat the full dataset"):
            iter(iterator)

    def test_dataset_property_exposes_wrapped_dataset(self) -> None:  # noqa: D102
        dataset = _DummyBlockDataset()
        assert BlockDatasetIterator(dataset).dataset is dataset

    def test_close_closes_wrapped_dataset(self) -> None:  # noqa: D102
        dataset = _DummyBlockDataset()
        iterator = BlockDatasetIterator(dataset)

        iterator.close()

        assert dataset.closed is True

    def test_context_manager_closes_wrapped_dataset(self) -> None:  # noqa: D102
        dataset = _DummyBlockDataset()
        with BlockDatasetIterator(dataset) as iterator:
            assert next(iterator) == {"events": 0}
        assert dataset.closed is True

    def test_torch_dataloader_rereads_every_epoch(self) -> None:  # noqa: D102
        iterator = BlockDatasetIterator(_DummyBlockDataset())
        loader = DataLoader(iterator, batch_size=2, collate_fn=_events_only)
        expected_batches = [[0, 1], [2]]

        first_epoch = list(loader)
        second_epoch = list(loader)

        assert first_epoch == second_epoch == expected_batches
