"""Tests for the DSEC dataset wrappers."""

from pathlib import Path

import numpy as np
import pytest

from evlib.datasets import BlockAccessDataset
from evlib.datasets import DSECDataset
from evlib.datasets import DSECIterator
from evlib.datasets import EventDataset
from evlib.datasets import IteratorAccessDataset
from evlib.datasets import dsec_collate_fn
from evlib.datasets import event_sample_collate
from evlib.types import RawEvents
from tests.dataloaders.test_dsec import N_FLOW
from tests.dataloaders.test_dsec import N_IMAGES
from tests.dataloaders.test_dsec import SEQ
from tests.dataloaders.test_dsec import _build_cleaned_dsec_tree


@pytest.fixture()
def dsec_dir(tmp_path: Path) -> Path:
    """Build a synthetic DSEC sequence tree."""
    _build_cleaned_dsec_tree(tmp_path)
    return tmp_path


class TestDSECDataset:
    """Tests for map style DSEC dataset access."""

    def test_dataset_basic_sample(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(
            str(dsec_dir),
            SEQ,
            load_flow_forward=True,
            load_images=True,
            load_disparity=True,
            event_load_mode="cached",
        ) as dataset:
            assert len(dataset) == N_FLOW
            assert dataset.sequence == SEQ
            assert dataset.split == "train"
            assert dataset.camera == "left"
            assert isinstance(dataset, BlockAccessDataset)
            assert isinstance(dataset, EventDataset)

            sample = dataset[0]
            t_start, t_end = sample["timestamp"]
            assert t_end > t_start
            assert isinstance(sample["events"], RawEvents)
            assert sample["image_start"] is not None
            assert sample["image_end"] is not None
            assert sample["flow"] is not None
            assert sample["disparity"] is not None

    def test_len_uses_image_intervals_without_flow(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            assert len(dataset) == N_IMAGES - 1

    def test_loader_remains_available(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            assert dataset.loader.num_events > 0
            assert len(dataset.loader.load_events(0, 10)) == 10

    def test_repr_contains_dataset_identity(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            text = repr(dataset)
            assert "DSECDataset" in text
            assert str(dsec_dir) in text
            assert SEQ in text
            assert "train" in text
            assert "left" in text

    def test_event_helpers_delegate_to_loader(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            loader = dataset.loader
            window_start = dataset.index_to_time(0)
            window_end = dataset.index_to_time(5)

            assert dataset.num_events == loader.num_events
            assert dataset.event_load_mode == loader.event_load_mode
            assert len(dataset.load_events(0, 10)) == 10
            assert dataset.time_to_index(window_end) == loader.time_to_index(window_end)
            np.testing.assert_array_equal(
                dataset.get_events_by_time(window_start, window_end).x,
                loader.get_events_by_time(window_start, window_end).x,
            )
            np.testing.assert_array_equal(
                dataset.times_to_indices(np.array([window_start, window_end])),
                loader.times_to_indices(np.array([window_start, window_end])),
            )
            np.testing.assert_array_equal(
                dataset.indices_to_times(np.array([0, 5])),
                loader.indices_to_times(np.array([0, 5])),
            )

    def test_modality_delegations_match_loader(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(
            str(dsec_dir),
            SEQ,
            load_images=True,
            load_flow_forward=True,
            load_disparity=True,
        ) as dataset:
            loader = dataset.loader

            assert dataset.has_images == loader.has_images
            assert dataset.num_images == loader.num_images
            assert dataset.has_flow_forward == loader.has_flow_forward
            assert dataset.num_flow_forward == loader.num_flow_forward
            assert dataset.has_disparity == loader.has_disparity
            assert dataset.num_disparity_frames == loader.num_disparity_frames
            assert dataset.has_calibration == loader.has_calibration
            assert dataset.has_rectify_map == loader.has_rectify_map
            assert dataset.events_prerectified == loader.events_prerectified
            assert dataset.find_nearest_image_index(0.0) == loader.find_nearest_image_index(0.0)
            np.testing.assert_array_equal(dataset.load_image(0), loader.load_image(0))

            dataset_flow, dataset_valid = dataset.load_flow_forward(0)
            loader_flow, loader_valid = loader.load_flow_forward(0)
            np.testing.assert_array_equal(dataset_flow, loader_flow)
            np.testing.assert_array_equal(dataset_valid, loader_valid)

    def test_rectify_events_delegates_to_loader(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True, load_rectify_map=True) as dataset:
            raw_events = dataset.load_events(0, 10)

            rectified = dataset.rectify_events(raw_events)
            expected = dataset.loader.rectify_events(raw_events)

            np.testing.assert_array_equal(rectified.x, expected.x)
            np.testing.assert_array_equal(rectified.y, expected.y)

    def test_optional_modalities_absent_without_loading(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            assert dataset.has_imu is False
            assert dataset.has_lidar is False
            assert dataset.imu_timestamps is None
            assert dataset.lidar_timestamps is None
            assert dataset.num_imu_samples == 0
            assert dataset.num_lidar_scans == 0
            assert dataset.calibration is None


class TestDSECIterator:
    """Tests for iterable DSEC dataset access."""

    def test_iterator_basic_and_reset(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECIterator(str(dsec_dir), SEQ, load_images=True) as iterator:
            first_sample = next(iter(iterator))
            next(iterator)

            iterator.reset()
            reset_sample = next(iterator)

            assert isinstance(first_sample["events"], RawEvents)
            assert first_sample["timestamp"] == reset_sample["timestamp"]

    def test_iterator_stop(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECIterator(str(dsec_dir), SEQ, load_images=True) as iterator:
            for _sample in iterator:
                pass
            with pytest.raises(StopIteration):
                next(iterator)

    def test_iterator_identity_and_type(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECIterator(str(dsec_dir), SEQ, load_images=True) as iterator:
            assert isinstance(iterator, IteratorAccessDataset)
            assert isinstance(iterator, EventDataset)
            assert iterator.sequence == SEQ
            assert iterator.split == "train"
            assert iterator.camera == "left"
            assert "DSECIterator" in repr(iterator)
            assert SEQ in repr(iterator)


class TestDSECCollate:
    """Tests for DSEC collation."""

    def test_collate_aliases_generic_helper(self) -> None:  # noqa: D102
        assert dsec_collate_fn is event_sample_collate

    def test_collate_stacks_interval_timestamps(self, dsec_dir: Path) -> None:  # noqa: D102
        with DSECDataset(str(dsec_dir), SEQ, load_images=True) as dataset:
            batch = dsec_collate_fn([dataset[0], dataset[1]])

        assert batch["timestamp"].shape == (2, 2)
        assert batch["timestamp"].dtype == np.float64
        assert isinstance(batch["events"], list)
        assert isinstance(batch["events"][0], RawEvents)
