"""Dataset base classes for event camera datasets."""

from ._base import BlockAccessDataset
from ._base import BlockDatasetIterator
from ._base import EventDataset
from ._base import IteratorAccessDataset
from ._base import event_sample_collate
from .dsec import DSECDataset
from .dsec import DSECIterator
from .dsec import dsec_collate_fn
from .ecd import ECD_DAVIS240C_SENSOR_RESOLUTION
from .ecd import ECD_REAL_SEQUENCES
from .ecd import ECD_SEQUENCES
from .ecd import ECD_SYNTHETIC_SEQUENCES
from .ecd import ECDDataset
from .ecd import ECDIterator
from .ecd import ecd_collate_fn
from .mvsec import MVSECDataset
from .mvsec import MVSECIterator
from .mvsec import mvsec_collate_fn


__all__ = [
    "BlockAccessDataset",
    "BlockDatasetIterator",
    "DSECDataset",
    "DSECIterator",
    "ECDDataset",
    "ECDIterator",
    "ECD_DAVIS240C_SENSOR_RESOLUTION",
    "ECD_REAL_SEQUENCES",
    "ECD_SEQUENCES",
    "ECD_SYNTHETIC_SEQUENCES",
    "EventDataset",
    "IteratorAccessDataset",
    "dsec_collate_fn",
    "ecd_collate_fn",
    "event_sample_collate",
    "MVSECDataset",
    "MVSECIterator",
    "mvsec_collate_fn",
]
