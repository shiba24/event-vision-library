# Loading event data

`evlib.datasets` provides frame-aligned datasets for PyTorch.
`evlib.dataloaders` provides low-level readers for arbitrary event windows. Use
`torch.utils.data.DataLoader` to batch a dataset.

## PyTorch datasets

Map-style datasets support indexed access, shuffling, sampling, and worker
processes. Each sample is a dictionary containing `RawEvents`, timestamps, and
requested aligned data.

```python
from torch.utils.data import DataLoader

from evlib.datasets import DSECDataset
from evlib.datasets import dsec_collate_fn

with DSECDataset(
    "/data/DSEC",
    "zurich_city_01_a",
    camera="left",
    load_images="lazy",
    load_flow_forward="lazy",
) as dataset:
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        collate_fn=dsec_collate_fn,
    )
    batch = next(iter(loader))
```

The collate functions stack `timestamp` as a NumPy array and keep other fields
as lists, including variable-length events and `None` values. Apply
task-specific preprocessing after loading.

`ECDDataset` and `MVSECDataset` use the same interface and provide
`ecd_collate_fn` and `mvsec_collate_fn`.

## Loading modes

Optional modalities use a shared loading convention:

| Value              | Behavior                   |
| ------------------ | -------------------------- |
| `False`            | Do not load it.            |
| `True` or `"lazy"` | Load it when requested.    |
| `"cached"`         | Load it at initialization. |

Use lazy loading for exploration or worker-based access. Use cached loading for
repeated passes when the data fits in memory.

## Arbitrary event windows

Use a reader for windows that do not follow frame boundaries:

```python
from evlib.dataloaders import MVSECDataLoader

with MVSECDataLoader(
    "/data/MVSEC/indoor_flying",
    "indoor_flying1",
    event_load_mode="lazy",
    image_load_mode="lazy",
) as reader:
    packet = reader.load_events(0, 30_000)
    start = reader.index_to_time(0)
    window = reader.get_events_by_time(start, start + 0.02)
```

Index and time ranges are half-open: `[start, end)`. `RawEvents` exposes `x`,
`y`, `timestamp`, and `polarity` arrays. `RawEvents.as_numpy()` returns an
array of shape `(N, 4)` with columns `[y, x, t, p]`.

## Sequential iteration

`ECDIterator`, `DSECIterator`, and `MVSECIterator` iterate through frame-aligned
samples in sequence order:

```python
from evlib.datasets import ECDIterator

with ECDIterator("/data/ECD/datasets/davis", "shapes_rotation") as stream:
    for sample in stream:
        events = sample["events"]
```

These iterators do not shard across worker processes. Use `num_workers=0` with
a PyTorch `DataLoader`, or pass the map-style dataset directly when using
workers or shuffling.

Use datasets and readers as context managers to close file handles. See the
[`examples` directory][examples] for runnable workflows and the
[API reference](./reference.md) for all options.

[examples]: https://github.com/shiba24/event-vision-library/tree/main/examples
