# Data loading examples

This directory contains examples for loading DSEC, ECD/DAVIS, and MVSEC
recordings. Each script prints a summary and saves a visualization.

`evlib.datasets` provides frame-aligned PyTorch datasets. `evlib.dataloaders`
provides low-level readers for custom event windows.

| Example                            | Demonstrates                                     |
| ---------------------------------- | ------------------------------------------------ |
| `datasets/dsec_flow_batch.py`      | DSEC optical-flow samples and PyTorch batching   |
| `datasets/ecd_frame_activity.py`   | Event activity across ECD frame intervals        |
| `readers/ecd_event_count_image.py` | A fixed-count ECD packet and event count image   |
| `readers/mvsec_event_windows.py`   | Fixed-count and flow-aligned MVSEC event windows |

## Run the examples

```console
python examples/datasets/dsec_flow_batch.py \
    /path/to/DSEC zurich_city_01_a

python examples/datasets/ecd_frame_activity.py \
    /path/to/ECD/datasets/davis shapes_rotation

python examples/readers/ecd_event_count_image.py \
    /path/to/ECD/datasets/davis/shapes_rotation

python examples/readers/mvsec_event_windows.py \
    /path/to/MVSEC/indoor_flying indoor_flying1
```

The scripts save figures under `outputs/`. Use `--output` to change the path.
Other settings are constants near the top of each script. The ECD count-image
example accepts `--sensor-resolution HEIGHT WIDTH` for recordings without frames.
The MVSEC example requires the sequence's data and ground-truth HDF5 files.

The examples cover data loading and visual checks, not model training or
evaluation.
