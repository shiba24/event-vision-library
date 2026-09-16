r"""Visualize a fixed-count ECD event packet and event-count image.

Run:
    python examples/readers/ecd_event_count_image.py \
        /data/ECD/datasets/davis/shapes_rotation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from evlib.dataloaders import DavisRecordingLoader
from evlib.representation import Histogram
from evlib.vis.view2d import events as render_events


NUM_EVENTS = 30_000
OUTPUT_PATH = Path("outputs/ecd_event_count_image.png")


def get_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording", help="ECD/DAVIS recording directory.")
    parser.add_argument(
        "--sensor-resolution",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Sensor size for recordings without frames.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Figure output path.")
    return parser


def main() -> None:
    """Run the ECD event-count-image example."""
    args = get_args_parser().parse_args()
    requested_resolution = None if args.sensor_resolution is None else tuple(args.sensor_resolution)

    with DavisRecordingLoader(
        args.recording,
        depth_load_mode=False,
        sensor_resolution=requested_resolution,
    ) as reader:
        sensor_resolution = reader.sensor_resolution
        if sensor_resolution is None:
            raise RuntimeError(
                "The recording has no frames to infer the sensor size from. "
                "Pass --sensor-resolution HEIGHT WIDTH."
            )

        events = reader.load_events(0, min(NUM_EVENTS, reader.num_events))
        if len(events) == 0:
            raise RuntimeError("The event window is empty.")

        start = float(events.timestamp[0])
        end = float(events.timestamp[-1])
        count_image = Histogram(sensor_resolution, use_polarity=False)(events.as_numpy())

        print(
            f"{reader.sequence}: {len(events):,} of {reader.num_events:,} events "
            f"over {end - start:.6f} seconds"
        )

        fig, (events_axis, count_axis) = plt.subplots(1, 2, figsize=(9.5, 3.8))
        events_axis.imshow(render_events(events.as_numpy(), sensor_resolution))
        events_axis.set_title(f"{len(events):,} events")
        count_axis.imshow(count_image, cmap="magma")
        count_axis.set_title("event count image")
        for axis in (events_axis, count_axis):
            axis.axis("off")

        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=140)
        plt.close(fig)
        print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
