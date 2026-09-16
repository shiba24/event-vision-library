r"""Compare fixed-count and flow-aligned MVSEC event windows.

Run:
    python examples/readers/mvsec_event_windows.py \
        /data/MVSEC/indoor_flying indoor_flying1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from evlib.dataloaders import MVSECDataLoader
from evlib.vis.view2d import events as render_events
from evlib.vis.view2d import optical_flow as render_flow


NUM_EVENTS = 30_000
OUTPUT_PATH = Path("outputs/mvsec_event_windows.png")


def get_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", help="MVSEC dataset root.")
    parser.add_argument("sequence", help="MVSEC sequence name.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Figure output path.")
    return parser


def main() -> None:
    """Run the MVSEC event-window example."""
    args = get_args_parser().parse_args()

    with MVSECDataLoader(
        args.root,
        args.sequence,
        camera="left",
        event_load_mode="lazy",
        image_load_mode="lazy",
        load_gt_flow_hdf5="lazy",
    ) as reader:
        flow_timestamps = reader.gt_flow_hdf5_timestamps
        if flow_timestamps is None or len(flow_timestamps) < 2:
            raise RuntimeError("The MVSEC sequence does not contain ground-truth flow.")

        flow_index = (len(flow_timestamps) - 1) // 2
        start = float(flow_timestamps[flow_index])
        end = float(flow_timestamps[flow_index + 1])
        window_duration = end - start

        packet_start = max(reader.time_to_index(start) + 1, 0)
        packet_end = min(packet_start + NUM_EVENTS, reader.num_events)
        event_packet = reader.load_events(packet_start, packet_end)
        if len(event_packet) == 0:
            raise RuntimeError("The MVSEC sequence does not contain events.")
        packet_duration = float(event_packet.timestamp[-1] - event_packet.timestamp[0])

        time_window = reader.get_events_by_time(start, end)
        if len(time_window) == 0:
            raise RuntimeError("The MVSEC flow interval contains no events.")

        flow = reader.load_flow_hdf5(flow_index)
        if flow is None:
            raise RuntimeError("The MVSEC flow interval could not be loaded.")

        print(
            f"{reader.sequence} ({reader.camera}): {len(event_packet):,} events over "
            f"{packet_duration:.6f} seconds; {len(time_window):,} events over "
            f"the {window_duration * 1e3:.0f} ms ground-truth flow interval"
        )

        count_image = render_events(event_packet.as_numpy(), reader.IMAGE_SHAPE)
        time_image = render_events(time_window.as_numpy(), reader.IMAGE_SHAPE)
        flow_image, _ = render_flow(flow[1], flow[0], visualize_color_wheel=False)

        fig, (count_axis, time_axis, flow_axis) = plt.subplots(1, 3, figsize=(13.5, 3.8))
        count_axis.imshow(count_image)
        count_axis.set_title(f"fixed count: {len(event_packet):,}\n{packet_duration * 1e3:.0f} ms")
        time_axis.imshow(time_image)
        time_axis.set_title(
            f"flow interval: {window_duration * 1e3:.0f} ms\n{len(time_window):,} events"
        )
        flow_axis.imshow(flow_image)
        flow_axis.set_title("ground-truth optical flow")
        for axis in (count_axis, time_axis, flow_axis):
            axis.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=140)
        plt.close(fig)
        print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
