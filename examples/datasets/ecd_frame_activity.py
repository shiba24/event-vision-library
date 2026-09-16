r"""Plot event activity across ECD frame intervals.

Run:
    python examples/datasets/ecd_frame_activity.py \
        /data/ECD/datasets/davis shapes_rotation
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evlib.datasets import ECDIterator


MAX_SAMPLES = 200
OUTPUT_PATH = Path("outputs/ecd_frame_activity.png")


def get_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", help="ECD dataset root.")
    parser.add_argument("sequence", help="ECD sequence name.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Figure output path.")
    return parser


def main() -> None:
    """Run the ECD frame-activity example."""
    args = get_args_parser().parse_args()

    with ECDIterator(
        args.root,
        args.sequence,
        load_imu=False,
        load_gt_pose=False,
        depth_load_mode=False,
    ) as iterator:
        stream = itertools.islice(iterator, MAX_SAMPLES)
        timestamps = []
        event_counts = []
        preview_image = None
        preview_count = -1
        for sample in stream:
            timestamps.append(sample["timestamp"])
            event_count = len(sample["events"])
            event_counts.append(event_count)
            if event_count > preview_count and sample["image"] is not None:
                preview_image = sample["image"]
                preview_count = event_count
        if not event_counts:
            raise RuntimeError("The ECD sequence has no frame intervals.")
        if preview_image is None:
            raise RuntimeError("The ECD sequence does not contain images.")

        print(f"{iterator.sequence}: inspected {len(event_counts):,} frame intervals")
        print(f"events per interval: min={min(event_counts):,} max={max(event_counts):,}")

    relative_time = np.asarray(timestamps, dtype=np.float64)
    relative_time -= relative_time[0]

    fig, (count_axis, image_axis) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    count_axis.plot(relative_time, event_counts, linewidth=1.0, marker=".", markersize=2)
    count_axis.set_xlabel("time from first interval (s)")
    count_axis.set_ylabel("events per interval")
    count_axis.set_title(f"{args.sequence} event activity")
    count_axis.grid(alpha=0.3)
    image_axis.imshow(preview_image, cmap="gray")
    image_axis.set_title(f"image at highest-activity interval ({preview_count:,} events)")
    image_axis.axis("off")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
