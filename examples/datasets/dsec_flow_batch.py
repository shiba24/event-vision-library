r"""Load and inspect a batch of synchronized DSEC optical-flow samples.

Run:
    python examples/datasets/dsec_flow_batch.py \
        /data/DSEC zurich_city_01_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from evlib.datasets import DSECDataset
from evlib.datasets import dsec_collate_fn
from evlib.vis.view2d import events as render_events
from evlib.vis.view2d import optical_flow as render_flow


BATCH_SIZE = 4
NUM_WORKERS = 2
OUTPUT_PATH = Path("outputs/dsec_flow_batch.png")


def get_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", help="DSEC dataset root.")
    parser.add_argument("sequence", help="DSEC sequence name.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Figure output path.")
    return parser


def flow_to_image(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Render optical flow as an RGB image."""
    valid_flow = np.where(valid[..., None], flow, 0.0)
    image, _ = render_flow(
        valid_flow[..., 1],
        valid_flow[..., 0],
        visualize_color_wheel=False,
    )
    return np.asarray(image)


def main() -> None:
    """Run the DSEC batching example."""
    args = get_args_parser().parse_args()

    with DSECDataset(
        args.root,
        args.sequence,
        split="train",
        camera="left",
        load_images="lazy",
        load_flow_forward="lazy",
        load_rectify_map=False,
        event_load_mode="lazy",
    ) as dataset:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=dsec_collate_fn,
        )
        batch = next(iter(loader))

        events = batch["events"][0]
        image = batch["image_start"][0]
        flow_sample = batch["flow"][0]
        if len(events) == 0:
            raise RuntimeError("The first DSEC sample contains no events.")
        if image is None or flow_sample is None:
            raise RuntimeError("The DSEC sequence does not contain images and flow.")
        flow, valid = flow_sample
        start, end = batch["timestamp"][0]
        event_counts = [len(sample_events) for sample_events in batch["events"]]

        print(
            f"{dataset.sequence} ({dataset.camera}): "
            f"{len(dataset):,} samples, {dataset.num_events:,} events"
        )
        print(f"batch size: {len(event_counts)}, event counts: {event_counts}")

        fig, (events_axis, image_axis, flow_axis) = plt.subplots(1, 3, figsize=(13.2, 3.8))
        events_axis.imshow(render_events(events.as_numpy(), dataset.EVENT_SHAPE))
        events_axis.set_title(f"raw events (distorted), {end - start:.3f} s")
        image_axis.imshow(image, cmap="gray")
        image_axis.set_title("start image (rectified)")
        flow_axis.imshow(flow_to_image(flow, valid))
        flow_axis.set_title("forward flow (rectified view)")
        for axis in (events_axis, image_axis, flow_axis):
            axis.axis("off")

        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=140)
        plt.close(fig)
        print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
