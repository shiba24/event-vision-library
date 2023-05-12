"""Hot-pixel removal filters.
"""
import numpy as np
import torch

from ..types import NUMPY_TORCH



def hot_pixel_filter(events: NUMPY_TORCH, image_shape: tuple, hot_pixel: int = 10) -> NUMPY_TORCH:
    """Apply hot-pixel removal, a speed-optimized implementation.
        The filtering is for a single batch of events, not multiple or asynchronous.

    Args:
        events (NUMPY_TORCH): [N, 4]
        image_shape (tuple): tuple of (H, W).
        hot_pixel (int): threshold of hot pixels.

    Returns:
        NUMPY_TORCH: Filered events
    """
    xy = events[..., :2]
    if isinstance(events, np.ndarray):
        pix_index = (xy[..., 0] + xy[..., 1] * image_shape[0]).astype(np.int64)   # unique indices at all pixels
        val, counts = np.unique(pix_index, return_counts=True)
        dup_vals = val[counts > hot_pixel]  # the unique index for hot pixel locations
        hot_event_index = np.in1d(pix_index, dup_vals)  # bool of the events belong to the hot pixels
    elif isinstance(events, torch.Tensor):
        pix_index = (xy[..., 0] + xy[..., 1] * image_shape[0]).long()   # unique indices at all pixels
        val, counts = torch.unique(pix_index, return_counts=True)
        dup_vals = val[counts > hot_pixel]  # the unique index for hot pixel locations
        hot_event_index = torch.isin(pix_index, dup_vals)  # bool of the events belong to the hot pixels
    filtered_event = events[~hot_event_index]
    return filtered_event

