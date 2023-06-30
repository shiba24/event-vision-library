"""2-D visualization of various quantities.
"""
import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


from PIL import Image, ImageDraw

from . import utils

TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)

MIN_DEPTH = 0.1
MAX_DEPTH = 100


def image(image: Any) -> Image.Image:
    """Visualize image.

    Args:
        image (Any): str, np.ndarray, or PIL Image.

    Returns:
        Image.Image: PIL Image object
    """
    image = utils.load_image(image)
    return image


def depth(depth: np.ndarray) -> Image.Image:
    """Visualize depth image.

    Args:
        depth (np.ndarray): depth image.

    Returns:
        Image.Image: PIL Image object, depth
    """
    image = utils.color_depth_with_nan(depth, MIN_DEPTH, MAX_DEPTH, MAX_DEPTH)
    image = Image.fromarray(image)
    return image


def optical_flow(
    flow_v: np.ndarray,
    flow_h: np.ndarray,
    visualize_color_wheel: bool = True,
    ord: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Visualize optical flow.
    
    Args:
        flow_v (numpy.ndarray): [H x W], vertical direction.
        flow_h (numpy.ndarray): [H x W], horizontal direction.
        visualize_color_wheel (bool): If True, it also visualizes the color wheel (legend for OF).
        file_prefix (Optional[str], optional): [description]. Defaults to None.
            If specified, the save location will be `save_dir/{prefix}_{unique}.png`.

    Returns:
        image (PIL.Image): PIL image.
    """
    flow_rgb, color_wheel, _ = utils.color_optical_flow(flow_v, flow_h, ord=ord)
    image = Image.fromarray(flow_rgb)

    if visualize_color_wheel:
        wheel = Image.fromarray(color_wheel)
    else:
        wheel = None
    return image, wheel


def events(events: np.ndarray, image_shape: tuple) -> Image.Image:
    """Visualize events with polarity color.

    Args:
        events (np.ndarray): _description_
        image_shape (tuple): _description_

    Returns:
        _type_: _description_
    """
    background_color = 255
    image = (
        np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        * background_color
    )  # RGBA channel

    events[:, 0] = np.clip(events[:, 0], 0, image_shape[0] - 1)
    events[:, 1] = np.clip(events[:, 1], 0, image_shape[1] - 1)
    colors = np.array([(255, 0, 0) if e[3] == 1 else (0, 0, 255) for e in events])
    image[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32), :] = colors

    image = Image.fromarray(image)
    return image
