import glob
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt


from typing import Any, Dict, List, Optional
from PIL import Image


def load_image(image: Any) -> Image.Image:
    """A wrapper function to get image and returns PIL Image object.

    Args:
        image (str or np.ndarray): If it is str, open and load the image.
        If it is numpy array, it converts to PIL.Image.

    Returns:
        Image.Image: PIl Image object.
    """
    if type(image) == str:
        image = Image.open(image)
    elif type(image) == np.ndarray:
        image = Image.fromarray(image)
    return image


def color_depth_with_nan(
    depth: np.ndarray, min_depth: float, max_depth: float, fillnan: float
) -> np.ndarray:
    """Color depth image even with nan values.
    The nan will be padded with `fillnan` value.
    Args:
        depth (np.ndarray) ... [1, H, W]?
    Returns:
        (np.ndarray) ... [H, W, 3]?
    """
    depth[np.isnan(depth)] = fillnan
    return color_depth(depth, min_depth, max_depth)

def color_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Color depth image.
    Args:
        depth (np.ndarray) ... [1, H, W]?  # TODO test and check
    Returns:
        (np.ndarray) ... [H, W, 3]?
    """
    depth = np.clip(depth, min_depth, max_depth)
    depth_relative = (depth - min_depth) / (max_depth - min_depth)
    return (255 * plt.cm.viridis(depth_relative.astype(np.float32))[0, :, :, :3]).astype(
        np.uint8
    )
    
    
    
def color_optical_flow(
    flow_x: np.ndarray, flow_y: np.ndarray, max_magnitude=None, ord=1.0
):
    """Color optical flow.
    Args:
        flow_x (numpy.ndarray) ... [H x W], height direction.
        flow_y (numpy.ndarray) ... [H x W], width direction.
        max_magnitude (float, optional) ... Max magnitude used for the colorization. Defaults to None.
        ord (float) ... 1: our usual, 0.5: DSEC colorinzing.

    Returns:
        flow_rgb (np.ndarray) ... [W, H]
        color_wheel (np.ndarray) ... [H, H] color wheel
        max_magnitude (float) ... max magnitude of the flow.
    """
    flows = np.stack((flow_x, flow_y), axis=2)
    flows[np.isinf(flows)] = 0
    flows[np.isnan(flows)] = 0
    mag = np.linalg.norm(flows, axis=2) ** ord
    ang = (np.arctan2(flow_y, flow_x) + np.pi) * 180.0 / np.pi / 2.0
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    if max_magnitude is None:
        max_magnitude = mag.max()
    hsv[:, :, 2] = (255 * mag / max_magnitude).astype(np.uint8)
    # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Color wheel
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[0], 3], dtype=np.uint8)
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, flow_x.shape[0]), np.linspace(-1, 1, flow_x.shape[0])
    )
    mag = np.linalg.norm(np.stack((xx, yy), axis=2), axis=2)
    # ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
    ang = (np.arctan2(xx, yy) + np.pi) * 180 / np.pi / 2.0
    hsv[:, :, 0] = ang.astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (255 * mag / mag.max()).astype(np.uint8)
    # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    color_wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_rgb, color_wheel, max_magnitude