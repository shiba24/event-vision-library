import os
import numpy as np

from evlib.utils import basics as basic_utils
from evlib.preproc import hot_pixel



def test_hot_pixel_none():
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    hot_pixel_filtered = hot_pixel.hot_pixel_filter(ev, (height, width), 500)
    assert len(ev) == len(hot_pixel_filtered)


def test_hot_pixel_all():
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    hot_pixel_filtered = hot_pixel.hot_pixel_filter(ev, (height, width), 0)
    assert len(hot_pixel_filtered) == 0

