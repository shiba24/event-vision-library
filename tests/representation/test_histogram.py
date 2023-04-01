import os
import numpy as np

from evlib.utils import basics as basic_utils
from evlib.representation import Histogram


def test_generate_events():    # type: ignore
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    hist_builder = Histogram((height, width), use_polarity=True)
    hist = hist_builder(ev)
    assert hist.shape == (height, width)
