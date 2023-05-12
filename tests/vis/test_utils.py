import numpy as np

from evlib.utils import basics as basic_utils
from evlib.vis import utils as vis_utils


def test_color_optical_flow():  # type: ignore
    imshape = (100, 50)
    maxval = 10.
    flow = basic_utils.generate_random_optical_flow(imshape, maxval)
    flow_rgb, wheel, max_magnitude = vis_utils.color_optical_flow(flow[0], flow[1])
    assert flow_rgb.shape == imshape + (3, )

