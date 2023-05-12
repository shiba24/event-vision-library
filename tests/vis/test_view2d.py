import numpy as np

from evlib.utils import basics as basic_utils
from evlib.vis import view2d


def test_optical_flow():  # type: ignore
    imshape = (100, 50)
    flow = basic_utils.generate_random_optical_flow(imshape, 10.)
    _, wheel = view2d.optical_flow(flow[0], flow[1], visualize_color_wheel=True)
    assert wheel is not None

    _, wheel = view2d.optical_flow(flow[0], flow[1], visualize_color_wheel=False)
    assert wheel is None
