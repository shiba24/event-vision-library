import os
import numpy as np

from evlib.utils import basics as basic_utils


def test_import_torch():  # type: ignore
    import torch


def test_generate_events(tmp_path):    # type: ignore
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    assert len(ev) == ne
    assert ev[:, 0].max() == height - 1
    assert ev[:, 1].max() == width - 1
    assert np.allclose(np.unique(ev[:, 3]), np.array([0, 1]))


def test_generate_random_optical_flow(): # type: ignore
    imsize = (100, 50)
    max_val = 10.
    flow = basic_utils.generate_random_optical_flow(imsize, max_val)
    assert flow.min() >= -max_val
    assert flow.max() <= max_val


def test_generate_uniform_optical_flow(): # type: ignore
    imsize = (100, 50)
    flow_v = 10.
    flow_h = -8.
    flow = basic_utils.generate_uniform_optical_flow(imsize, flow_v, flow_h)
    assert np.allclose(flow[0], flow_v)
    assert np.allclose(flow[1], flow_h)
