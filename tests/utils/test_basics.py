import os
import numpy as np

from evlib.utils import basics as basic_utils

def test_generate_events(tmp_path):    # type: ignore
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    assert len(ev) == ne
    assert ev[:, 0].max() == height - 1
    assert ev[:, 1].max() == width - 1
    assert np.allclose(np.unique(ev[:, 3]), np.array([0, 1]))
