

import os
import numpy as np

from evlib import types
from evlib.utils import basics as basic_utils

def test_raw_events_properties():    # type: ignore
    ne = 20
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    e = types.RawEvents(y=ev[:, 0], x=ev[:, 1], timestamp=ev[:, 2], polarity=ev[:, 3])
    assert len(e) == ne
    np.testing.assert_allclose(e[5], ev[5])
    np.testing.assert_allclose(e.as_numpy(), ev)
