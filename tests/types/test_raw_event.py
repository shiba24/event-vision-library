

import os
import numpy as np

from evlib import types
from evlib.utils import basics as basic_utils



def test_raw_event_properties():    # type: ignore
    ne = 5
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)[0]
    e = types.RawEvent(y=ev[0], x=ev[1], timestamp=ev[2], polarity=ev[3])
    assert ev[0] == e.y
    assert ev[1] == e.x
    assert ev[2] == e.t == e.timestamp
    assert ev[3] == e.p == e.polarity
    np.testing.assert_allclose(e.as_numpy(), ev)
