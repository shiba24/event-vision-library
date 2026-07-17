# noqa: D100
import numpy as np

from evlib import types
from evlib.utils import basics as basic_utils


def test_raw_events_properties():  # type: ignore  # noqa: D103
    ne = 20
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    e = types.RawEvents(
        y=ev[:, 0],
        x=ev[:, 1],
        timestamp=ev[:, 2],
        polarity=ev[:, 3],
    )
    assert len(e) == ne
    np.testing.assert_allclose(e[5], ev[5])
    np.testing.assert_allclose(e.as_numpy(), ev)


def test_raw_events_append_preserves_polarity():  # type: ignore  # noqa: D103
    events = types.RawEvents(
        x=np.array([10, 11], dtype=np.int16),
        y=np.array([20, 21], dtype=np.int16),
        timestamp=np.array([0.1, 0.2], dtype=np.float64),
        polarity=np.array([True, False], dtype=np.bool_),
    )
    event = types.RawEvent(
        x=np.int16(12),
        y=np.int16(22),
        timestamp=np.float64(0.3),
        polarity=True,
    )

    events.append(event)

    np.testing.assert_array_equal(events.x, np.array([10, 11, 12], dtype=np.int16))
    np.testing.assert_array_equal(events.y, np.array([20, 21, 22], dtype=np.int16))
    np.testing.assert_allclose(events.timestamp, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(events.polarity, np.array([True, False, True]))


def test_raw_events_slice_returns_raw_events():  # type: ignore  # noqa: D103
    events = types.RawEvents(
        x=np.arange(20, dtype=np.int16),
        y=np.arange(100, 120, dtype=np.int16),
        timestamp=np.arange(20, dtype=np.float64) / 10.0,
        polarity=np.arange(20) % 2 == 0,
    )

    sliced = events[10:20]

    assert isinstance(sliced, types.RawEvents)
    np.testing.assert_array_equal(sliced.x, events.x[10:20])
    np.testing.assert_array_equal(sliced.y, events.y[10:20])
    np.testing.assert_allclose(sliced.timestamp, events.timestamp[10:20])
    np.testing.assert_array_equal(sliced.polarity, events.polarity[10:20])
