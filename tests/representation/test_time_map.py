from evlib.utils import basics as basic_utils
from evlib.representation import TimeMap


def test_build_time_map_shape():    # type: ignore
    ne = 500
    height, width = 20, 40
    ev = basic_utils.generate_events(ne, height, width, 0.1, 0.24)
    time_map_builder = TimeMap((height, width), decay=0.01)
    time_map = time_map_builder(ev)
    assert time_map.shape == (height, width)
