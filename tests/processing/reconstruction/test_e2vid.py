from evlib.utils import basics as basic_utils
from evlib.processing.reconstruction import E2Vid


def test_run_e2vid_gpu():    # type: ignore
    num_events = 500
    height, width = 20, 40
    events = basic_utils.generate_events(num_events, height, width, 0.1, 0.24)
    reconstructor = E2Vid((height, width), use_gpu=True)
    image = reconstructor(events)
    assert image.shape == (height, width)


def test_run_e2vid_cpu():    # type: ignore
    num_events = 500
    height, width = 20, 40
    events = basic_utils.generate_events(num_events, height, width, 0.1, 0.24)
    reconstructor = E2Vid((height, width), use_gpu=False)
    image = reconstructor(events)
    assert image.shape == (height, width)