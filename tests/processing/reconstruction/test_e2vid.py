from evlib.utils import basics as basic_utils
from evlib.processing.reconstruction import E2Vid

import torch


def test_build_e2vid_shape():    # type: ignore
    num_events = 500
    height, width = 20, 40
    events = basic_utils.generate_events(num_events, height, width, 0.1, 0.24)
    # reconstructor = E2Vid((height, width), use_gpu=torch.cuda.is_available()
    # For now, avoid downloading the pretrained model.
    # image = reconstructor(events)
    # assert image.shape == (height, width)
