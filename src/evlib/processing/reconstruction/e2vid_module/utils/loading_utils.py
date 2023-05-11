from typing import Any
import logging

import torch
from ..model.model import *

logger = logging.getLogger('ev_lib')


def load_model(path_to_model: str) -> Any:
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model, map_location=torch.device('cpu'))  # TODO handle CPU / GPU switch here
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])
    return model


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    logger.info(f'Device: {device}')

    return device
