"""Custom data types."""

from typing import Tuple, Union

import numpy as np
import torch

from .raw_event import RawEvent
from .raw_events import RawEvents

NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]
