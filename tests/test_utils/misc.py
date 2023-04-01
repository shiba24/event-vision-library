import numpy as np

from evlib.codec import fileformat
from evlib.utils import basics as basic_utils


def generate_random_event_text_file(filename: str, n_events: int, height: int, width: int, tmin: float, tmax: float) -> np.ndarray:
    ev = basic_utils.generate_events(n_events, height, width, tmin, tmax)
    fileformat.conversion.write_to_text(ev, filename)
    return ev
