import os
import numpy as np

from evlib.codec import fileformat
from evlib.utils import basics as basic_utils
from ...test_utils.misc import generate_random_event_text_file


def test_iterator_bin_event_next_dtype(tmp_path):  # type: ignore
    tmp_file_path = os.path.join(tmp_path, "event.bin")

    # Ideally we want to save it to a .bin file and load it, but the way N-Caltech builds the files is very convoluted
    # We will save a placeholder temporary file instead, and just ignore the loading of it
    evs = basic_utils.generate_events(200, 20, 40, 0.1, 0.3)
    evs.reshape(1, -1).tofile(tmp_file_path)
    iter_bin = fileformat.IteratorBinEvent(tmp_file_path)

    # Instead, ignore the file load and check that the iterator works properly
    iter_bin.raw_evs = evs
    iter_bin.size_x = iter_bin.raw_evs[:, 0].max() + 1
    iter_bin.size_y = iter_bin.raw_evs[:, 1].max() + 1

    for ev in iter_bin:
        assert ev["t"].dtype.type is np.float64
        assert ev["x"].dtype.type is np.int32
        assert ev["y"].dtype.type is np.int32
        assert ev["p"].dtype.type is np.bool_
