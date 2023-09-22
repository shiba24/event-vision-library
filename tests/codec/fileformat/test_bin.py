import numpy as np

from evlib.codec import fileformat
from evlib.utils import basics as basic_utils


def test_iterator_bin_event_next_dtype():  # type: ignore
    # Ideally we want to save it to a .bin file and load it, but the way N-Caltech builds the files is very convoluted
    evs = basic_utils.generate_events(200, 20, 40, 0.1, 0.3)
    iter_bin = fileformat.IteratorBinEvent(None)

    # Instead, ignore the file load and check that the iterator works properly
    iter_bin.raw_evs = evs
    iter_bin.size_x = iter_bin.raw_evs[:, 0].max() + 1
    iter_bin.size_y = iter_bin.raw_evs[:, 1].max() + 1

    for ev in iter_bin:
        assert ev["t"].dtype.type is np.float64
        assert ev["x"].dtype.type is np.int32
        assert ev["y"].dtype.type is np.int32
        assert ev["p"].dtype.type is np.bool_
