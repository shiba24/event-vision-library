import os
import numpy as np

from evlib.codec import fileformat
from ...test_utils.misc import generate_random_event_text_file


def test_iterator_text_event_rw_consistency(tmp_path):    # type: ignore
    """Read and write consistency."""
    # Write
    tmp_file_path = os.path.join(tmp_path, "event.txt")
    ev = generate_random_event_text_file(tmp_file_path, 200, 20, 40, 0.1, 0.3)

    iter_text = fileformat.IteratorTextEvent(tmp_file_path)
    read_ev = []
    for it in iter_text:
        it_ev = np.stack([it["y"], it["x"], it["t"], it["p"]]).T  # n, 4
        read_ev.append(it_ev)
    assert np.allclose(ev, np.concatenate(read_ev, axis=0))


def test_iterator_text_event_next_dtype(tmp_path):  # type: ignore
    tmp_file_path = os.path.join(tmp_path, "event.txt")
    ev = generate_random_event_text_file(tmp_file_path, 200, 20, 40, 0.1, 0.3)
    iter_text = fileformat.IteratorTextEvent(tmp_file_path)

    for ev in iter_text:
        assert ev["t"].dtype.type is np.float64
        assert ev["x"].dtype.type is np.int32
        assert ev["y"].dtype.type is np.int32
        assert ev["p"].dtype.type is np.bool_
