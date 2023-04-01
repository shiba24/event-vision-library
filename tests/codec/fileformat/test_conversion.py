import os

from evlib.codec import fileformat

from evlib.codec import fileformat
from ...test_utils.misc import generate_random_event_text_file


def test_conversion_text_to_hdf5(tmp_path):    # type: ignore
    tmp_file_path = os.path.join(tmp_path, "event.txt")
    ev = generate_random_event_text_file(tmp_file_path, 200, 20, 40, 0.1, 0.3)
    ev_iter = fileformat.IteratorTextEvent(tmp_file_path)
    data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}

    output_file_name = os.path.join(tmp_path, 'event.hdf5')
    fileformat.convert_iterator_access_to_hdf5(ev_iter, output_file_name, data_keys)
