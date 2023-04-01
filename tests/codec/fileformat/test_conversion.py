import os

from evlib.codec import fileformat
from evlib import constant

TEST_FILE_NAME = os.path.join(constant.ARTIFACTORY_DIR, "sample_data/event.txt")

def test_conversion_text_to_hdf5(tmp_path):    # type: ignore
    ev_iter = fileformat.IteratorTextEvent(TEST_FILE_NAME)
    data_keys = {"x": "raw_events/x", "y": "raw_events/y", "t": "raw_events/t", "p": "raw_events/p"}
    output_file_name = os.path.join(tmp_path, 'event.hdf5')
    fileformat.convert_iterator_access_to_hdf5(ev_iter, output_file_name, data_keys)
