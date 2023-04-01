import os
import numpy as np

from evlib.codec import fileformat
from evlib import constant

TEST_FILE_NAME = os.path.join(constant.ARTIFACTORY_DIR, "sample_data/event.txt")

def test_iterator_text_event_next_dtype() -> None:
    iter_text = fileformat.IteratorTextEvent(TEST_FILE_NAME)

    for ev in iter_text:
        assert ev["t"].dtype.type is np.float64
        assert ev["x"].dtype.type is np.int32
        assert ev["y"].dtype.type is np.int32
        assert ev["p"].dtype.type is np.bool_

