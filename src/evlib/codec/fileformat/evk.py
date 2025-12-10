"""evk file formats, mainly for Metavision SDK software by Prophesee.
"""
import logging
from typing import Any
from typing import Dict

import numpy as np
from expelliarmus import Wizard


logger = logging.getLogger(__name__)

from ...types import RawEvents
from ._iterator_access import IteratorAccess


# TODO make parser abstract and merge these classes.


class IteratorEvk3(IteratorAccess):
    FORMAT = "evk3"

    def __init__(self, evk3file: str) -> None:
        super().__init__(evk3file)
        self.wizard = Wizard(encoding="evt3", fpath=self.file_name, chunk_size=16384)
        self.count = 0

    def __next__(self) -> Dict[str, Any]:
        chunk = next(self.wizard.read_chunk())  # type: ignore        
        _l = len(chunk)
        x = np.zeros((_l,), dtype=np.int32)
        y = np.zeros((_l,), dtype=np.int32)
        t = np.zeros((_l,), dtype=np.float64)
        p = np.zeros((_l,), dtype=bool)
        for _i, e in enumerate(chunk):
            t[_i] = np.float64(e[0])
            x[_i] = int(e[1])
            y[_i] = int(e[2])
            p[_i] = int(e[3])
        self.count += _l
        return RawEvents(x=x, y=y, timestamp=t, polarity=p)
