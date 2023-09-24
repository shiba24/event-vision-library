"""Bin format.
Used in N-Caltech101, N-cars.
"""
import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

from ...types import RawEvents
from ._iterator_access import IteratorAccess


# TODO make parser abstract and merge these classes.


class IteratorBin(IteratorAccess):
    FORMAT = "bin"

    def __init__(self, binfile: str) -> None:
        # TODO add parse format option
        super().__init__(binfile)
        # Variables
        self.raw_evs = np.zeros((4, 1))
        self.size_x = 0
        self.size_y = 0
        # Initialize variables
        self.file = open(self.file_name, 'rb')
        self._init_vars_()

    def __iter__(self) -> Any:
        self.count = 0
        return self

    def _init_vars_(self) -> None:
        # Read event file
        self.raw_evs = self.read_ev_file()
        # Estimate width and height of sensor given the events in the stream
        self.size_x = self.raw_evs[:, 0].max() + 1
        self.size_y = self.raw_evs[:, 1].max() + 1

    def read_ev_file(self) -> np.ndarray:
        # From https://github.com/gorchard/event-Python/blob/master/eventvision.py#L532
        # Change np.fromfile() to use "offset" if needed for future datasets
        raw_data = np.fromfile(self.file, dtype=np.uint8)
        self.file.close()
        raw_data = np.uint32(raw_data)
        raw_evs = self._transform_raw_to_evs_(raw_data)
        return raw_evs

    @staticmethod
    def _transform_raw_to_evs_(raw_data: np.ndarray) -> np.ndarray:
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        x_ = all_x[td_indices]
        y_ = all_y[td_indices]
        t_ = all_ts[td_indices]
        p_ = all_p[td_indices]

        raw_evs = np.vstack([x_, y_, t_, p_]).T
        return raw_evs


class IteratorBinEvent(IteratorBin):
    def __next__(self) -> RawEvents:
        """
        Returns:
            RawEvents: events
        """
        raw_evs = self.raw_evs
        _l = len(raw_evs)

        # As we are reading the whole file at once this works. Re-check if we decide to go for chunk reading.
        if self.count >= _l:
            raise StopIteration

        x = raw_evs[:, 0].astype(np.int32)
        y = raw_evs[:, 1].astype(np.int32)
        t = raw_evs[:, 2].astype(np.float64)
        p = raw_evs[:, 3].astype(bool)

        self.count += _l
        return RawEvents(x=x, y=y, timestamp=t, polarity=p)
