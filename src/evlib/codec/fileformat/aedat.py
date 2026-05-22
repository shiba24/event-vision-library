"""aedat file formats, mainly for DV software by iniVation."""

import logging
from typing import Any
from typing import Dict

from ...types import RawEvents
from ._aedat4 import Aedat4Reader
from ._iterator_access import IteratorAccess


logger = logging.getLogger(__name__)


# TODO make parser abstract and merge these classes.


class IteratorAedat4(IteratorAccess):
    """Base AEDAT4 iterator backed by the local AEDAT4 reader."""

    FORMAT = "aedat4"

    def __init__(self, aedat4_file: str) -> None:
        """Initialize the iterator for an AEDAT4 file."""
        super().__init__(aedat4_file)
        self.reader = Aedat4Reader(self.file_name)
        self.count = 0

    def __iter__(self) -> Any:
        """Return the iterator after resetting to the first packet."""
        self.count = 0
        self.reader.reset()
        return self

    def close(self) -> None:
        """Close the underlying AEDAT4 reader."""
        self.reader.close()


class IteratorAedat4Event(IteratorAedat4):
    """Iterator over AEDAT4 event packets."""

    def __next__(self) -> RawEvents:
        """Return the next AEDAT4 event packet.

        Returns:
            RawEvents: Event packet with x, y, timestamp, and polarity arrays.
        """
        events = self.reader.next_events()
        self.count += len(events)
        return events


class IteratorAedat4Trigger(IteratorAedat4):
    """Iterator over AEDAT4 trigger packets."""

    def __next__(self) -> Dict[str, Any]:
        """Return the next AEDAT4 trigger row.

        Returns:
            Dict[str, Any]: Dictionary with ``trigger`` as an array of
            ``[timestamp, type]`` rows and ``num`` as the row count.
        """
        trigger = self.reader.next_trigger()
        self.count += trigger["num"]
        return trigger


class IteratorAedat4Imu(IteratorAedat4):
    """Iterator over AEDAT4 IMU packets."""

    def __next__(self) -> Dict[str, Any]:
        """Return the next AEDAT4 IMU row.

        Returns:
            Dict[str, Any]: Dictionary with ``imu`` as
            ``[timestamp, ax, ay, az, gx, gy, gz]`` and ``num`` as the row count.
        """
        imu = self.reader.next_imu()
        self.count += imu["num"]
        return imu


class IteratorAedat4Frame(IteratorAedat4):
    """Iterator over AEDAT4 frame packets."""

    def __next__(self) -> Dict[str, Any]:
        """Return the next AEDAT4 frame packet.

        Returns:
            Dict[str, Any]: Dictionary with ``frame`` as a
            ``(1, height, width, channels)`` array, ``t`` as the frame timestamp,
            and ``num`` as the frame count.
        """
        frame = self.reader.next_frame()
        self.count += frame["num"]
        return frame
