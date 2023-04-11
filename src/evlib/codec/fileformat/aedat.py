"""aedat file formats, mainly for DV software by iniVation.
"""
import logging
from typing import Any
from typing import Dict

import dv
import numpy as np
from dv import AedatFile


logger = logging.getLogger(__name__)

from ...types import RawEvents
from ._iterator_access import IteratorAccess


# TODO make parser abstract and merge these classes.


class IteratorAedat4(IteratorAccess):
    FORMAT = "aedat4"

    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file = dv.AedatFile(self.file_name)
        self.count = 0


class IteratorAedat4Event(IteratorAedat4):
    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file_iter = self.file["events"].numpy()

    def __next__(self) -> RawEvents:
        """
        Returns:
            RawEvents:
        """
        ev = self.file_iter.__next__()
        return RawEvents(x=ev["x"], y=ev["y"], timestamp=ev["timestamp"], polarity=ev["polarity"])


class IteratorAedat4Trigger(IteratorAedat4):
    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file_iter = self.file["triggers"]

    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"trigger": (1, 2)}
        """
        tr = self.file_iter.__next__()
        trigger = np.array([[tr.timestamp, tr.type]], dtype=np.float64)
        return {"trigger": trigger, "num": len(trigger)}


class IteratorAedat4Imu(IteratorAedat4):
    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file_iter = self.file["imu"]

    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"imu": np.ndarray (1, N, 7)}
            7 is t, ax, ay, az, gx, gy, gz
        """
        imu = self.file["imu"].__next__()
        return {
            "imu": np.array([imu.timestamp, *imu.accelerometer, *imu.gyroscope]),
            "num": 1,
        }


class IteratorAedat4Frame(IteratorAedat4):
    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file_iter = self.file["frames"]

    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"t" (1, ), "image": (1, H, W, 3)}
        """
        fr = self.file_iter.__next__()
        ts = np.array([fr.timestamp])
        image = np.array(fr.image, dtype=np.uint8)[None]  # H, W, 1
        return {"frame": image, "t": ts}
