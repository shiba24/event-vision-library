import logging
from typing import Any
from typing import Dict

import dv
import numpy as np
from dv import AedatFile


logger = logging.getLogger(__name__)

from ._iterator_access import IteratorAccess


# TODO make parser abstract and merge these classes.


class IteratorAedat4(IteratorAccess):
    FORMAT = "aedat4"

    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file = dv.AedatFile(self.file_name)
        self.count = 0

    def __next__(self) -> Dict[str, Any]:
        raise NotImplementedError


class IteratorAedat4Event(IteratorAedat4):
    def __init__(self, aedat4file: str) -> None:
        super().__init__(aedat4file)
        self.file_iter = self.file["events"].numpy()

    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"x", "y", "t", "p": all np.ndarray (N)}
        """
        ev = self.file_iter.__next__()
        return {"x": ev["y"], "y": ev["x"], "t": ev["timestamp"], "p": ev["polarity"]}


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
        # print('-=-=-', tr, tr.type)
        # trigger_type = np.array([trig.type for trig in tr])
        # trigger_ts = np.array([trig.timestamp for trig in tr])
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
