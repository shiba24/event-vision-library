import logging
import os
from typing import Any
from typing import Dict
from typing import List

import cv2
import numpy as np


logger = logging.getLogger(__name__)

from ._iterator_access import IteratorAccess


# TODO make parser abstract and merge these classes.


class IteratorText(IteratorAccess):
    FORMAT = "text"

    def __init__(self, textfile: str) -> None:
        # TODO add parse format option
        super().__init__(textfile)
        self.file = open(self.file_name)

    def __iter__(self) -> Any:
        self.count = 0
        return self

    def read_next_lines(self) -> List[str]:
        lines = self.file.readlines()
        if not lines:
            raise StopIteration
        return lines

    def __next__(self) -> Dict[str, Any]:
        raise NotImplementedError


class IteratorTextEvent(IteratorText):
    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"x", "y", "t", "p": all np.ndarray (N)}
        """
        lines = self.read_next_lines()
        _l = len(lines)
        x = np.zeros((_l,), dtype=np.int32)
        y = np.zeros((_l,), dtype=np.int32)
        t = np.zeros((_l,), dtype=np.float64)
        p = np.zeros((_l,), dtype=bool)
        for _i, line in enumerate(lines):
            val = line.split()
            t[_i] = np.float64(val[0])
            x[_i] = int(val[2])
            y[_i] = int(val[1])
            p[_i] = int(val[3])
        self.count += _l
        return {"x": x, "y": y, "t": t, "p": p, "num": _l}


class IteratorTextPose(IteratorText):
    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"pose": np.ndarray (N, 7)}
        """
        lines = self.read_next_lines()
        _l = len(lines)
        pose_list = []
        for _i, line in enumerate(lines):
            val = line.split()
            pose_list.append(np.array(val, dtype=np.float64))
        self.count += _l
        return {"pose": np.stack(pose_list, axis=0), "num": _l}


class IteratorTextImu(IteratorText):
    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"imu": np.ndarray (N, 7)}
        """
        lines = self.read_next_lines()
        _l = len(lines)
        imu_list = []
        for _i, line in enumerate(lines):
            val = line.split()
            imu_list.append(np.array(val, dtype=np.float64))
        self.count += _l
        return {"imu": np.stack(imu_list, axis=0), "num": _l}


class IteratorTextFrame(IteratorText):
    def __next__(self) -> Dict[str, Any]:
        """
        Returns:
            dict: {"frame": np.ndarray (N, H, W), "t": np.ndarray (N)}
        """
        # TODO assumption is that always images.txt and the image root dir is the same.
        image_dir = os.path.dirname(self.file_name)
        lines = self.read_next_lines()
        _l = len(lines)
        image_list = []
        image_timestamp = []
        for line in lines:
            val = line.split()
            image_timestamp.append(np.float64(val[0]))
            image_file = os.path.join(image_dir, val[1])
            image_list.append(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
        self.count += _l
        image_timestamps = np.array(image_timestamp)
        images = np.array(image_list)
        return {"frame": images, "t": image_timestamps, "num": _l}
