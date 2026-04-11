"""Common storage helpers shared by dataloader implementations."""

import os
from typing import Literal
from typing import Optional
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from evlib.codec.fileformat.hdf5 import open_hdf5


ResidentLoadMode = Literal["cached", "lazy"]


def normalize_resident_load_mode(name: str, value: str) -> ResidentLoadMode:
    """Validate and normalize a resident loading mode string."""
    if value not in ("cached", "lazy"):
        raise ValueError(f"{name} must be 'cached' or 'lazy', got {value!r}")

    mode = cast(ResidentLoadMode, value)
    return mode


class _LazyH5Dataset:
    """Process local lazy reader for a single HDF5 dataset.

    Reopens the file handle after fork() or unpickle since HDF5
    handles are not safe across process boundaries.
    """

    def __init__(self, file_path: str, dataset_key: str, dtype: type) -> None:
        self._file_path = file_path
        self._dataset_key = dataset_key
        self._dtype = dtype
        self._file: Optional[h5py.File] = None
        self._dataset: Optional[h5py.Dataset] = None
        self._pid: Optional[int] = None
        self._closed = False

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        state["_pid"] = None
        return state

    @property
    def has_open_handle(self) -> bool:
        return self._file is not None

    def _ensure_open(self) -> Optional[h5py.Dataset]:
        if self._closed:
            return None

        pid = os.getpid()
        same_process = self._pid == pid
        has_dataset = self._dataset is not None
        if same_process and has_dataset:
            return self._dataset

        self._drop_handles()

        file_handle = open_hdf5(self._file_path)
        dataset = file_handle[self._dataset_key]

        self._file = file_handle
        self._dataset = dataset
        self._pid = pid
        return dataset

    def _drop_handles(self) -> None:
        file_handle = self._file
        if file_handle is not None:
            file_handle.close()

        self._file = None
        self._dataset = None
        self._pid = None

    def read(self, index: int) -> "Optional[npt.NDArray[np.generic]]":
        dataset = self._ensure_open()
        if dataset is None:
            return None

        array: npt.NDArray[np.generic] = np.asarray(dataset[index], dtype=self._dtype)
        array.flags.writeable = False
        return array

    def close(self) -> None:
        self._drop_handles()
        self._closed = True
