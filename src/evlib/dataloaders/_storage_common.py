"""Common storage helpers shared by dataloader implementations."""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal
from typing import Optional
from typing import Union

import h5py
import numpy as np
import numpy.typing as npt

from evlib.codec.fileformat.hdf5 import open_hdf5


LoadMode = Union[bool, Literal["lazy", "cached"], "LoadingType"]
ResidentLoadMode = Union[Literal["lazy", "cached"], "LoadingType"]


class LoadingType(str, Enum):
    """internal loading state for dataset modalities."""

    OFF = "off"
    LAZY = "lazy"
    CACHED = "cached"

    @classmethod
    def from_value(
        cls,
        value: LoadMode,
        *,
        name: str = "loading type",
    ) -> "LoadingType":
        if isinstance(value, cls):
            return value

        if value is False:
            return cls.OFF

        if value is True or value == "lazy":
            return cls.LAZY

        if value == "cached":
            return cls.CACHED

        raise ValueError(f"Invalid {name}: {value!r}. Expected False, True, 'lazy', or 'cached'.")

    @classmethod
    def from_resident_value(
        cls,
        value: ResidentLoadMode,
        *,
        name: str,
    ) -> "LoadingType":
        if isinstance(value, cls):
            if value is cls.OFF:
                raise ValueError(f"{name} must be 'cached' or 'lazy', got {value!r}")
            return value

        if value == "lazy":
            return cls.LAZY

        if value == "cached":
            return cls.CACHED

        raise ValueError(f"{name} must be 'cached' or 'lazy', got {value!r}")

    @property
    def should_load(self) -> bool:
        return self is not self.OFF

    @property
    def should_cache(self) -> bool:
        return self is self.CACHED


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
