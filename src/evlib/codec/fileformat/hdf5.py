import logging
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np

H5_SET = Tuple[str, str, Any]

logger = logging.getLogger(__name__)


def load_hdf5(path: str, key_dtype_pairs: Union[List[H5_SET], H5_SET]) -> Dict[str, np.ndarray]:
    """Load all the contents of .hdf5 file at once.
    Args:
        path (str) ... Path to the .hdf5 file.
        key_dtype_pairs (list of tuple) ... A triplet, or list of triplets of
            (key of the return dictionary, key of the hdf5 file data, data type for numpy).
            For example,
            [("ts", "raw_events/ts", np.int32), ("x", "raw_events/x", np.int16), ...]
    Returns:
        dict ... {key of the return dictionary: np.ndarray}
    """
    if isinstance(key_dtype_pairs, tuple):
        key_dtype_pairs = [key_dtype_pairs]  # make it list.
    f = h5py.File(path, "r")
    data = {k: np.array(f[v], dtype=t) for (k, v, t) in key_dtype_pairs}
    f.close()
    return data


def open_hdf5(path: str) -> Any:
    """Open .hdf5 file, not to load them at once.
    Args:
        path (str) ... Path to the .hdf5 file.
    Returns:
        (Any) opened hdf5 object.
    """
    return h5py.File(path, "r")


def load_event_timestamp_hdf5(
    path: str, key_pairs: Tuple[str, str], dtype=np.int32
) -> Dict[str, np.ndarray]:
    """For utility: load only timestamps from the hdf5 data.
    Args:
        path (str) ... Path to the .hdf5 file.
        key_pairs  ... The tuple of
            (key of the return dictionary, key of the hdf5 file data)
    Returns:
    """
    data = load_hdf5(path, list(key_pairs + (dtype,)))
    if np.max(data[key_pairs[0]]) == np.iinfo(dtype):
        w = f"Please check the size of the data and data type. {dtype} may not be enough."
        logger.warning(w)
    return data


def hdf5append(data, new_arr):
    n_old = data.shape[0]
    n_new = n_old + len(new_arr)
    data.resize(n_new, axis=0)
    data[n_old:] = new_arr
