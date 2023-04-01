"""conversion functions across different formats.
"""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import h5py
import numpy as np

from ._iterator_access import IteratorAccess
from .hdf5 import hdf5append


logger = logging.getLogger(__name__)


# TODO this file should be abstracted layer of each fileformat R/W.
# Move this function to hdf5
def convert_iterator_access_to_hdf5(
    iterator_access: IteratorAccess,
    hdf5file: str,
    key_pairs: Dict[str, str],
    image_keys: List[str] = [],
) -> int:
    """Convert IteratorAccess data format into HDF5 format.

    Args:
        iterator_access (IteratorAccess): Iterator access format. Each iteration should return dictionary.
        hdf5file (str): file path for the HDF5 file.
        key_pairs (dict): {data_key: hdf5_key}
        image_keys (list of str): data_key list for specifying image data (more than 1 dimensional).
            Used to define the block size of hdf5.
    Returns:
        int ... number of data processed.
    """
    logger.info(f"Convert data {iterator_access.FORMAT} from {iterator_access.file_name}")
    logger.info(f"The key-value map (Iterator data -> Hdf5 key): {key_pairs}")
    logger.info(f"Saving events into {hdf5file}")
    # TODO add filecheck of the hdf5.
    i = 0
    with h5py.File(hdf5file, "w") as f:
        for iter_data in iterator_access:
            if i == 0:
                for (k, v) in key_pairs.items():
                    if k in image_keys:
                        image = iter_data[k]
                        maxshape = (None,) + image.shape[1:]
                        f.create_dataset(
                            v,
                            data=image,
                            maxshape=maxshape,
                            compression="gzip",
                            compression_opts=9,
                        )
                    else:
                        f.create_dataset(
                            v,
                            data=iter_data[k],
                            maxshape=(None,),
                            compression="gzip",
                            compression_opts=9,
                        )
            else:
                for (k, v) in key_pairs.items():
                    hdf5append(f[v], iter_data[k])
            i += len(iter_data)
    logger.info(f"Done. Total {i} data points are processed.")
    return i


# TODO this file should be abstracted layer of each fileformat R/W.
# Move this function to text
def write_to_text(
    event: np.ndarray,
    file_name: str
) -> None:
    np.savetxt(file_name, event[:, [2, 1, 0, 3]], fmt=['%.9f', '%d', '%d', '%d'])
