"""To be used: define dataset.
"""

from typing import Any


class BlockAccessDataset:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def setup(*args: Any, **kwargs: Any) -> BlockAccessDataset:
    return BlockAccessDataset(args, kwargs)
