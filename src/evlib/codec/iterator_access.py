"""To be used: define dataset.
"""

from typing import Any


class IteratorAccessDataset:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def setup(*args: Any, **kwargs: Any) -> IteratorAccessDataset:
    return IteratorAccessDataset(args, kwargs)
