from typing import Any
from typing import Dict


class IteratorAccess:
    FORMAT = "base"

    def __init__(self, file_name: str, **kwargs: Any) -> None:
        self.file_name = file_name

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        raise NotImplementedError
