"""evk file formats, mainly for Metavision SDK software by Prophesee."""

import logging
from typing import Any

from ...types import RawEvents
from ._evt3 import Evt3RawReader
from ._iterator_access import IteratorAccess


logger = logging.getLogger(__name__)


# TODO make parser abstract and merge these classes.


class IteratorEvk3(IteratorAccess):
    """Iterator over Prophesee EVT3 RAW CD events."""

    FORMAT = "evk3"

    def __init__(self, evk3file: str) -> None:
        """Create an iterator for an EVT3 RAW file.

        Args:
            evk3file: Path to the EVT3 RAW file.
        """
        super().__init__(evk3file)
        self.reader = Evt3RawReader(self.file_name, chunk_size=16384)
        self.count = 0

    def __iter__(self) -> Any:
        """Reset and return this iterator."""
        self.count = 0
        self.reader.reset()
        return self

    def close(self) -> None:
        """Close the underlying EVT3 RAW reader."""
        self.reader.close()

    def __next__(self) -> RawEvents:
        """Return the next decoded CD event chunk."""
        chunk = self.reader.next_chunk()
        self.count += len(chunk)
        return RawEvents(x=chunk.x, y=chunk.y, timestamp=chunk.t, polarity=chunk.p)
