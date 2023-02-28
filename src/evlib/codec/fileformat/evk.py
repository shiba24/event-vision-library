import logging

import numpy as np
from expelliarmus import Wizard

logger = logging.getLogger(__name__)

from ._iterator_access import IteratorAccess

# TODO make parser abstract and merge these classes.


class IteratorEvk3(IteratorAccess):
    FORMAT = "evk3"

    def __init__(self, evk3file: str) -> None:
        super().__init__(evk3file)
        self.wizard = Wizard(encoding="evt3", fpath=self.file_name, chunk_size=16384)
        self.count = 0

    def __iter__(self):
        return self.wizard.read_chunk()
