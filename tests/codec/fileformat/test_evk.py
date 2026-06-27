"""EVK/EVT3 iterator tests."""

import os
import struct
from typing import List
from typing import Optional

import numpy as np
import pytest

from evlib.codec import fileformat
from evlib.codec.fileformat._evt3 import Evt3RawError
from evlib.codec.fileformat._evt3 import Evt3RawReader


def _write_evt3_raw(path: str, words: List[int], header: Optional[bytes] = None) -> None:
    if header is None:
        header = b"% evt 3.0\n% format EVT3;height=10;width=32\n% end\n"
    with open(path, "wb") as file_handle:
        file_handle.write(header)
        for word in words:
            file_handle.write(struct.pack("<H", word))


def test_iterator_evk3_decodes_evt3_events(tmp_path):  # type: ignore
    """Decode EVT3 single and vector CD events into RawEvents."""
    path = os.path.join(tmp_path, "events.raw")
    words = [
        0x8001,  # time high = 1
        0x6002,  # time low = 2
        0x0003,  # y = 3
        0x2804,  # x = 4, p = 1
        0x300A,  # vector base x = 10, p = 0
        0x4005,  # x = 10 and 12
        0x5003,  # x = 22 and 23
    ]
    _write_evt3_raw(path, words)

    events = next(fileformat.IteratorEvk3(path))

    assert events.x.dtype.type is np.int32
    assert events.y.dtype.type is np.int32
    assert events.t.dtype.type is np.float64
    assert events.p.dtype.type is np.bool_
    assert events.x.tolist() == [4, 10, 12, 22, 23]
    assert events.y.tolist() == [3, 3, 3, 3, 3]
    assert events.t.tolist() == [4098.0, 4098.0, 4098.0, 4098.0, 4098.0]
    assert events.p.tolist() == [True, False, False, False, False]


def test_evt3_reader_preserves_complete_vector_when_chunk_limit_is_reached(tmp_path):  # type: ignore
    """Keep vector events together when they cross the target chunk size."""
    path = os.path.join(tmp_path, "events.raw")
    words = [
        0x8000,
        0x6000,
        0x0001,
        0x2001,
        0x300A,
        0x400F,
        0x201E,
    ]
    _write_evt3_raw(path, words)

    reader = Evt3RawReader(path, chunk_size=2)
    first_chunk = reader.next_chunk()
    second_chunk = reader.next_chunk()

    assert first_chunk.x.tolist() == [1, 10, 11, 12, 13]
    assert second_chunk.x.tolist() == [30]


def test_evt3_reader_tracks_time_low_and_high_wraps(tmp_path):  # type: ignore
    """Reconstruct timestamps across EVT3 low and high timestamp wraps."""
    path = os.path.join(tmp_path, "events.raw")
    words = [
        0x8FFF,
        0x6FFF,
        0x0001,
        0x2001,
        0x8000,
        0x6000,
        0x0001,
        0x2002,
    ]
    _write_evt3_raw(path, words)

    events = next(fileformat.IteratorEvk3(path))

    assert events.t.tolist() == [16777215.0, 16781312.0]


def test_evt3_reader_accepts_raw_without_header(tmp_path):  # type: ignore
    """Decode headerless EVT3 words by treating the file as raw data."""
    path = os.path.join(tmp_path, "events.raw")
    _write_evt3_raw(path, [0x8000, 0x6001, 0x0002, 0x2803], header=b"")

    events = next(fileformat.IteratorEvk3(path))

    assert events.x.tolist() == [3]
    assert events.y.tolist() == [2]
    assert events.t.tolist() == [1.0]
    assert events.p.tolist() == [True]


def test_evt3_reader_ignores_odd_trailing_byte_at_eof(tmp_path):  # type: ignore
    """Ignore a final byte that cannot form a complete EVT3 word."""
    path = os.path.join(tmp_path, "events.raw")
    _write_evt3_raw(path, [0x8000, 0x6001, 0x0002, 0x2803])
    with open(path, "ab") as file_handle:
        file_handle.write(b"\n")

    events = next(fileformat.IteratorEvk3(path))

    assert events.x.tolist() == [3]
    assert events.y.tolist() == [2]
    assert events.t.tolist() == [1.0]
    assert events.p.tolist() == [True]


def test_evt3_reader_rejects_non_evt3_raw_header(tmp_path):  # type: ignore
    """Reject RAW files whose header declares a different event encoding."""
    path = os.path.join(tmp_path, "events.raw")
    _write_evt3_raw(path, [0x0000], header=b"% evt 2.0\n% format EVT2;height=10;width=32\n% end\n")

    with pytest.raises(Evt3RawError, match="Only Prophesee EVT3 RAW"):
        fileformat.IteratorEvk3(path)
