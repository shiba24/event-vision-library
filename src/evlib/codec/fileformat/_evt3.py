from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from typing import Iterator

import numpy as np
import numpy.typing as npt


_HEADER_PREFIX = b"% "
# 64kb blocks seem to bench fastest for the NumPy decoder
_DEFAULT_READ_SIZE = 64 << 10
_UINT16_SIZE = 2

# Each EVT3 word is a 16bit little endian value:
#   top nibble - the word type
#   low 12 bits - payload
# These are the wordtype codes
_EVT_ADDR_Y = 0x0
_EVT_ADDR_X = 0x2
_VECT_BASE_X = 0x3
_VECT_12 = 0x4
_VECT_8 = 0x5
_EVT_TIME_LOW = 0x6
_EVT_TIME_HIGH = 0x8

_PAYLOAD_MASK = 0x0FFF  # low 12 bits
_ADDRESS_MASK = 0x07FF  # low 11 bits: an X or Y coordinate
_POLARITY_BIT = 0x0800  # bit 11 of an address word
_VECTOR_8_MASK = 0x00FF  # VECT_8 stores its mask in the low 8 bits
_TIME_LOW_RANGE = 1 << 12  # time-low wraps every 4096 ticks
_TIME_HIGH_RANGE_US = 1 << 24  # time-high wraps every 2**24 us


def _build_vector_tables(
    vector_size: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
    # For every possible vector bitmask
    # precompute how many bits are set (counts) and which bit positions are set (offsets)
    # The decoder then expands a vector word with a table lookup instead of per bit work
    counts = np.empty((1 << vector_size,), dtype=np.uint8)
    offsets = np.zeros((1 << vector_size, vector_size), dtype=np.int32)
    for mask in range(1 << vector_size):
        set_bit_offsets = []
        for bit_index in range(vector_size):
            if mask & (1 << bit_index):
                set_bit_offsets.append(bit_index)

        offset_count = len(set_bit_offsets)
        counts[mask] = offset_count
        if offset_count:
            offsets[mask, :offset_count] = set_bit_offsets
    return counts, offsets


_VECTOR_8_COUNTS, _VECTOR_8_OFFSETS = _build_vector_tables(8)
_VECTOR_12_COUNTS, _VECTOR_12_OFFSETS = _build_vector_tables(12)
_VECTOR_8_INDEXES = np.arange(8, dtype=np.int64)
_VECTOR_12_INDEXES = np.arange(12, dtype=np.int64)


class Evt3RawError(RuntimeError):
    """Raised when an EVT3 RAW file cannot be decoded."""


@dataclass(frozen=True)
class Evt3EventChunk:
    """Decoded EVT3 CD event chunk."""

    t: npt.NDArray[np.float64]
    x: npt.NDArray[np.int32]
    y: npt.NDArray[np.int32]
    p: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        """Return the number of decoded CD events."""
        return len(self.t)


@dataclass(frozen=True)
class _DecodedEventBlock:
    t: npt.NDArray[np.float64]
    x: npt.NDArray[np.int32]
    y: npt.NDArray[np.int32]
    p: npt.NDArray[np.bool_]
    ends_word: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.t)

    def slice(self, start: int, stop: int) -> Evt3EventChunk:
        return Evt3EventChunk(
            t=self.t[start:stop],
            x=self.x[start:stop],
            y=self.y[start:stop],
            p=self.p[start:stop],
        )


@dataclass
class _Evt3State:
    y: int = 0
    base_x: int = 0
    vector_polarity: bool = False
    time_high: int = 0
    time_low: int = 0
    time_high_overflow: int = 0
    time_low_overflow: int = 0

    @property
    def timestamp(self) -> int:
        return (
            self.time_high_overflow
            + self.time_low_overflow
            + (self.time_high << 12)
            + self.time_low
        )


class Evt3RawReader:
    """Streaming decoder for the Prophesee EVT3 RAW subset.

    The reader decodes CD events from EVT3 event stream RAW files. It handles
    RAW ASCII headers and the EVT3 words that produce CD events:
    Y addresses, single X events, vector bases, vector events, and timestamp
    high/low updates.
    """

    def __init__(
        self,
        file_name: str | Path,
        *,
        chunk_size: int = 16384,
        read_size: int = _DEFAULT_READ_SIZE,
    ) -> None:
        """Open an EVT3 RAW file.

        Args:
            file_name: Path to a Prophesee EVT3 ``.raw`` file.
            chunk_size: Target number of CD events per decoded chunk.
            read_size: Number of encoded bytes to read per internal block.

        Raises:
            ValueError: If chunk or read sizes are invalid.
        """
        if chunk_size <= 0:
            raise ValueError("EVT3 chunk_size must be positive")
        if read_size <= 0:
            raise ValueError("EVT3 read_size must be positive")

        self.file_name = str(file_name)
        self.chunk_size = chunk_size
        self.read_size = read_size
        self.file: BinaryIO = open(self.file_name, "rb")
        self.header = self._read_raw_header()
        self.data_offset = self.file.tell()
        self.state = _Evt3State()
        self._decode_initial_state = _Evt3State()
        self._event_block = self._empty_event_block()
        self._event_index = 0
        self._pending_byte = b""
        self._finished = False

    def close(self) -> None:
        """Close the underlying file handle."""
        self.file.close()

    def reset(self) -> None:
        """Reset decoding to the first encoded word."""
        self.file.seek(self.data_offset)
        self.state = _Evt3State()
        self._decode_initial_state = _Evt3State()
        self._event_block = self._empty_event_block()
        self._event_index = 0
        self._pending_byte = b""
        self._finished = False

    def read_chunks(self) -> Iterator[Evt3EventChunk]:
        """Yield decoded CD events in chunks."""
        while True:
            try:
                yield self.next_chunk()
            except StopIteration:
                return

    def next_chunk(self) -> Evt3EventChunk:
        """Return the next decoded CD event chunk.

        Returns:
            The next decoded CD event chunk.

        Raises:
            StopIteration: When the file has no more CD events.
        """
        slices: list[Evt3EventChunk] = []
        event_count = 0

        while event_count < self.chunk_size:
            if self._event_index >= len(self._event_block):
                if not self._load_decoded_event_block():
                    if event_count == 0:
                        raise StopIteration
                    break

            block = self._event_block
            start = self._event_index
            remaining = self.chunk_size - event_count
            stop = min(start + remaining, len(block))

            # Never split a vector's events across chunks,
            # extend to the next word boundary even if it slightly overshoots chunk_size
            if stop < len(block) and stop > start:
                while stop < len(block) and not block.ends_word[stop - 1]:
                    stop += 1

            slices.append(block.slice(start, stop))
            self._event_index = stop
            event_count += stop - start

        if len(slices) == 1:
            return slices[0]

        return Evt3EventChunk(
            t=np.concatenate([chunk.t for chunk in slices]),
            x=np.concatenate([chunk.x for chunk in slices]),
            y=np.concatenate([chunk.y for chunk in slices]),
            p=np.concatenate([chunk.p for chunk in slices]),
        )

    def _read_raw_header(self) -> dict[str, str]:
        header: dict[str, str] = {}
        while True:
            line_start = self.file.tell()
            line = self.file.readline()
            if not line:
                self._validate_header(header)
                return header
            if not line.startswith(_HEADER_PREFIX):
                self.file.seek(line_start)
                self._validate_header(header)
                return header

            decoded_line = line[2:].decode("ascii", errors="replace").strip()
            if decoded_line == "end":
                self._validate_header(header)
                return header
            key, _, value = decoded_line.partition(" ")
            header[key] = value

    @staticmethod
    def _validate_header(header: dict[str, str]) -> None:
        if not header:
            return

        format_value = header.get("format", "")
        evt_value = header.get("evt", "")
        if "EVT3" in format_value.upper():
            return
        if evt_value.strip() in {"3", "3.0"}:
            return
        if format_value or evt_value:
            raise Evt3RawError(
                "Only Prophesee EVT3 RAW event streams are supported by IteratorEvk3"
            )

    @staticmethod
    def _empty_event_block() -> _DecodedEventBlock:
        return _DecodedEventBlock(
            t=np.empty((0,), dtype=np.float64),
            x=np.empty((0,), dtype=np.int32),
            y=np.empty((0,), dtype=np.int32),
            p=np.empty((0,), dtype=np.bool_),
            ends_word=np.empty((0,), dtype=np.bool_),
        )

    def _load_decoded_event_block(self) -> bool:
        if self._finished:
            return False

        while True:
            words = self._read_word_block()
            if words is None:
                return False

            block = self._decode_word_block(words)
            if len(block) > 0:
                self._event_block = block
                self._event_index = 0
                return True

    def _read_word_block(self) -> npt.NDArray[np.uint16] | None:
        data = self.file.read(self.read_size)
        if not data:
            # At EOF any held byte cannot complete a word, so drop it
            # Prophesee files append a trailing newline that lands here
            self._finished = True
            self._pending_byte = b""
            return None

        if self._pending_byte:
            data = self._pending_byte + data
            self._pending_byte = b""

        # A read can end mid word, hold the spare byte for the next read
        usable_size = len(data) - (len(data) % _UINT16_SIZE)
        if usable_size != len(data):
            self._pending_byte = data[-1:]
            data = data[:usable_size]

        if not data:
            return self._read_word_block()
        return np.frombuffer(data, dtype="<u2")

    def _decode_word_block(self, words: npt.NDArray[np.uint16]) -> _DecodedEventBlock:
        # Classify every word by type, then collect the positions of each kind
        word_types = (words >> 12).astype(np.uint16, copy=False)
        payloads = (words & _PAYLOAD_MASK).astype(np.uint16, copy=False)
        y_positions = np.flatnonzero(word_types == _EVT_ADDR_Y)
        single_x_positions = np.flatnonzero(word_types == _EVT_ADDR_X)
        base_positions = np.flatnonzero(word_types == _VECT_BASE_X)
        vector_12_positions = np.flatnonzero(word_types == _VECT_12)
        vector_8_positions = np.flatnonzero(word_types == _VECT_8)
        low_positions = np.flatnonzero(word_types == _EVT_TIME_LOW)
        high_positions = np.flatnonzero(word_types == _EVT_TIME_HIGH)

        # Width each vector word spans in X (0 for nonvector words)
        # The running cumulative sum tells each vector how far base-X has advanced
        vector_sizes = np.zeros(len(words), dtype=np.int32)
        vector_sizes[vector_12_positions] = 12
        vector_sizes[vector_8_positions] = 8
        cumulative_vector_sizes = np.empty(len(words) + 1, dtype=np.int64)
        cumulative_vector_sizes[0] = 0
        np.cumsum(vector_sizes, out=cumulative_vector_sizes[1:])

        # Snapshot state at the block start, then advance it past this block so
        # the next block decodes from the correct carried-over state.
        self._update_state(
            words,
            payloads,
            vector_sizes,
            y_positions,
            low_positions,
            high_positions,
            base_positions,
        )

        # How many CD events each word emits:
        #    one per single-X word
        #    one per set bit in a vector word
        event_counts = np.zeros(len(words), dtype=np.uint8)
        event_counts[single_x_positions] = 1
        event_counts[vector_12_positions] = _VECTOR_12_COUNTS[payloads[vector_12_positions]]
        vector_8_masks = words[vector_8_positions] & _VECTOR_8_MASK
        event_counts[vector_8_positions] = _VECTOR_8_COUNTS[vector_8_masks]

        event_count = int(event_counts.sum())
        if event_count == 0:
            return self._empty_event_block()

        # Output slot where each word's events start
        starts = np.empty(len(words), dtype=np.int64)
        np.cumsum(event_counts, out=starts)
        starts -= event_counts

        # Allocate output columns,
        # then compute the decoder state that applies to each EVT_ADDR_X or vector word that produces events
        timestamps = np.empty(event_count, dtype=np.int64)
        xs = np.empty(event_count, dtype=np.int32)
        ys = np.empty(event_count, dtype=np.int32)
        polarities = np.empty(event_count, dtype=np.bool_)
        ends_word = np.empty(event_count, dtype=np.bool_)
        event_word_positions = np.flatnonzero(event_counts > 0)
        event_word_timestamps = self._timestamps_at_positions(
            payloads,
            event_word_positions,
            low_positions,
            high_positions,
        )
        y_values = (words[y_positions] & _ADDRESS_MASK).astype(np.int64, copy=False)
        event_word_ys = self._int_state_from_updates(
            y_positions,
            y_values,
            event_word_positions,
            self._decode_initial_state.y,
        )
        event_word_types = word_types[event_word_positions]
        is_single_x_event_word = event_word_types == _EVT_ADDR_X
        is_vector_12_event_word = event_word_types == _VECT_12
        is_vector_8_event_word = event_word_types == _VECT_8
        is_vector_event_word = is_vector_12_event_word | is_vector_8_event_word
        single_x_event_indexes = np.flatnonzero(is_single_x_event_word)
        vector_12_event_indexes = np.flatnonzero(is_vector_12_event_word)
        vector_8_event_indexes = np.flatnonzero(is_vector_8_event_word)
        vector_event_indexes = np.flatnonzero(is_vector_event_word)
        vector_event_positions = event_word_positions[vector_event_indexes]
        vector_12_indexes_by_word = np.searchsorted(vector_event_indexes, vector_12_event_indexes)
        vector_8_indexes_by_word = np.searchsorted(vector_event_indexes, vector_8_event_indexes)
        base_polarities = (words[base_positions] & _POLARITY_BIT) != 0
        vector_polarities = self._bool_state_at_positions(
            base_positions,
            base_polarities,
            vector_event_positions,
            self._decode_initial_state.vector_polarity,
        )
        vector_base_xs = self._vector_base_x_at_positions(
            words,
            base_positions,
            vector_event_positions,
            cumulative_vector_sizes,
        )

        self._fill_single_x_events(
            words,
            event_word_positions[single_x_event_indexes],
            starts,
            single_x_event_indexes,
            event_word_timestamps,
            event_word_ys,
            timestamps,
            xs,
            ys,
            polarities,
            ends_word,
        )
        self._fill_vector_events(
            words,
            payloads,
            event_word_positions[vector_12_event_indexes],
            starts,
            vector_12_event_indexes,
            event_word_timestamps,
            event_word_ys,
            vector_12_indexes_by_word,
            vector_polarities,
            vector_base_xs,
            timestamps,
            xs,
            ys,
            polarities,
            ends_word,
            vector_size=12,
        )
        self._fill_vector_events(
            words,
            payloads,
            event_word_positions[vector_8_event_indexes],
            starts,
            vector_8_event_indexes,
            event_word_timestamps,
            event_word_ys,
            vector_8_indexes_by_word,
            vector_polarities,
            vector_base_xs,
            timestamps,
            xs,
            ys,
            polarities,
            ends_word,
            vector_size=8,
        )
        return _DecodedEventBlock(
            t=timestamps.astype(np.float64, copy=False),
            x=xs,
            y=ys,
            p=polarities,
            ends_word=ends_word,
        )

    def _fill_single_x_events(
        self,
        words: npt.NDArray[np.uint16],
        event_positions: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        event_indexes: npt.NDArray[np.int64],
        event_word_timestamps: npt.NDArray[np.int64],
        event_word_ys: npt.NDArray[np.int64],
        timestamps: npt.NDArray[np.int64],
        xs: npt.NDArray[np.int32],
        ys: npt.NDArray[np.int32],
        polarities: npt.NDArray[np.bool_],
        ends_word: npt.NDArray[np.bool_],
    ) -> None:
        if len(event_positions) == 0:
            return

        target_indices = starts[event_positions]
        timestamps[target_indices] = event_word_timestamps[event_indexes]
        event_words = words[event_positions]
        xs[target_indices] = event_words & _ADDRESS_MASK
        ys[target_indices] = event_word_ys[event_indexes]
        polarities[target_indices] = (event_words & _POLARITY_BIT) != 0
        ends_word[target_indices] = True

    def _fill_vector_events(
        self,
        words: npt.NDArray[np.uint16],
        payloads: npt.NDArray[np.uint16],
        vector_positions: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        event_indexes: npt.NDArray[np.int64],
        event_word_timestamps: npt.NDArray[np.int64],
        event_word_ys: npt.NDArray[np.int64],
        vector_indexes_by_word: npt.NDArray[np.int64],
        vector_polarities: npt.NDArray[np.bool_],
        vector_base_xs: npt.NDArray[np.int32],
        timestamps: npt.NDArray[np.int64],
        xs: npt.NDArray[np.int32],
        ys: npt.NDArray[np.int32],
        polarities: npt.NDArray[np.bool_],
        ends_word: npt.NDArray[np.bool_],
        *,
        vector_size: int,
    ) -> None:
        if len(vector_positions) == 0:
            return

        if vector_size == 12:
            masks = payloads[vector_positions]
            counts = _VECTOR_12_COUNTS[masks].astype(np.int64, copy=False)
            offsets = _VECTOR_12_OFFSETS[masks]
            vector_indexes = _VECTOR_12_INDEXES
        else:
            masks = (words[vector_positions] & _VECTOR_8_MASK).astype(np.uint16, copy=False)
            counts = _VECTOR_8_COUNTS[masks].astype(np.int64, copy=False)
            offsets = _VECTOR_8_OFFSETS[masks]
            vector_indexes = _VECTOR_8_INDEXES

        nonempty = counts > 0
        if not np.any(nonempty):
            return

        vector_positions = vector_positions[nonempty]
        counts = counts[nonempty]
        offsets = offsets[nonempty]
        event_indexes = event_indexes[nonempty]
        vector_indexes_by_word = vector_indexes_by_word[nonempty]

        has_event_at_offset = vector_indexes < counts[:, None]
        vector_start_indices = starts[vector_positions, None]
        target_indices = (vector_start_indices + vector_indexes)[has_event_at_offset]

        timestamps[target_indices] = np.repeat(event_word_timestamps[event_indexes], counts)
        ys[target_indices] = np.repeat(event_word_ys[event_indexes], counts)
        polarities[target_indices] = np.repeat(
            vector_polarities[vector_indexes_by_word],
            counts,
        )
        base_xs = vector_base_xs[vector_indexes_by_word]
        expanded_xs = base_xs[:, None] + offsets
        xs[target_indices] = expanded_xs[has_event_at_offset]
        ends_word[target_indices] = False
        last_event_indices = starts[vector_positions] + counts - 1
        ends_word[last_event_indices] = True

    def _timestamps_at_positions(
        self,
        payloads: npt.NDArray[np.uint16],
        event_positions: npt.NDArray[np.int64],
        low_positions: npt.NDArray[np.int64],
        high_positions: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        low_values, _ = self._unwrapped_values(
            payloads[low_positions],
            initial_value=self._decode_initial_state.time_low,
            initial_overflow=self._decode_initial_state.time_low_overflow,
            wrap_increment=_TIME_LOW_RANGE,
            shift=0,
        )
        high_values, _ = self._unwrapped_values(
            payloads[high_positions],
            initial_value=self._decode_initial_state.time_high,
            initial_overflow=self._decode_initial_state.time_high_overflow,
            wrap_increment=_TIME_HIGH_RANGE_US,
            shift=12,
        )
        initial_low_timestamp = (
            self._decode_initial_state.time_low_overflow + self._decode_initial_state.time_low
        )
        initial_high_timestamp = self._decode_initial_state.time_high_overflow + (
            self._decode_initial_state.time_high << 12
        )
        low_at_events = self._int_state_from_updates(
            low_positions,
            low_values,
            event_positions,
            initial_low_timestamp,
        )
        high_at_events = self._int_state_from_updates(
            high_positions,
            high_values,
            event_positions,
            initial_high_timestamp,
        )
        np.add(low_at_events, high_at_events, out=low_at_events)
        return low_at_events

    @staticmethod
    def _int_state_from_updates(
        update_positions: npt.NDArray[np.int64],
        update_values: npt.NDArray[np.int64],
        event_positions: npt.NDArray[np.int64],
        initial_value: int,
    ) -> npt.NDArray[np.int64]:
        if len(update_positions) == 0:
            return np.full(len(event_positions), initial_value, dtype=np.int64)

        # For each event, take the value from the most recent update at or before it;
        # events before any update keep the carried in initial value
        update_indexes = np.searchsorted(update_positions, event_positions, side="right") - 1
        values = np.empty(len(event_positions), dtype=np.int64)
        has_update = update_indexes >= 0
        values[~has_update] = initial_value
        values[has_update] = update_values[update_indexes[has_update]]
        return values

    @staticmethod
    def _bool_state_at_positions(
        update_positions: npt.NDArray[np.int64],
        update_values: npt.NDArray[np.bool_],
        event_positions: npt.NDArray[np.int64],
        initial_value: bool,
    ) -> npt.NDArray[np.bool_]:
        if len(update_positions) == 0:
            return np.full(len(event_positions), initial_value, dtype=np.bool_)

        update_indexes = np.searchsorted(update_positions, event_positions, side="right") - 1
        values = np.empty(len(event_positions), dtype=np.bool_)
        has_update = update_indexes >= 0
        values[~has_update] = initial_value
        values[has_update] = update_values[update_indexes[has_update]]
        return values

    @staticmethod
    def _unwrapped_values(
        values: npt.NDArray[np.uint16],
        *,
        initial_value: int,
        initial_overflow: int,
        wrap_increment: int,
        shift: int,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        int_values = values.astype(np.int64, copy=False)
        if len(int_values) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

        # The timer counts up then wraps to 0, so a value smaller than predecessor marks a wrap
        # Accumulating wraps recovers the high bits
        wraps = np.empty(len(int_values), dtype=np.bool_)
        wraps[0] = int_values[0] < initial_value
        current_values = int_values[1:]
        previous_values = int_values[:-1]
        wraps[1:] = current_values < previous_values
        overflows = np.cumsum(wraps, dtype=np.int64)
        overflows *= wrap_increment
        overflows += initial_overflow
        unwrapped = int_values << shift
        unwrapped += overflows
        return unwrapped, overflows

    @staticmethod
    def _last_overflow(
        values: npt.NDArray[np.uint16],
        *,
        initial_value: int,
        initial_overflow: int,
        wrap_increment: int,
    ) -> int:
        int_values = values.astype(np.int64, copy=False)
        wrap_count = int(int_values[0] < initial_value)
        if len(int_values) > 1:
            current_values = int_values[1:]
            previous_values = int_values[:-1]
            wrap_count += int(np.count_nonzero(current_values < previous_values))
        return initial_overflow + (wrap_count * wrap_increment)

    def _vector_base_x_at_positions(
        self,
        words: npt.NDArray[np.uint16],
        base_positions: npt.NDArray[np.int64],
        vector_positions: npt.NDArray[np.int64],
        cumulative_sizes: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        if len(base_positions) == 0:
            carried_base_xs = self._decode_initial_state.base_x + cumulative_sizes[vector_positions]
            return carried_base_xs.astype(np.int32, copy=False)

        # base-X advances by the width of every vector word emitted since its base word,
        # the gap between cumulative sizes gives the running base
        base_indexes = np.searchsorted(base_positions, vector_positions, side="right") - 1
        base_values = (words[base_positions] & _ADDRESS_MASK).astype(np.int64, copy=False)
        base_xs = np.empty(len(vector_positions), dtype=np.int64)
        has_base = base_indexes >= 0

        positions_without_base = vector_positions[~has_base]
        base_xs[~has_base] = (
            self._decode_initial_state.base_x + cumulative_sizes[positions_without_base]
        )

        positions_with_base = vector_positions[has_base]
        matching_base_indexes = base_indexes[has_base]
        matching_base_positions = base_positions[matching_base_indexes]
        vector_distance_from_base = (
            cumulative_sizes[positions_with_base] - cumulative_sizes[matching_base_positions]
        )
        base_xs[has_base] = base_values[matching_base_indexes] + vector_distance_from_base
        return base_xs.astype(np.int32, copy=False)

    def _update_state(
        self,
        words: npt.NDArray[np.uint16],
        payloads: npt.NDArray[np.uint16],
        vector_sizes: npt.NDArray[np.int32],
        y_positions: npt.NDArray[np.int64],
        low_positions: npt.NDArray[np.int64],
        high_positions: npt.NDArray[np.int64],
        base_positions: npt.NDArray[np.int64],
    ) -> None:
        self._decode_initial_state = _Evt3State(
            y=self.state.y,
            base_x=self.state.base_x,
            vector_polarity=self.state.vector_polarity,
            time_high=self.state.time_high,
            time_low=self.state.time_low,
            time_high_overflow=self.state.time_high_overflow,
            time_low_overflow=self.state.time_low_overflow,
        )

        if len(y_positions) > 0:
            self.state.y = int(words[y_positions[-1]] & _ADDRESS_MASK)

        if len(low_positions) > 0:
            raw_lows = payloads[low_positions]
            self.state.time_low = int(raw_lows[-1])
            self.state.time_low_overflow = self._last_overflow(
                raw_lows,
                initial_value=self._decode_initial_state.time_low,
                initial_overflow=self._decode_initial_state.time_low_overflow,
                wrap_increment=_TIME_LOW_RANGE,
            )

        if len(high_positions) > 0:
            raw_highs = payloads[high_positions]
            self.state.time_high = int(raw_highs[-1])
            self.state.time_high_overflow = self._last_overflow(
                raw_highs,
                initial_value=self._decode_initial_state.time_high,
                initial_overflow=self._decode_initial_state.time_high_overflow,
                wrap_increment=_TIME_HIGH_RANGE_US,
            )

        total_vector_size = int(vector_sizes.sum())
        if len(base_positions) > 0:
            last_base_position = int(base_positions[-1])
            last_base_x = int(words[last_base_position] & _ADDRESS_MASK)
            vector_width_after_last_base = int(vector_sizes[last_base_position + 1 :].sum())
            self.state.base_x = last_base_x + vector_width_after_last_base
            self.state.vector_polarity = bool(words[last_base_position] & _POLARITY_BIT)
        else:
            self.state.base_x += total_vector_size
