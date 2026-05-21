from __future__ import annotations

import os
import struct
from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import BinaryIO
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import numpy as np
from defusedxml import ElementTree

from ...types import RawEvents


AEDAT4_VERSION = b"#!AER-DAT4.0\r\n"
PACKET_HEADER_SIZE = 8
_PACKET_HEADER = struct.Struct("<ii")


class Aedat4Error(RuntimeError):
    """Raised when an AEDAT4 file cannot be parsed."""


@dataclass(frozen=True)
class StreamInfo:
    """Metadata for a single AEDAT4 stream."""

    stream_id: int
    name: str
    type_identifier: str
    source: str | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class PacketHeader:
    """AEDAT4 packet header."""

    stream_id: int
    payload_size: int


@dataclass(frozen=True)
class IOHeader:
    """Parsed AEDAT4 IOHeader fields."""

    compression: int
    data_table_position: int
    info_node: str


@dataclass(frozen=True)
class PacketIndex:
    """Byte location for one AEDAT4 packet body."""

    data_offset: int
    payload_size: int


class _FlatBuffer:
    """Small FlatBuffer reader for the AEDAT4 schemas used here."""

    def __init__(self, data_buffer: bytes, *, size_prefixed: bool = False) -> None:
        self.buffer = data_buffer
        self.base = 4 if size_prefixed else 0
        if size_prefixed:
            declared_size = self._u32(0)
            if declared_size != len(data_buffer) - 4:
                raise Aedat4Error(
                    f"FlatBuffer size prefix declares {declared_size} bytes, got "
                    f"{len(data_buffer) - 4}"
                )
        if len(data_buffer) < self.base + 8:
            raise Aedat4Error("FlatBuffer is too small")
        self.root_table = self.base + self._u32(self.base)
        self._check_bounds(self.root_table, 4)
        self.vtable_start = self.root_table - self._i32(self.root_table)
        self._check_bounds(self.vtable_start, 4)

    def identifier(self) -> bytes:
        return self.buffer[self.base + 4 : self.base + 8]

    def require_identifier(self, expected: bytes) -> None:
        actual = self.identifier()
        if actual != expected:
            raise Aedat4Error(f"Expected FlatBuffer identifier {expected!r}, got {actual!r}")

    def scalar(self, field: int, struct_format: str, default: Any) -> Any:
        field_offset = self._field_location(field)
        if field_offset is None:
            return default
        return struct.unpack_from(struct_format, self.buffer, field_offset)[0]

    def string(self, field: int, default: str = "") -> str:
        field_offset = self._field_location(field)
        if field_offset is None:
            return default
        string_offset = field_offset + self._u32(field_offset)
        string_length = self._u32(string_offset)
        string_start = string_offset + 4
        string_end = string_start + string_length
        self._check_bounds(string_start, string_length)
        return self.buffer[string_start:string_end].decode("utf-8")

    def bytes_vector(self, field: int) -> memoryview:
        vector_start, vector_length = self._vector_data(field)
        self._check_bounds(vector_start, vector_length)
        return memoryview(self.buffer)[vector_start : vector_start + vector_length]

    def struct_vector(self, field: int, item_size: int) -> tuple[int, int]:
        vector_start, vector_length = self._vector_data(field)
        self._check_bounds(vector_start, vector_length * item_size)
        return vector_start, vector_length

    def table_vector_range(self, field: int) -> tuple[int, int]:
        vector_start, vector_length = self._vector_data(field)
        self._check_bounds(vector_start, vector_length * 4)
        return vector_start, vector_length

    def table_from_vector(self, vector_start: int, table_index: int) -> int:
        table_reference_offset = vector_start + (table_index * 4)
        return table_reference_offset + self._u32(table_reference_offset)

    def table_scalar(
        self,
        table_offset: int,
        field: int,
        struct_format: str,
        default: Any,
    ) -> Any:
        field_offset = self._field_location_from_table(table_offset, field)
        if field_offset is None:
            return default
        return struct.unpack_from(struct_format, self.buffer, field_offset)[0]

    def table_struct_location(self, table_offset: int, field: int) -> int | None:
        return self._field_location_from_table(table_offset, field)

    def _vector_data(self, field: int) -> tuple[int, int]:
        field_offset = self._field_location(field)
        if field_offset is None:
            return 0, 0
        vector_offset = field_offset + self._u32(field_offset)
        vector_length = self._u32(vector_offset)
        return vector_offset + 4, vector_length

    def _field_location(self, field: int) -> int | None:
        return self._field_location_from_table(self.root_table, field)

    def _field_location_from_table(self, table_offset: int, field: int) -> int | None:
        vtable_start = table_offset - self._i32(table_offset)
        self._check_bounds(vtable_start, 4)
        vtable_size = self._u16(vtable_start)
        if field >= vtable_size:
            return None
        field_offset = self._u16(vtable_start + field)
        if field_offset == 0:
            return None
        return table_offset + field_offset

    def _check_bounds(self, offset: int, size: int) -> None:
        if offset < 0 or offset + size > len(self.buffer):
            raise Aedat4Error("FlatBuffer offset is out of bounds")

    def _u16(self, offset: int) -> int:
        self._check_bounds(offset, 2)
        return int(struct.unpack_from("<H", self.buffer, offset)[0])

    def _u32(self, offset: int) -> int:
        self._check_bounds(offset, 4)
        return int(struct.unpack_from("<I", self.buffer, offset)[0])

    def _i32(self, offset: int) -> int:
        self._check_bounds(offset, 4)
        return int(struct.unpack_from("<i", self.buffer, offset)[0])


class Aedat4Reader:
    """Sequential AEDAT4 reader only for evlib's supported streams."""

    _EVENT_PACKET = b"EVTS"
    _FRAME = b"FRME"
    _IMU_PACKET = b"IMUS"
    _TRIGGER_PACKET = b"TRIG"

    _STREAM_TYPES = {
        "events": _EVENT_PACKET.decode(),
        "frames": _FRAME.decode(),
        "imu": _IMU_PACKET.decode(),
        "triggers": _TRIGGER_PACKET.decode(),
    }

    # AEDAT4 events use a packed 16 byte struct. Padding keeps the dtype aligned
    # with the file layout.
    _EVENT_DTYPE = np.dtype(
        [
            ("timestamp", "<i8"),
            ("x", "<i2"),
            ("y", "<i2"),
            ("polarity", "u1"),
            ("_pad0", "u1"),
            ("_pad1", "<i2"),
        ]
    )

    # Frame ids group by channel count. 0, 8, 16, and 24 start the one to four
    # channel unsigned byte formats. Nearby ids use other element types.
    _FRAME_FORMATS: dict[int, tuple[np.dtype, int]] = {
        0: (np.dtype(np.uint8), 1),
        1: (np.dtype(np.int8), 1),
        2: (np.dtype("<u2"), 1),
        3: (np.dtype("<i2"), 1),
        4: (np.dtype("<i4"), 1),
        5: (np.dtype("<f4"), 1),
        6: (np.dtype("<f8"), 1),
        8: (np.dtype(np.uint8), 2),
        9: (np.dtype(np.int8), 2),
        10: (np.dtype("<u2"), 2),
        11: (np.dtype("<i2"), 2),
        12: (np.dtype("<i4"), 2),
        13: (np.dtype("<f4"), 2),
        14: (np.dtype("<f8"), 2),
        16: (np.dtype(np.uint8), 3),
        17: (np.dtype(np.int8), 3),
        18: (np.dtype("<u2"), 3),
        19: (np.dtype("<i2"), 3),
        20: (np.dtype("<i4"), 3),
        21: (np.dtype("<f4"), 3),
        22: (np.dtype("<f8"), 3),
        24: (np.dtype(np.uint8), 4),
        25: (np.dtype(np.int8), 4),
        26: (np.dtype("<u2"), 4),
        27: (np.dtype("<i2"), 4),
        28: (np.dtype("<i4"), 4),
        29: (np.dtype("<f4"), 4),
        30: (np.dtype("<f8"), 4),
    }

    def __init__(self, file_name: str) -> None:
        """Open an AEDAT4 file and parse its stream metadata."""
        self.file_name = file_name
        self.file: BinaryIO = open(file_name, "rb")
        self.header = self._read_header()
        self._decompress = self._make_decompressor(self.header.compression)
        self.streams = self._parse_streams(self.header.info_node)
        self._streams_by_kind = self._resolve_streams_by_kind()
        self._file_size = os.fstat(self.file.fileno()).st_size
        self._first_packet_offset = self.file.tell()
        self._end_of_packets = self._packet_end_offset()
        self._packet_index = self._initial_packet_index()
        self._packet_index_positions_by_stream: dict[int, int] = {}
        self._imu_buffer: Deque[np.ndarray] = deque()
        self._trigger_buffer: Deque[np.ndarray] = deque()

    def close(self) -> None:
        """Close the underlying file handle."""
        self.file.close()

    def __enter__(self) -> Aedat4Reader:
        """Return the reader as a context manager."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the reader when leaving a context manager."""
        self.close()

    def reset(self) -> None:
        """Reset packet iteration state to the first AEDAT4 packet."""
        self.file.seek(self._first_packet_offset)
        self._packet_index_positions_by_stream.clear()
        self._imu_buffer.clear()
        self._trigger_buffer.clear()

    def next_events(self) -> RawEvents:
        """Return the next event packet.

        Returns:
            RawEvents: Event packet with x, y, timestamp, and polarity arrays.
        """
        payload = self._next_payload("events")
        return self._decode_events(payload)

    def next_frame(self) -> dict[str, Any]:
        """Return the next frame packet.

        Returns:
            dict[str, Any]: Dictionary with ``frame``, ``t``, and ``num``.
        """
        payload = self._next_payload("frames")
        return self._decode_frame(payload)

    def next_imu(self) -> dict[str, Any]:
        """Return the next IMU row.

        Returns:
            dict[str, Any]: Dictionary with ``imu`` and ``num``.
        """
        if not self._imu_buffer:
            payload = self._next_payload("imu")
            self._imu_buffer.extend(self._decode_imu(payload))
        imu = self._imu_buffer.popleft()
        return {"imu": imu, "num": 1}

    def next_trigger(self) -> dict[str, Any]:
        """Return the next trigger row.

        Returns:
            dict[str, Any]: Dictionary with ``trigger`` and ``num``.
        """
        if not self._trigger_buffer:
            payload = self._next_payload("triggers")
            self._trigger_buffer.extend(self._decode_triggers(payload))
        trigger = self._trigger_buffer.popleft()[None, :]
        return {"trigger": trigger, "num": len(trigger)}

    def _read_header(self) -> IOHeader:
        # AEDAT4 starts with magic text then an IOHeader FlatBuffer.
        # Packet payloads use size prefixes. The header does not.
        version_line = self.file.read(len(AEDAT4_VERSION))
        if version_line != AEDAT4_VERSION:
            raise Aedat4Error("AEDAT4.0: no valid version line found")
        header_size_bytes = self.file.read(4)
        if len(header_size_bytes) != 4:
            raise Aedat4Error("AEDAT4.0: missing IOHeader size")
        header_size = struct.unpack("<I", header_size_bytes)[0]
        header_data = self.file.read(header_size)
        if len(header_data) != header_size:
            raise Aedat4Error("AEDAT4.0: truncated IOHeader")
        flatbuffer = _FlatBuffer(header_data)
        flatbuffer.require_identifier(b"IOHE")
        return IOHeader(
            compression=flatbuffer.scalar(4, "<i", 0),
            data_table_position=flatbuffer.scalar(6, "<q", -1),
            info_node=flatbuffer.string(8),
        )

    def _parse_streams(self, info_node: str) -> dict[str, StreamInfo]:
        streams: dict[str, StreamInfo] = {}
        if not info_node:
            return streams

        metadata_root = ElementTree.fromstring(info_node)
        # Some writers wrap outInfo in a <dv> node.
        output_info_node = self._find_named_node(metadata_root, "outInfo")
        if output_info_node is None:
            return streams

        for stream_node in output_info_node.findall("node"):
            stream_id_text = stream_node.attrib.get("name")
            if stream_id_text is None:
                continue
            try:
                stream_id = int(stream_id_text)
            except ValueError:
                continue

            attributes = self._node_attributes(stream_node)
            stream_name = attributes.get("originalOutputName")
            type_identifier = attributes.get("typeIdentifier")
            if stream_name is None or type_identifier is None:
                continue

            stream_info_node = self._find_named_node(stream_node, "info")
            metadata = (
                self._node_attributes(stream_info_node) if stream_info_node is not None else {}
            )
            streams[stream_name] = StreamInfo(
                stream_id=stream_id,
                name=stream_name,
                type_identifier=type_identifier,
                source=metadata.get("source"),
                metadata=metadata,
            )

        return streams

    def _find_named_node(
        self,
        current_node: ElementTree.Element,
        name: str,
    ) -> ElementTree.Element | None:
        if current_node.tag == "node" and current_node.attrib.get("name") == name:
            return current_node
        for child_node in current_node:
            found = self._find_named_node(child_node, name)
            if found is not None:
                return found
        return None

    @staticmethod
    def _node_attributes(node: ElementTree.Element) -> dict[str, str]:
        return {
            attribute.attrib["key"]: attribute.text or ""
            for attribute in node.findall("attr")
            if "key" in attribute.attrib
        }

    def _resolve_streams_by_kind(self) -> dict[str, StreamInfo]:
        streams: dict[str, StreamInfo] = {}
        streams_by_type = {stream.type_identifier: stream for stream in self.streams.values()}
        for stream_name, expected_type in self._STREAM_TYPES.items():
            # Prefer known stream names. Fall back to stream type for custom names.
            stream = self.streams.get(stream_name) or streams_by_type.get(expected_type)
            if stream is not None:
                streams[stream_name] = stream
        return streams

    def _initial_packet_index(self) -> dict[int, list[PacketIndex]]:
        # Single stream files do not need FileDataTable. Sequential reads avoid
        # bad or missing tables.
        if self.header.data_table_position < 0 or len(self.streams) <= 1:
            return {}
        return self._read_packet_index()

    def _packet_end_offset(self) -> int:
        if self.header.data_table_position >= 0:
            return min(self._file_size, self.header.data_table_position)
        return self._file_size

    def _read_packet_index(self) -> dict[int, list[PacketIndex]]:
        if self.header.data_table_position < 0:
            return {}

        data_table_size = self._file_size - self.header.data_table_position
        if data_table_size <= 0:
            raise Aedat4Error("AEDAT4.0: FileDataTable is marked present but missing")

        original_position = self.file.tell()
        try:
            self.file.seek(self.header.data_table_position)
            data_table_payload = self.file.read(data_table_size)
        finally:
            self.file.seek(original_position)

        if len(data_table_payload) != data_table_size:
            raise Aedat4Error("AEDAT4.0: truncated FileDataTable")

        flatbuffer = _FlatBuffer(self._decompress(data_table_payload), size_prefixed=True)
        flatbuffer.require_identifier(b"FTAB")

        packet_index: dict[int, list[PacketIndex]] = {}
        table_vector_start, table_count = flatbuffer.table_vector_range(4)
        for table_index in range(table_count):
            entry_table = flatbuffer.table_from_vector(table_vector_start, table_index)
            data_offset = flatbuffer.table_scalar(entry_table, 4, "<q", -1)
            packet_header_offset = flatbuffer.table_struct_location(entry_table, 6)
            if data_offset < 0 or packet_header_offset is None:
                continue
            packet_stream_id, packet_size = _PACKET_HEADER.unpack_from(
                flatbuffer.buffer,
                packet_header_offset,
            )
            packet_end = data_offset + packet_size
            if (
                packet_size < 0
                or data_offset < self._first_packet_offset
                or packet_end > self._end_of_packets
            ):
                raise Aedat4Error("AEDAT4.0: corrupt FileDataTable entry")
            packet_index.setdefault(packet_stream_id, []).append(
                PacketIndex(data_offset=data_offset, payload_size=packet_size)
            )

        return packet_index

    def _next_payload(self, stream_name: str) -> bytes:
        stream = self._stream_for_name(stream_name)
        indexed_packets = self._packet_index.get(stream.stream_id)
        if indexed_packets is not None:
            packet_index_position = self._packet_index_positions_by_stream.get(stream.stream_id, 0)
            if packet_index_position >= len(indexed_packets):
                raise StopIteration
            packet = indexed_packets[packet_index_position]
            self._packet_index_positions_by_stream[stream.stream_id] = packet_index_position + 1
            self.file.seek(packet.data_offset)
            payload = self.file.read(packet.payload_size)
            if len(payload) != packet.payload_size:
                raise Aedat4Error("AEDAT4.0: truncated indexed packet body")
            return self._decompress(payload)

        while self.file.tell() + PACKET_HEADER_SIZE <= self._end_of_packets:
            packet_header = self._read_packet_header()
            packet_data_offset = self.file.tell()
            packet_end = packet_data_offset + packet_header.payload_size
            if packet_header.payload_size < 0 or packet_end > self._end_of_packets:
                raise Aedat4Error("AEDAT4.0: truncated or corrupt packet")
            if packet_header.stream_id != stream.stream_id:
                self.file.seek(packet_end)
                continue
            payload = self.file.read(packet_header.payload_size)
            if len(payload) != packet_header.payload_size:
                raise Aedat4Error("AEDAT4.0: truncated packet body")
            return self._decompress(payload)
        raise StopIteration

    def _read_packet_header(self) -> PacketHeader:
        header_bytes = self.file.read(PACKET_HEADER_SIZE)
        if len(header_bytes) != PACKET_HEADER_SIZE:
            raise StopIteration
        stream_id, payload_size = _PACKET_HEADER.unpack(header_bytes)
        return PacketHeader(stream_id=stream_id, payload_size=payload_size)

    def _stream_for_name(self, stream_name: str) -> StreamInfo:
        stream = self._streams_by_kind.get(stream_name)
        if stream is not None:
            return stream
        raise KeyError(f"AEDAT4 stream {stream_name!r} is not available")

    @staticmethod
    def _make_decompressor(compression: int) -> Callable[[bytes], bytes]:
        if compression == 0:
            return lambda payload: payload
        if compression in (1, 2):
            import lz4.frame

            return cast(Callable[[bytes], bytes], lz4.frame.decompress)
        if compression in (3, 4):
            try:
                import zstd

                return cast(Callable[[bytes], bytes], zstd.decompress)
            except ImportError:
                import zstandard

                return cast(Callable[[bytes], bytes], zstandard.ZstdDecompressor().decompress)
        raise Aedat4Error(f"Unsupported AEDAT4 compression type: {compression}")

    def _decode_events(self, payload: bytes) -> RawEvents:
        flatbuffer = _FlatBuffer(payload, size_prefixed=True)
        flatbuffer.require_identifier(self._EVENT_PACKET)
        event_vector_start, event_count = flatbuffer.struct_vector(4, self._EVENT_DTYPE.itemsize)
        events = np.frombuffer(
            payload,
            dtype=self._EVENT_DTYPE,
            count=event_count,
            offset=event_vector_start,
        )
        # Copy packet backed arrays before the decompressed payload is released.
        # Match RawEvents dtypes.
        return RawEvents(
            x=events["x"].copy(),
            y=events["y"].copy(),
            timestamp=events["timestamp"].astype(np.float64, copy=False),
            polarity=events["polarity"].view(np.bool_).copy(),
        )

    def _decode_frame(self, payload: bytes) -> dict[str, Any]:
        flatbuffer = _FlatBuffer(payload, size_prefixed=True)
        flatbuffer.require_identifier(self._FRAME)
        timestamp = flatbuffer.scalar(4, "<q", 0)
        format_id = flatbuffer.scalar(14, "<b", 0)
        width = flatbuffer.scalar(16, "<h", 0)
        height = flatbuffer.scalar(18, "<h", 0)
        if format_id not in self._FRAME_FORMATS:
            raise Aedat4Error(f"Unsupported AEDAT4 frame format: {format_id}")
        pixel_dtype, channel_count = self._FRAME_FORMATS[format_id]
        pixel_values = np.frombuffer(flatbuffer.bytes_vector(24), dtype=pixel_dtype)
        expected_pixel_count = width * height * channel_count
        if pixel_values.size != expected_pixel_count:
            shape_text = f"{width}x{height}x{channel_count}"
            raise Aedat4Error(
                f"Frame payload has {pixel_values.size} pixels, "
                f"expected {expected_pixel_count} for {shape_text}"
            )
        # Keep raw channel order to match existing iterator output.
        frame = pixel_values.reshape((height, width, channel_count))
        frame = frame[None]
        return {"frame": frame, "t": np.array([timestamp], dtype=np.float64), "num": 1}

    def _decode_imu(self, payload: bytes) -> np.ndarray:
        flatbuffer = _FlatBuffer(payload, size_prefixed=True)
        flatbuffer.require_identifier(self._IMU_PACKET)
        table_vector_start, imu_count = flatbuffer.table_vector_range(4)
        if imu_count == 0:
            raise StopIteration

        imu_rows = np.empty((imu_count, 7), dtype=np.float64)
        for row_index in range(imu_count):
            imu_table = flatbuffer.table_from_vector(table_vector_start, row_index)
            # evlib uses timestamp, accelerometer, and gyroscope. Temperature
            # and magnetometer stay ignored.
            imu_rows[row_index] = (
                flatbuffer.table_scalar(imu_table, 4, "<q", 0),
                flatbuffer.table_scalar(imu_table, 8, "<f", 0.0),
                flatbuffer.table_scalar(imu_table, 10, "<f", 0.0),
                flatbuffer.table_scalar(imu_table, 12, "<f", 0.0),
                flatbuffer.table_scalar(imu_table, 14, "<f", 0.0),
                flatbuffer.table_scalar(imu_table, 16, "<f", 0.0),
                flatbuffer.table_scalar(imu_table, 18, "<f", 0.0),
            )
        return imu_rows

    def _decode_triggers(self, payload: bytes) -> np.ndarray:
        flatbuffer = _FlatBuffer(payload, size_prefixed=True)
        flatbuffer.require_identifier(self._TRIGGER_PACKET)
        table_vector_start, trigger_count = flatbuffer.table_vector_range(4)
        if trigger_count == 0:
            raise StopIteration
        trigger_rows = np.empty((trigger_count, 2), dtype=np.float64)
        for row_index in range(trigger_count):
            trigger_table = flatbuffer.table_from_vector(table_vector_start, row_index)
            trigger_rows[row_index] = (
                flatbuffer.table_scalar(trigger_table, 4, "<q", 0),
                flatbuffer.table_scalar(trigger_table, 6, "<b", 0),
            )
        return trigger_rows
