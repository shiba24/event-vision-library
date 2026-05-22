"""AEDAT4 iterator tests."""

import os
import struct

import flatbuffers
import numpy as np

from evlib.codec import fileformat


def _build_header(stream_metadata_xml: str, data_table_position: int = -1) -> bytes:
    builder = flatbuffers.Builder(0)
    info_node = builder.CreateString(stream_metadata_xml)
    builder.StartObject(3)
    builder.PrependInt32Slot(0, 0, 0)
    builder.PrependInt64Slot(1, data_table_position, -1)
    builder.PrependUOffsetTRelativeSlot(2, info_node, 0)
    root_table = builder.EndObject()
    builder.Finish(root_table, file_identifier=b"IOHE")
    header_flatbuffer = bytes(builder.Output())
    return struct.pack("<I", len(header_flatbuffer)) + header_flatbuffer


def _build_packet_header(
    builder: flatbuffers.Builder,
    stream_id: int,
    payload_size: int,
) -> int:
    builder.Prep(4, 8)
    builder.PrependInt32(payload_size)
    builder.PrependInt32(stream_id)
    return int(builder.Offset())


def _build_data_table(data_table_entries):  # type: ignore
    builder = flatbuffers.Builder(0)
    entry_offsets = []
    for byte_offset, stream_id, payload_size in data_table_entries:
        packet_header_offset = _build_packet_header(builder, stream_id, payload_size)
        builder.StartObject(5)
        builder.PrependStructSlot(1, packet_header_offset, 0)
        builder.PrependInt64Slot(0, byte_offset, 0)
        entry_offsets.append(builder.EndObject())

    builder.StartVector(4, len(entry_offsets), 4)
    for entry_offset in reversed(entry_offsets):
        builder.PrependUOffsetTRelative(entry_offset)
    table_entries = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, table_entries, 0)
    root_table = builder.EndObject()
    builder.FinishSizePrefixed(root_table, file_identifier=b"FTAB")
    return bytes(builder.Output())


def _build_event_packet(events):  # type: ignore
    builder = flatbuffers.Builder(0)
    builder.StartVector(16, len(events), 8)
    for timestamp, x, y, polarity in reversed(events):
        builder.Prep(8, 16)
        builder.PrependInt16(0)
        builder.PrependUint8(0)
        builder.PrependBool(polarity)
        builder.PrependInt16(y)
        builder.PrependInt16(x)
        builder.PrependInt64(timestamp)
    event_elements = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, event_elements, 0)
    root_table = builder.EndObject()
    builder.FinishSizePrefixed(root_table, file_identifier=b"EVTS")
    return bytes(builder.Output())


def _build_frame(timestamp: int, image: np.ndarray, format_id: int = 0) -> bytes:
    builder = flatbuffers.Builder(0)
    pixel_data = builder.CreateByteVector(image.astype(np.uint8, copy=False).ravel().tobytes())
    builder.StartObject(13)
    builder.PrependInt64Slot(0, timestamp, 0)
    builder.PrependInt8Slot(5, format_id, 0)
    builder.PrependInt16Slot(6, image.shape[1], 0)
    builder.PrependInt16Slot(7, image.shape[0], 0)
    builder.PrependUOffsetTRelativeSlot(10, pixel_data, 0)
    root_table = builder.EndObject()
    builder.FinishSizePrefixed(root_table, file_identifier=b"FRME")
    return bytes(builder.Output())


def _build_imu_packet(imu_rows):  # type: ignore
    builder = flatbuffers.Builder(0)
    row_offsets = []
    for imu_row in imu_rows:
        builder.StartObject(11)
        builder.PrependInt64Slot(0, imu_row[0], 0)
        builder.PrependFloat32Slot(2, imu_row[1], 0.0)
        builder.PrependFloat32Slot(3, imu_row[2], 0.0)
        builder.PrependFloat32Slot(4, imu_row[3], 0.0)
        builder.PrependFloat32Slot(5, imu_row[4], 0.0)
        builder.PrependFloat32Slot(6, imu_row[5], 0.0)
        builder.PrependFloat32Slot(7, imu_row[6], 0.0)
        row_offsets.append(builder.EndObject())
    builder.StartVector(4, len(row_offsets), 4)
    for row_offset in reversed(row_offsets):
        builder.PrependUOffsetTRelative(row_offset)
    imu_elements = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, imu_elements, 0)
    root_table = builder.EndObject()
    builder.FinishSizePrefixed(root_table, file_identifier=b"IMUS")
    return bytes(builder.Output())


def _build_trigger_packet(trigger_rows):  # type: ignore
    builder = flatbuffers.Builder(0)
    row_offsets = []
    for timestamp, trigger_type in trigger_rows:
        builder.StartObject(2)
        builder.PrependInt64Slot(0, timestamp, 0)
        builder.PrependInt8Slot(1, trigger_type, 0)
        row_offsets.append(builder.EndObject())
    builder.StartVector(4, len(row_offsets), 4)
    for row_offset in reversed(row_offsets):
        builder.PrependUOffsetTRelative(row_offset)
    trigger_elements = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, trigger_elements, 0)
    root_table = builder.EndObject()
    builder.FinishSizePrefixed(root_table, file_identifier=b"TRIG")
    return bytes(builder.Output())


def _write_packet(file_handle, stream_id: int, payload: bytes):  # type: ignore
    file_handle.write(struct.pack("<ii", stream_id, len(payload)))
    file_handle.write(payload)


def _write_aedat4(path: str, with_data_table: bool = False) -> None:
    stream_metadata_xml = """<dv><node name="outInfo">
        <node name="0">
            <attr key="originalOutputName" type="string">events</attr>
            <attr key="typeIdentifier" type="string">EVTS</attr>
            <node name="info"><attr key="source" type="string">cam</attr></node>
        </node>
        <node name="1">
            <attr key="originalOutputName" type="string">frames</attr>
            <attr key="typeIdentifier" type="string">FRME</attr>
            <node name="info"><attr key="source" type="string">cam</attr></node>
        </node>
        <node name="2">
            <attr key="originalOutputName" type="string">imu</attr>
            <attr key="typeIdentifier" type="string">IMUS</attr>
            <node name="info"><attr key="source" type="string">cam</attr></node>
        </node>
        <node name="3">
            <attr key="originalOutputName" type="string">triggers</attr>
            <attr key="typeIdentifier" type="string">TRIG</attr>
            <node name="info"><attr key="source" type="string">cam</attr></node>
        </node>
    </node></dv>"""
    packets = [
        (1, _build_frame(10, np.arange(6, dtype=np.uint8).reshape(2, 3))),
        (0, _build_event_packet([(1, 2, 3, True), (4, 5, 6, False)])),
        (2, _build_imu_packet([(7, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3)])),
        (3, _build_trigger_packet([(8, 2)])),
    ]

    data_table_position = -1
    data_table = b""
    while True:
        aedat_header = _build_header(stream_metadata_xml, data_table_position)
        next_packet_offset = len(b"#!AER-DAT4.0\r\n") + len(aedat_header)
        data_table_entries = []
        for stream_id, payload in packets:
            data_table_entries.append((next_packet_offset + 8, stream_id, len(payload)))
            next_packet_offset += 8 + len(payload)

        if not with_data_table:
            break

        data_table = _build_data_table(data_table_entries)
        if data_table_position == next_packet_offset:
            break
        data_table_position = next_packet_offset

    with open(path, "wb") as file_handle:
        file_handle.write(b"#!AER-DAT4.0\r\n")
        file_handle.write(aedat_header)
        for stream_id, payload in packets:
            _write_packet(file_handle, stream_id, payload)
        file_handle.write(data_table)


def test_iterator_aedat4_event_dtype_and_values(tmp_path):  # type: ignore
    """Read AEDAT4 event packets into RawEvents with stable dtypes."""
    path = os.path.join(tmp_path, "sample.aedat4")
    _write_aedat4(path)

    events = next(fileformat.IteratorAedat4Event(path))

    assert events.x.dtype.type is np.int16
    assert events.y.dtype.type is np.int16
    assert events.t.dtype.type is np.float64
    assert events.p.dtype.type is np.bool_
    assert events.x.tolist() == [2, 5]
    assert events.y.tolist() == [3, 6]
    assert events.t.tolist() == [1.0, 4.0]
    assert events.p.tolist() == [True, False]


def test_iterator_aedat4_frame_imu_and_trigger(tmp_path):  # type: ignore
    """Read AEDAT4 frame, IMU, and trigger streams."""
    path = os.path.join(tmp_path, "sample.aedat4")
    _write_aedat4(path)

    frame = next(fileformat.IteratorAedat4Frame(path))
    imu = next(fileformat.IteratorAedat4Imu(path))
    trigger = next(fileformat.IteratorAedat4Trigger(path))

    assert frame["t"].tolist() == [10.0]
    assert frame["frame"].shape == (1, 2, 3, 1)
    assert frame["frame"].dtype.type is np.uint8
    assert imu["num"] == 1
    assert np.allclose(imu["imu"], np.array([7.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3]))
    assert trigger["num"] == 1
    assert trigger["trigger"].tolist() == [[8.0, 2.0]]


def test_iterator_aedat4_frame_preserves_bgr_order(tmp_path):  # type: ignore
    """Read AEDAT4 BGR frames without channel conversion."""
    path = os.path.join(tmp_path, "sample-color.aedat4")
    stream_metadata_xml = """<dv><node name="outInfo">
        <node name="0">
            <attr key="originalOutputName" type="string">frames</attr>
            <attr key="typeIdentifier" type="string">FRME</attr>
            <node name="info"><attr key="source" type="string">cam</attr></node>
        </node>
    </node></dv>"""
    bgr_frame = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    with open(path, "wb") as file_handle:
        file_handle.write(b"#!AER-DAT4.0\r\n")
        file_handle.write(_build_header(stream_metadata_xml))
        _write_packet(file_handle, 0, _build_frame(12, bgr_frame, format_id=16))

    frame = next(fileformat.IteratorAedat4Frame(path))

    np.testing.assert_array_equal(frame["frame"][0], bgr_frame)


def test_iterator_aedat4_event_reads_with_file_data_table(tmp_path):  # type: ignore
    """Read AEDAT4 event packets through a FileDataTable index."""
    path = os.path.join(tmp_path, "sample-indexed.aedat4")
    _write_aedat4(path, with_data_table=True)

    iterator = fileformat.IteratorAedat4Event(path)
    try:
        assert iterator.reader._packet_index
        iterator.reader.file.seek(iterator.reader._end_of_packets)

        events = next(iterator)

        assert len(events) == 2
        assert events.t[-1] == 4.0
    finally:
        iterator.close()
