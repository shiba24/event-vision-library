"""EVK4Link (usb_link.py) tests.

No real hardware needed: usb.core.find is patched to return a small
fake device object, and pyusb's real usb.util functions classify its
fake endpoints (they're pure bit-masking, no hardware required).
"""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest
import usb.core
import usb.util

from evlib.codec.usb_link import EVK4Link, PROP_REG32, PROP_REJECT_FLAG, PROP_WRITE_FLAG


class FakeEndpoint:
    def __init__(self, address: int, transfer_type: int = usb.util.ENDPOINT_TYPE_BULK) -> None:
        self.bEndpointAddress = address
        self.bmAttributes = transfer_type


def _make_device(endpoints: list[FakeEndpoint]) -> MagicMock:
    dev = MagicMock()
    dev.get_active_configuration.return_value = {(0, 0): endpoints}
    return dev


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

def test_raises_when_device_not_found() -> None:
    with patch("usb.core.find", return_value=None):
        with pytest.raises(RuntimeError, match="EVK4 not found"):
            EVK4Link()


def test_raises_when_fewer_than_three_bulk_endpoints() -> None:
    dev = _make_device([FakeEndpoint(0x02), FakeEndpoint(0x81)])
    with patch("usb.core.find", return_value=dev):
        with pytest.raises(RuntimeError, match="Expected 3 bulk endpoints"):
            EVK4Link()


def test_raises_when_endpoint_directions_are_wrong_shape() -> None:
    # 2 OUT, 1 IN instead of the required 1 OUT, 2 IN.
    dev = _make_device([FakeEndpoint(0x02), FakeEndpoint(0x03), FakeEndpoint(0x81)])
    with patch("usb.core.find", return_value=dev):
        with pytest.raises(RuntimeError, match="Expected exactly 1 OUT and 2 IN"):
            EVK4Link()


def test_correctly_assigns_endpoints_by_matching_number() -> None:
    # OUT=0x02, matching ctrl-IN=0x82, data-IN=0x81.
    dev = _make_device([FakeEndpoint(0x02), FakeEndpoint(0x82), FakeEndpoint(0x81)])
    with patch("usb.core.find", return_value=dev):
        link = EVK4Link()
    assert link.ep_out.bEndpointAddress == 0x02
    assert link.ep_in.bEndpointAddress == 0x82
    assert link.ep_data.bEndpointAddress == 0x81


def test_falls_back_to_address_order_when_no_number_match() -> None:
    dev = _make_device([FakeEndpoint(0x05), FakeEndpoint(0x83), FakeEndpoint(0x81)])
    with patch("usb.core.find", return_value=dev):
        link = EVK4Link()
    assert link.ep_in.bEndpointAddress == 0x81
    assert link.ep_data.bEndpointAddress == 0x83


# ---------------------------------------------------------------------------
# tz / register access
# ---------------------------------------------------------------------------

def _linked_with_mock_dev() -> EVK4Link:
    link = EVK4Link.__new__(EVK4Link)
    link.dev = MagicMock()
    link.ep_out = FakeEndpoint(0x02)
    link.ep_in = FakeEndpoint(0x82)
    link.ep_data = FakeEndpoint(0x81)
    return link


def test_tz_sends_correct_header_and_returns_stripped_payload() -> None:
    link = _linked_with_mock_dev()
    payload = b"\x01\x02\x03\x04"
    resp = struct.pack("<II", PROP_REG32, 4) + b"\xAA\xBB\xCC\xDD"
    link.dev.read.return_value = resp.ljust(512, b"\x00")

    result = link.tz(PROP_REG32, payload)

    sent = link.dev.write.call_args[0][1]
    assert sent == struct.pack("<II", PROP_REG32, len(payload)) + payload
    assert result == b"\xAA\xBB\xCC\xDD"


def test_tz_raises_when_device_rejects_property() -> None:
    link = _linked_with_mock_dev()
    resp = struct.pack("<II", PROP_REG32 | PROP_REJECT_FLAG, 0)
    link.dev.read.return_value = resp.ljust(512, b"\x00")

    with pytest.raises(RuntimeError, match="device rejected property"):
        link.tz(PROP_REG32)


def test_reg_read_returns_correct_int_value() -> None:
    link = _linked_with_mock_dev()
    resp = struct.pack("<II", PROP_REG32, 12) + b"\x00" * 8 + struct.pack("<I", 0xDEADBEEF)
    link.dev.read.return_value = resp.ljust(512, b"\x00")

    value = link.reg_read(0x1234)

    assert value == 0xDEADBEEF
    assert isinstance(value, int)


def test_reg_write_sets_write_flag_and_sends_value() -> None:
    link = _linked_with_mock_dev()
    link.dev.read.return_value = struct.pack("<II", 0, 0).ljust(512, b"\x00")

    link.reg_write(0x1234, 0x5678)

    sent = link.dev.write.call_args[0][1]
    prop, size = struct.unpack("<II", sent[:8])
    device_id, addr, val = struct.unpack("<III", sent[8:8 + size])
    assert prop == PROP_REG32 | PROP_WRITE_FLAG
    assert addr == 0x1234
    assert val == 0x5678


def test_reg_write_field_only_changes_masked_bits() -> None:
    link = _linked_with_mock_dev()
    with patch.object(link, "reg_read", return_value=0b1111_0000), \
         patch.object(link, "reg_write") as mock_write:
        link.reg_write_field(0x1234, data=0b0000_1010, mask=0b0000_1111)

    written_addr, written_val, _device_id = mock_write.call_args[0]
    assert written_addr == 0x1234
    assert written_val == 0b1111_1010


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def test_run_start_writes_expected_number_of_registers() -> None:
    link = _linked_with_mock_dev()
    with patch.object(link, "reg_write") as mock_write, \
         patch.object(link, "reg_write_field") as mock_write_field:
        link.run_start()

    assert mock_write.call_count == 3
    assert mock_write_field.call_count == 2


def test_read_data_returns_bytes_from_data_endpoint() -> None:
    link = _linked_with_mock_dev()
    link.dev.read.return_value = b"\x01\x02\x03"

    result = link.read_data(size=1024, timeout=500)

    assert result == b"\x01\x02\x03"
    link.dev.read.assert_called_once_with(link.ep_data.bEndpointAddress, 1024, timeout=500)


def test_stream_yields_empty_bytes_on_timeout_then_continues() -> None:
    link = _linked_with_mock_dev()
    timeout_error = usb.core.USBError("timed out")
    timeout_error.errno = 60
    link.dev.read.side_effect = [timeout_error, b"\x01\x02\x03"]

    gen = link.stream()

    assert next(gen) == b""
    assert next(gen) == b"\x01\x02\x03"


def test_stream_reraises_non_timeout_errors() -> None:
    link = _linked_with_mock_dev()
    error = usb.core.USBError("permission denied")
    error.errno = 13  # not a recognized timeout errno
    link.dev.read.side_effect = error

    with pytest.raises(usb.core.USBError):
        next(link.stream())