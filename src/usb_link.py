"""Low-level pyusb connection to the EVK4 (04b4:00f5).

Implements the TZ property request/response framing used to talk to the
device, plus bulk-endpoint discovery and the streaming data path.

Everything in here is either:
  (a) verified against hardware already (device discovery, reg_read), or
  (b) explicitly marked UNVERIFIED (run_start) pending confirmation on
      real hardware.

No OpenEB source is read or referenced at runtime -- this is a standalone
reimplementation of the wire protocol.
"""

from __future__ import annotations

import struct
from typing import Iterator

import usb.core
import usb.util

VENDOR_ID = 0x04B4
PRODUCT_ID = 0x00F5

# TZ property IDs, from the request/response framing already confirmed
# working against hardware.
PROP_REG32 = 0x10102
PROP_WRITE_FLAG = 0x40000000
PROP_REJECT_FLAG = 0x80000000

_TZ_RESPONSE_SIZE = 512
_TZ_HEADER_FORMAT = "<II"
_USB_TIMEOUT_ERRNOS = (110, 60)  # ETIMEDOUT variants raised by libusb backends


class EVK4Link:
    """USB bulk-transfer link to an EVK4 event camera."""

    def __init__(self, vendor_id: int = VENDOR_ID, product_id: int = PRODUCT_ID) -> None:
        """Open the EVK4 device and resolve its bulk endpoints.

        Args:
            vendor_id: USB vendor ID to search for.
            product_id: USB product ID to search for.

        Raises:
            RuntimeError: If no matching device is found, or if the
                expected bulk endpoint layout (1 OUT, 2 IN) isn't present.
        """
        self.dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if self.dev is None:
            raise RuntimeError(
                f"EVK4 not found ({vendor_id:#06x}:{product_id:#06x}). "
                f"Check `system_profiler SPUSBDataType | grep -A15 0x04b4`."
            )
        self.dev.set_configuration()
        intf = self.dev.get_active_configuration()[(0, 0)]
        bulk_eps = [
            e for e in intf
            if usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK
        ]
        if len(bulk_eps) < 3:
            raise RuntimeError(
                f"Expected 3 bulk endpoints (ctrl-out, ctrl-in, data-in), found {len(bulk_eps)}."
            )

        out_eps = [e for e in bulk_eps if usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT]
        in_eps = [e for e in bulk_eps if usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN]

        if len(out_eps) != 1 or len(in_eps) != 2:
            raise RuntimeError(
                f"Expected exactly 1 OUT and 2 IN bulk endpoints, "
                f"found {len(out_eps)} OUT and {len(in_eps)} IN. "
                f"Run inspect_endpoints.py and adjust endpoint selection manually."
            )

        self.ep_out = out_eps[0]
        out_number = self.ep_out.bEndpointAddress & 0x0F

        # The IN endpoint sharing the OUT endpoint's number is treated as
        # the paired control-response channel; the other IN endpoint is
        # the data stream. This matches the observed layout (0x02 OUT /
        # 0x82 IN paired, 0x81 IN as data) but is inferred from numbering,
        # not verified against firmware documentation -- worth revisiting
        # if streaming still doesn't work.
        matching = [e for e in in_eps if (e.bEndpointAddress & 0x0F) == out_number]
        if len(matching) == 1:
            self.ep_in = matching[0]
            self.ep_data = [e for e in in_eps if e is not self.ep_in][0]
        else:
            # Fall back to address order if the numbering heuristic
            # doesn't apply.
            self.ep_in, self.ep_data = sorted(in_eps, key=lambda e: e.bEndpointAddress)

        print(
            f"Resolved endpoints: OUT={self.ep_out.bEndpointAddress:#04x}, "
            f"ctrl-IN={self.ep_in.bEndpointAddress:#04x}, "
            f"data-IN={self.ep_data.bEndpointAddress:#04x}"
        )

    # -- low-level property request/response ---------------------------------

    def tz(
        self,
        prop: int,
        payload: bytes = b"",
        timeout_write: int = 1000,
        timeout_read: int = 10000,
    ) -> bytes:
        """Send a TZ property request and return its response payload.

        Args:
            prop: TZ property ID, optionally OR'd with PROP_WRITE_FLAG.
            payload: Request payload bytes.
            timeout_write: Write timeout, in milliseconds.
            timeout_read: Read timeout, in milliseconds.

        Returns:
            The response payload, with the 8-byte TZ header stripped.

        Raises:
            RuntimeError: If the device rejects the property request.
        """
        header = struct.pack(_TZ_HEADER_FORMAT, prop, len(payload))
        self.dev.write(self.ep_out.bEndpointAddress, header + payload, timeout=timeout_write)
        resp = bytes(self.dev.read(self.ep_in.bEndpointAddress, _TZ_RESPONSE_SIZE, timeout=timeout_read))
        p, sz = struct.unpack(_TZ_HEADER_FORMAT, resp[:8])
        if p & PROP_REJECT_FLAG:
            raise RuntimeError(f"device rejected property {prop:#x}")
        return resp[8:8 + sz]

    # -- register access (verified read path) ---------------------------------

    def reg_read(self, addr: int, device_id: int = 0) -> int:
        """Read a 32-bit device register.

        Args:
            addr: Register address.
            device_id: TZ device ID the register belongs to.

        Returns:
            The register's current 32-bit value.
        """
        raw = self.tz(PROP_REG32, struct.pack("<III", device_id, addr, 1))
        return struct.unpack("<I", raw[8:12])[0]

    def reg_write(self, addr: int, val: int, device_id: int = 0) -> None:
        """Write a 32-bit device register.

        Args:
            addr: Register address.
            val: Value to write.
            device_id: TZ device ID the register belongs to.
        """
        self.tz(PROP_REG32 | PROP_WRITE_FLAG, struct.pack("<III", device_id, addr, val))

    def reg_write_field(self, addr: int, data: int, mask: int, device_id: int = 0) -> None:
        """Read-modify-write a masked field within a register.

        Args:
            addr: Register address.
            data: New field value (only bits under ``mask`` are used).
            mask: Bitmask selecting which bits of the register to modify.
            device_id: TZ device ID the register belongs to.
        """
        current = self.reg_read(addr, device_id)
        new_val = (current & ~mask) | (data & mask)
        self.reg_write(addr, new_val, device_id)

    # -- streaming control ------------------------------------------------------

    def run_start(self) -> None:
        """Run the short streaming-init register sequence.

        UNVERIFIED on hardware as of this module being written. This is
        the short 5-op sequence, not the full ~40-op cold-boot init. Try
        this first; if the data endpoint stays empty, the full init table
        is likely needed instead (not included here).
        """
        self.reg_write(0x0000B000, 0x000002F9)
        self.reg_write(0x00009028, 0x00000000)
        self.reg_write_field(0x00009008, 0x645, 0x00000001)
        self.reg_write(0x0000002C, 0x0022C724)          # Analog START
        self.reg_write_field(0x00000004, 0xF0005442, 0x00000400)

    def read_data(self, size: int = 16384, timeout: int = 2000) -> bytes:
        """Pull one chunk of raw bulk data off the streaming endpoint.

        Args:
            size: Maximum number of bytes to read.
            timeout: Read timeout, in milliseconds.

        Returns:
            The bytes read from the data endpoint.
        """
        return bytes(self.dev.read(self.ep_data.bEndpointAddress, size, timeout=timeout))

    def stream(self, chunk_size: int = 16384, timeout: int = 2000) -> Iterator[bytes]:
        """Yield raw byte chunks from the device until interrupted or a read fails.

        Args:
            chunk_size: Maximum number of bytes requested per read.
            timeout: Read timeout, in milliseconds.

        Yields:
            Raw bytes read from the data endpoint. An empty ``bytes``
            object is yielded on a read timeout with no data, which is
            common when the sensor is idle -- this surfaces the timeout
            to the caller rather than treating it as end of stream.

        Raises:
            usb.core.USBError: For any USB error other than a timeout.
        """
        while True:
            try:
                yield self.read_data(chunk_size, timeout)
            except usb.core.USBError as e:
                if e.errno in _USB_TIMEOUT_ERRNOS:
                    yield b""
                    continue
                raise