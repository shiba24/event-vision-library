"""Live preview for the EVK4 event camera.

Shows decoded events in a window while the camera is moved, so it's
possible to see directly whether sparse output is expected sensor
behavior (only motion/edges generate events) or an actual problem.

Everything read off the device is also written to a ``.raw`` file as it
arrives, so the session can be re-decoded afterward with
``decode_video.py`` for a lossless, full-resolution result -- this live
view intentionally drops stale frames for responsiveness (see
``LiveEvt3Reader`` and the decode loop in ``live_view()`` below), but the
``.raw`` file itself is byte-identical to what a plain recorder would
produce.

Usage:
    python3 live_view.py --output live_session.raw

Press 'q' in the preview window to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from evlib.codec.usb_link import EVK4Link

sys.path.insert(0, str(Path("./event-vision-library/src")))
from evlib.codec.fileformat._evt3 import Evt3RawReader
from decode_video import FrameAccumulator

_DEFAULT_SLICE_US = 20_000.0
_DEFAULT_CHUNK_SIZE = 16384
_DEFAULT_CHUNKS_PER_FLUSH = 20


class LiveEvt3Reader(Evt3RawReader):
    """Evt3RawReader variant for a file another process is still appending to.

    The base Evt3RawReader sets ``self._finished = True`` permanently on
    any empty read, which is correct for a closed file but wrong for a
    ``.raw`` file that ``live_view()`` is writing to concurrently. This
    override treats an empty read as "nothing new yet" rather than "the
    stream is over forever".
    """

    def _read_word_block(self) -> npt.NDArray[np.uint16] | None:
        data = self.file.read(self.read_size)
        if not data:
            # Transient: no new data currently on disk. Do NOT set
            # self._finished -- more bytes may show up on the next call.
            return None

        if self._pending_byte:
            data = self._pending_byte + data
            self._pending_byte = b""

        usable_size = len(data) - (len(data) % 2)
        if usable_size != len(data):
            self._pending_byte = data[-1:]
            data = data[:usable_size]

        if not data:
            return None
        return np.frombuffer(data, dtype="<u2")


def live_view(
    output_raw_path: str = "live_session.raw",
    width: int = 1280,
    height: int = 720,
    slice_us: float = _DEFAULT_SLICE_US,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunks_per_flush: int = _DEFAULT_CHUNKS_PER_FLUSH,
    attempt_run_start: bool = True,
    window_name: str = "EVK4 Live View (press q to stop)",
) -> None:
    """Stream, record, and preview EVT3 events from the EVK4 in real time.

    Args:
        output_raw_path: Path the raw EVT3 byte stream is written to.
        width: Sensor width in pixels (EVK4 is 1280).
        height: Sensor height in pixels (EVK4 is 720).
        slice_us: Event accumulation window, in microseconds, per rendered
            frame.
        chunk_size: Bytes requested per USB read from EVK4Link.stream().
        chunks_per_flush: Number of USB reads between disk flush() calls.
            Flushing every read forces a disk sync per chunk and can stall
            the capture loop; batching flushes trades a small durability
            window for throughput.
        attempt_run_start: Whether to call EVK4Link.run_start() before
            streaming begins.
        window_name: Title of the OpenCV preview window.
    """
    link = EVK4Link()

    if attempt_run_start:
        print("Attempting run_start() streaming init...")
        link.run_start()

    # Create the file first (even empty) so the reader can open it.
    open(output_raw_path, "wb").close()

    f_out = open(output_raw_path, "ab")
    reader = LiveEvt3Reader(output_raw_path)
    accumulator = FrameAccumulator(width=width, height=height, slice_us=slice_us)

    print(f"Live view running -- writing to {output_raw_path}, press 'q' in the window to stop.")
    bytes_written = 0
    events_seen = 0
    chunk_counter = 0

    try:
        for raw in link.stream(chunk_size=chunk_size):
            if raw:
                f_out.write(raw)
                chunk_counter += 1
                if chunk_counter % chunks_per_flush == 0:
                    f_out.flush()
                bytes_written += len(raw)

            # Decode every chunk currently available, not just one -- a
            # single next_chunk() per USB read lets the decode backlog
            # grow silently whenever bytes arrive faster than one chunk
            # per iteration, causing the preview to drift behind real
            # camera motion.
            latest_frame = None
            while True:
                try:
                    chunk = reader.next_chunk()
                except StopIteration:
                    break

                events_seen += len(chunk)
                for frame in accumulator.add_chunk(chunk):
                    # Keep only the newest frame for display -- pushing
                    # every intermediate frame to cv2.imshow during a
                    # catch-up burst is what stalls the GUI. The .raw
                    # file on disk still has every event for offline,
                    # lossless reconstruction via decode_video.py.
                    latest_frame = frame

            if latest_frame is not None:
                cv2.imshow(window_name, latest_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        f_out.flush()
        f_out.close()
        reader.close()
        cv2.destroyAllWindows()

    print(f"Stopped. {bytes_written} bytes recorded, {events_seen} events decoded live -> {output_raw_path}")
    if bytes_written == 0:
        print("WARNING: no bytes received -- same run_start()/endpoint issue as before, not a live-view problem.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live preview of EVK4 events while recording")
    p.add_argument("--output", default="live_session.raw")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--slice-us", type=float, default=_DEFAULT_SLICE_US)
    p.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunks-per-flush", type=int, default=_DEFAULT_CHUNKS_PER_FLUSH)
    p.add_argument("--no-run-start", action="store_true")
    args = p.parse_args()

    live_view(
        output_raw_path=args.output,
        width=args.width,
        height=args.height,
        slice_us=args.slice_us,
        chunk_size=args.chunk_size,
        chunks_per_flush=args.chunks_per_flush,
        attempt_run_start=not args.no_run_start,
    )