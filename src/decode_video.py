"""EVT3 to video renderer.

Runs Evt3RawReader (see ``_evt3.py``, unmodified) over a recorded EVT3
``.raw`` file and renders the decoded CD events into an MP4 video.

Frame model: events are binned into fixed-duration time slices (default
10ms of sensor time, from ``chunk.t``, which is in microseconds per
Evt3RawReader's timestamp semantics). Each slice becomes one output frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import cv2 # video i/o
import numpy as np
import numpy.typing as npt

from evlib.codec.fileformat._evt3 import Evt3EventChunk, Evt3RawReader

# Rendering defaults, named so they read the same way at every call site.
_DEFAULT_SLICE_US = 10_000.0
_DEFAULT_ON_COLOR = (255, 80, 0)   # BGR -- brightness increase, blue for on
_DEFAULT_OFF_COLOR = (0, 60, 255)  # BGR -- brightness decrease, ~red for off
_DEFAULT_INTENSITY_SCALE = 80.0    # stretches event counts to visible brightness range. multiplier applied to per-pixel event count before clipping to 0-255
_DEFAULT_STRENGTH_THRESHOLD = 20   # minimum brightness/strength (0-255) before a pixel is painted at all, reduces noise from very sparse events


class FrameAccumulator:
    """Bins decoded EVT3 CD events into time slices and renders BGR frames.

    Events are accumulated per pixel, per polarity, into float32 arrays.
    Each completed time slice is rendered into a BGR frame where pixel
    brightness reflects event count (denser motion/edges render brighter),
    with optional dilation for visibility and optional exponential decay
    for motion trails.
    """

    def __init__(
        self,
        width: int,
        height: int,
        slice_us: float = _DEFAULT_SLICE_US,
        dot_radius_px: int = 0,
        background: str = "white",
        on_color: tuple[int, int, int] = _DEFAULT_ON_COLOR,
        off_color: tuple[int, int, int] = _DEFAULT_OFF_COLOR,
        decay: float = 0.0,
        intensity_scale: float = _DEFAULT_INTENSITY_SCALE,
        strength_threshold: int = _DEFAULT_STRENGTH_THRESHOLD,
    ) -> None:
        """Configure a frame accumulator.

        Args:
            width: Output frame width in pixels (EVK4 sensor width is 1280).
            height: Output frame height in pixels (EVK4 sensor height is 720).
            slice_us: Sensor-time duration, in microseconds, binned into one
                output frame.
            dot_radius_px: Dilation radius in pixels for event visibility.
                0 disables dilation and renders raw, undilated pixels.
            background: Canvas fill color before events are painted, either
                "white" or "black".
            on_color: BGR color for ON events (brightness increase).
            off_color: BGR color for OFF events (brightness decrease).
            decay: Trail persistence applied between slices. 0.0 hard-clears
                every slice; values approaching 1.0 give long motion trails.
            intensity_scale: Multiplier applied to per-pixel event count
                before clipping to the 0-255 brightness range.
            strength_threshold: Minimum brightness (0-255) required before a
                pixel is painted, filtering out very sparse single events.
        """
        self.width = width
        self.height = height
        self.slice_us = slice_us
        self.dot_radius_px = max(0, dot_radius_px)
        self.background = background
        self.on_color = np.array(on_color, dtype=np.uint8)
        self.off_color = np.array(off_color, dtype=np.uint8)
        self.decay = decay
        self.intensity_scale = intensity_scale
        self.strength_threshold = strength_threshold

        self._on = np.zeros((height, width), dtype=np.float32)
        self._off = np.zeros((height, width), dtype=np.float32)
        self._start_ts: float | None = None
        #self._current_slice: int | None = None
        self._current_slice: int = 0

        # Precomputed once so render_frame() doesn't reallocate a kernel
        # every call.
        self._kernel: npt.NDArray[np.uint8] | None = None
        if self.dot_radius_px > 0:
            k = self.dot_radius_px * 2 + 1
            self._kernel = np.ones((k, k), np.uint8)

    def _clear_or_decay(self) -> None:
        if self.decay > 0:
            self._on *= self.decay
            self._off *= self.decay
        else:
            self._on.fill(0)
            self._off.fill(0)

    def add_chunk(self, chunk: Evt3EventChunk) -> Iterator[npt.NDArray[np.uint8]]:
        """Accumulate a decoded chunk and yield any completed frames.

        Args:
            chunk: Decoded CD events from Evt3RawReader.

        Yields:
            One rendered BGR frame per time slice completed by this chunk.
        """
        if len(chunk) == 0:
            return

        t, x, y, p = chunk.t, chunk.x, chunk.y, chunk.p

        if self._start_ts is None:
            self._start_ts = t[0]
            self._current_slice = 0

        slice_idx = np.floor((t - self._start_ts) / self.slice_us).astype(np.int64)

        # Split the chunk into runs of constant slice_idx.
        change_points = np.flatnonzero(np.diff(slice_idx)) + 1
        #boundaries = np.concatenate(([0], change_points, [len(t)]))
        boundaries = np.concatenate((np.array([0]), change_points, np.array([len(t)])))

        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            idx = int(slice_idx[s])

            if idx > self._current_slice:
                # Flush the current frame, plus any fully-empty slices
                # in between.
                for _ in range(idx - self._current_slice):
                    yield self.render_frame()
                    self._clear_or_decay()
                self._current_slice = idx

            xs, ys, ps = x[s:e], y[s:e], p[s:e]

            in_bounds = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            xs, ys, ps = xs[in_bounds], ys[in_bounds], ps[in_bounds]

            # Flatten (y, x) into a single index and use bincount for
            # scatter-accumulation -- substantially faster than np.add.at
            # at EVT3's typical event rates.
            on_flat = ys[ps].astype(np.int64) * self.width + xs[ps].astype(np.int64)
            off_flat = ys[~ps].astype(np.int64) * self.width + xs[~ps].astype(np.int64)

            if on_flat.size:
                counts = np.bincount(on_flat, minlength=self.width * self.height)
                self._on += counts.reshape(self.height, self.width)
            if off_flat.size:
                counts = np.bincount(off_flat, minlength=self.width * self.height)
                self._off += counts.reshape(self.height, self.width)

    def render_frame(self) -> npt.NDArray[np.uint8]:
        """Render the current accumulator state into a BGR frame.

        Returns:
            A ``(height, width, 3)`` uint8 BGR frame.
        """
        bg_val = 255 if self.background == "white" else 0
        frame = np.full((self.height, self.width, 3), bg_val, dtype=np.uint8)

        on_strength = np.clip(self._on * self.intensity_scale, 0, 255).astype(np.uint8)
        off_strength = np.clip(self._off * self.intensity_scale, 0, 255).astype(np.uint8)

        if self._kernel is not None:
            #on_strength = cv2.dilate(on_strength, self._kernel)
            cv2.dilate(on_strength, self._kernel, dst=on_strength)
            #off_strength = cv2.dilate(off_strength, self._kernel)
            cv2.dilate(off_strength, self._kernel, dst=off_strength)

        on_mask = on_strength > self.strength_threshold
        off_mask = off_strength > self.strength_threshold

        frame[on_mask] = self.on_color
        frame[off_mask] = self.off_color
        return frame

    def flush(self) -> Iterator[npt.NDArray[np.uint8]]:
        """Emit the final, partially-filled slice.

        Call once after the reader is exhausted -- ``add_chunk`` only
        emits a frame once a later slice index is observed, so the last
        slice otherwise never gets rendered.

        Yields:
            The final rendered BGR frame, if any events were accumulated.
        """
        if self._start_ts is not None:
            yield self.render_frame()


def decode_to_video(
    raw_path: str,
    output_video: str = "evk4_capture.mp4",
    width: int = 1280,
    height: int = 720,
    slice_us: float = _DEFAULT_SLICE_US,
    fps: float = 100.0,
    reader_chunk_size: int = 16384,
    dot_radius_px: int = 0,
    background: str = "white",
    decay: float = 0.0,
) -> str:
    """Decode an EVT3 ``.raw`` file into an MP4 video.

    Args:
        raw_path: Path to the recorded EVT3 ``.raw`` file.
        output_video: Output video file path.
        width: Sensor width in pixels (EVK4 is 1280).
        height: Sensor height in pixels (EVK4 is 720).
        slice_us: Event accumulation window, in microseconds.
        fps: Output video frame rate.
        reader_chunk_size: Events per chunk requested from Evt3RawReader.
        dot_radius_px: Dilation radius in pixels (0 = raw pixel detail).
        background: Canvas background, "white" or "black".
        decay: Trail persistence, 0.0-0.95.

    Returns:
        The output video path, unchanged from ``output_video``.
    """
    reader = Evt3RawReader(raw_path, chunk_size=reader_chunk_size)
    accumulator = FrameAccumulator(
        width=width,
        height=height,
        slice_us=slice_us,
        dot_radius_px=dot_radius_px,
        background=background,
        decay=decay,
    )
    writer = cv2.VideoWriter(
        #output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        output_video, cv2.VideoWriter.fourcc("m", "p", "4", "v"), fps, (width, height)
    )

    frames_written = 0
    events_seen = 0

    try:
        for chunk in reader.read_chunks():
            events_seen += len(chunk)
            for frame in accumulator.add_chunk(chunk):
                writer.write(frame)
                frames_written += 1
        for frame in accumulator.flush():
            writer.write(frame)
            frames_written += 1
    finally:
        reader.close()
        writer.release()

    print(f"Decoded {events_seen} events, wrote {frames_written} frames -> {output_video}")
    if events_seen == 0:
        print(
            "WARNING: zero events decoded. Check that the .raw file actually "
            "contains EVT3 data (nonzero size) and that width/height match "
            "the sensor (EVK4 is 1280x720)."
        )
    return output_video


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decode a recorded EVT3 .raw file into a video")
    p.add_argument("raw_file", help="Path to the recorded .raw file")
    p.add_argument("--output", default="evk4_capture.mp4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--slice-us", type=float, default=_DEFAULT_SLICE_US, help="Event accumulation window, microseconds")
    p.add_argument("--fps", type=float, default=100.0, help="Output video frame rate")
    p.add_argument("--reader-chunk-size", type=int, default=16384, help="Events per chunk from Evt3RawReader")
    p.add_argument("--dot-radius", type=int, default=0, help="Dilation radius in pixels (0 = raw pixel detail)")
    p.add_argument("--background", choices=["white", "black"], default="white")
    p.add_argument("--decay", type=float, default=0.0, help="Trail persistence, 0-0.95")
    args = p.parse_args()

    decode_to_video(
        args.raw_file,
        output_video=args.output,
        width=args.width,
        height=args.height,
        slice_us=args.slice_us,
        fps=args.fps,
        reader_chunk_size=args.reader_chunk_size,
        dot_radius_px=args.dot_radius,
        background=args.background,
        decay=args.decay,
    )