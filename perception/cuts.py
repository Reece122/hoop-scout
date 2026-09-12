"""Detects hard cuts in a fixed-camera recording that has been edited to skip
dead time between possessions (e.g. cutting straight to the next possession
after a made basket).

Standard scene-change detection misses these: the camera and background are
identical on both sides of the cut, so overall frame content barely changes.
An optical-flow discontinuity approach was tried first and failed outright
(0 cuts detected on real footage with confirmed cuts, in 10+ minutes) —
Farneback's smoothness prior apparently "explains away" the discontinuity
with a locally-smooth flow field instead of spiking on it.

What actually works: plain frame-to-frame mean pixel difference. At 60fps,
real human motion changes only a tiny fraction of the frame between
consecutive frames; a hard cut changes far more at once, since 6-8 people's
positions and poses all change simultaneously. Empirically (see the diagnostic
this was tuned against — a 352s clip with confirmed cuts), the diff signal is
bimodal: periodic small blips (~1.6-2.2, roughly every 5.5s, most likely
camera auto-exposure adjustment) versus real cuts (3.0+, irregularly spaced).
CUT_DIFF_THRESHOLD is tuned to that gap — it is a heuristic tuned per-footage,
not a universal constant, and may need adjusting for different source video.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CUT_DIFF_THRESHOLD = 3.0  # mean grayscale abs-diff at DOWNSCALE_SIZE; empirically tuned, see module docstring
DOWNSCALE_SIZE = (320, 180)
MERGE_WINDOW_SEC = 1.0  # candidate diffs within this window of each other are one cut, keep the strongest
MIN_SEGMENT_DURATION_SEC = 1.0  # shorter than this is almost certainly detector noise, not a real possession


@dataclass
class Segment:
    start_ts: float
    end_ts: float
    start_frame: int
    end_frame: int


def detect_cuts(video_path: str, threshold: float = CUT_DIFF_THRESHOLD) -> list[float]:
    """Returns timestamps (seconds) where a hard cut is detected."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    candidates: list[tuple[float, float]] = []
    prev_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, DOWNSCALE_SIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff > threshold:
                candidates.append((frame_idx / fps, diff))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return _merge_candidates(candidates)


def _merge_candidates(candidates: list[tuple[float, float]]) -> list[float]:
    merged: list[tuple[float, float]] = []
    for ts, diff in candidates:
        if merged and ts - merged[-1][0] < MERGE_WINDOW_SEC:
            if diff > merged[-1][1]:
                merged[-1] = (ts, diff)
        else:
            merged.append((ts, diff))
    return [ts for ts, _ in merged]


def build_segments(video_path: str, cut_timestamps: list[float]) -> list[Segment]:
    """Turns cut timestamps into contiguous [start, end) segments covering the whole video.

    Each segment corresponds to one possession that ended in a cut (per this
    project's source footage, cuts happen after made baskets). A segment can
    occasionally span more than one real possession if an earlier possession
    ended in a miss/turnover that didn't trigger a cut — full sub-segmentation
    within a segment isn't attempted, and is a known, documented limitation.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    cap.release()

    boundaries = [0.0, *sorted(cut_timestamps), duration]
    segments = []
    for start_ts, end_ts in zip(boundaries[:-1], boundaries[1:]):
        if end_ts - start_ts < MIN_SEGMENT_DURATION_SEC:
            continue
        segments.append(
            Segment(
                start_ts=start_ts,
                end_ts=end_ts,
                start_frame=int(start_ts * fps),
                end_frame=int(end_ts * fps),
            )
        )
    return segments
