"""Lightweight cross-possession player re-identification.

Hard cuts between possessions (perception/cuts.py) mean each possession's
tracking starts fresh — the same real player gets a new track_id in every
possession. This clusters tracks across possessions by appearance (torso
color) into a small number of persistent global player IDs, so roster
mapping and cross-possession stats (get_player_summary) mean something.

This is a deliberately simple heuristic, not real re-identification: with a
handful of players in a pickup game, average torso hue is usually enough to
tell people apart, but two players in near-identical shirts can get merged.
That's a known, documented limitation, not a silently-assumed guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIMILARITY_THRESHOLD = 0.3  # histogram correlation; below this, treat as a different player.
# Tuned down from an initial 0.7 after a real run produced 60 global IDs for an 8-10 person
# game — real-world lighting/pose variance between possessions correlates far more weakly
# than expected. Still a heuristic; retune against eval/ground_truth.json labels once available.
SAMPLES_PER_TRACK = 5


@dataclass
class TrackAppearance:
    possession_id: str
    track_id: int
    color_histogram: np.ndarray  # normalized HSV hue histogram of torso crops


def extract_track_appearance(
    clip_path: str,
    possession_id: str,
    track_id: int,
    boxes: list,  # list[perception.detect.TrackedBox] for this track only
) -> TrackAppearance:
    """Samples a few bounding-box crops for this track and averages their hue histogram."""
    cap = cv2.VideoCapture(clip_path)
    histograms = []

    step = max(1, len(boxes) // SAMPLES_PER_TRACK)
    for box in boxes[::step][:SAMPLES_PER_TRACK]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, box.frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(int, box.bbox_px)
        # torso only: middle band vertically, to avoid head/shoes/floor bleeding into the histogram
        torso_y1 = y1 + int((y2 - y1) * 0.25)
        torso_y2 = y1 + int((y2 - y1) * 0.65)
        crop = frame[max(0, torso_y1):torso_y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    cap.release()
    avg_hist = np.mean(histograms, axis=0) if histograms else np.zeros(32, dtype=np.float32)
    return TrackAppearance(possession_id=possession_id, track_id=track_id, color_histogram=avg_hist)


def cluster_global_identities(
    appearances: list[TrackAppearance], similarity_threshold: float = SIMILARITY_THRESHOLD
) -> dict[tuple[str, int], int]:
    """Greedy clustering: assign each (possession_id, track_id) to a global player ID.

    A cluster can never contain two tracks from the same possession — those are
    necessarily different real people, no matter how similar their appearance.
    """
    clusters: list[list[TrackAppearance]] = []
    assignment: dict[tuple[str, int], int] = {}

    for appearance in appearances:
        best_cluster_idx, best_score = None, -1.0
        for idx, cluster in enumerate(clusters):
            if any(a.possession_id == appearance.possession_id for a in cluster):
                continue  # can't be the same player as someone already in this possession
            cluster_mean = np.mean([a.color_histogram for a in cluster], axis=0)
            score = cv2.compareHist(
                appearance.color_histogram.astype(np.float32),
                cluster_mean.astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
            if score > best_score:
                best_score, best_cluster_idx = score, idx

        if best_cluster_idx is not None and best_score >= similarity_threshold:
            clusters[best_cluster_idx].append(appearance)
            assignment[(appearance.possession_id, appearance.track_id)] = best_cluster_idx
        else:
            clusters.append([appearance])
            assignment[(appearance.possession_id, appearance.track_id)] = len(clusters) - 1

    return assignment
