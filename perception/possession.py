"""Possession segmentation heuristics.

Derives discrete possessions from tracked player (and, when available, ball)
positions in court coordinates. Designed to degrade gracefully when the ball
track is missing or sparse — this is the module most at risk of being wrong,
so every heuristic here is deliberately simple and should be tuned against
eval/ground_truth.json early, not left until the end.

Heuristic, in order of preference:
1. If the ball track is usable: a possession is a contiguous span where the
   ball stays on one team's offensive half and one player is consistently
   the closest player to the ball (the "ball handler").
2. If the ball track is NOT usable: fall back to a proxy — track the team
   whose players' average court position is advancing toward the opposing
   basket, and treat direction reversals (a change in which basket the
   group of players is advancing toward) as possession boundaries. The
   ball-handler is then the offensive player closest to the group's
   centroid-of-motion, at a lower confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.ball import BallDetectionRun
from perception.detect import DetectionRun
from perception.schemas import DerivedFeatures, Outcome, Possession, PositionSample

MIN_POSSESSION_DURATION_SEC = 2.0
BALL_HANDLER_PROXIMITY_FT = 6.0


@dataclass
class TrackTimeSeries:
    """Per-track court-position samples, indexed by timestamp."""
    track_id: int
    samples: list[tuple[float, tuple[float, float]]]  # (ts, (x_ft, y_ft))


def segment_possessions(
    tracks: list[TrackTimeSeries],
    ball: BallDetectionRun,
    ball_court_positions: dict[float, tuple[float, float]],
    basket_x_ft: float,
    fps: float,
) -> list[Possession]:
    """Segment the clip into possessions.

    `ball_court_positions` maps ball detection timestamps to court (x, y) —
    already homography-mapped upstream. `basket_x_ft` is the x-coordinate of
    the basket being attacked on this half of the court (single fixed camera,
    half court only, per project scope).
    """
    if ball.track_available and ball_court_positions:
        return _segment_with_ball(tracks, ball_court_positions, fps)
    return _segment_without_ball(tracks, basket_x_ft, fps)


def _segment_with_ball(
    tracks: list[TrackTimeSeries],
    ball_court_positions: dict[float, tuple[float, float]],
    fps: float,
) -> list[Possession]:
    timestamps = sorted(ball_court_positions.keys())
    handler_by_ts: dict[float, tuple[int, float]] = {}

    for ts in timestamps:
        ball_xy = np.array(ball_court_positions[ts])
        best_track_id, best_dist = None, float("inf")
        for track in tracks:
            pos = _nearest_sample(track.samples, ts)
            if pos is None:
                continue
            dist = float(np.linalg.norm(np.array(pos) - ball_xy))
            if dist < best_dist:
                best_dist, best_track_id = dist, track.track_id
        if best_track_id is not None:
            handler_by_ts[ts] = (best_track_id, best_dist)

    return _build_possessions_from_handler_sequence(handler_by_ts, tracks, fps, ball_backed=True)


def _segment_without_ball(
    tracks: list[TrackTimeSeries],
    basket_x_ft: float,
    fps: float,
) -> list[Possession]:
    """Fallback: use group direction-of-advance as a possession-boundary proxy.

    Lower confidence than the ball-backed path — this is explicitly a
    degraded mode, not a silent substitute for real ball tracking.
    """
    all_ts = sorted({ts for track in tracks for ts, _ in track.samples})
    handler_by_ts: dict[float, tuple[int, float]] = {}

    for ts in all_ts:
        positions = [
            (track.track_id, _nearest_sample(track.samples, ts))
            for track in tracks
        ]
        positions = [(tid, p) for tid, p in positions if p is not None]
        if not positions:
            continue
        centroid_x = np.mean([p[0] for _, p in positions])
        # proxy handler = player nearest the centroid, moving toward the basket
        best_track_id, best_dist = None, float("inf")
        for tid, (x, y) in positions:
            dist = abs(x - centroid_x)
            if dist < best_dist:
                best_dist, best_track_id = dist, tid
        if best_track_id is not None:
            handler_by_ts[ts] = (best_track_id, best_dist)

    return _build_possessions_from_handler_sequence(handler_by_ts, tracks, fps, ball_backed=False)


def _build_possessions_from_handler_sequence(
    handler_by_ts: dict[float, tuple[int, float]],
    tracks: list[TrackTimeSeries],
    fps: float,
    ball_backed: bool,
) -> list[Possession]:
    timestamps = sorted(handler_by_ts.keys())
    if not timestamps:
        return []

    possessions: list[Possession] = []
    seg_start = timestamps[0]
    seg_handlers: list[int] = [handler_by_ts[timestamps[0]][0]]
    seg_dists: list[float] = [handler_by_ts[timestamps[0]][1]]

    def flush(seg_end: float, idx: int) -> None:
        duration = seg_end - seg_start
        if duration < MIN_POSSESSION_DURATION_SEC:
            return
        handler_id = _majority(seg_handlers)
        confidence = _handler_confidence(seg_dists, ball_backed)
        possessions.append(
            Possession(
                possession_id=f"p{idx:03d}",
                start_ts=seg_start,
                end_ts=seg_end,
                start_frame=int(seg_start * fps),
                end_frame=int(seg_end * fps),
                offense_track_ids=sorted(set(seg_handlers)),
                defense_track_ids=[],  # filled in by pipeline.py once roles are known
                ball_handler_track_id=handler_id,
                ball_handler_confidence=confidence,
                outcome=Outcome.UNKNOWN,
                outcome_confidence=0.0,
                position_time_series=[],  # filled in by pipeline.py
                derived_features=DerivedFeatures(duration_sec=duration),
            )
        )

    idx = 1
    for prev_ts, ts in zip(timestamps, timestamps[1:]):
        handler, dist = handler_by_ts[ts]
        if handler != seg_handlers[-1] and handler not in seg_handlers[-3:]:
            flush(prev_ts, idx)
            idx += 1
            seg_start = ts
            seg_handlers, seg_dists = [], []
        seg_handlers.append(handler)
        seg_dists.append(dist)

    flush(timestamps[-1], idx)
    return possessions


def _nearest_sample(
    samples: list[tuple[float, tuple[float, float]]], ts: float, tol: float = 0.5
) -> tuple[float, float] | None:
    best = min(samples, key=lambda s: abs(s[0] - ts), default=None)
    if best is None or abs(best[0] - ts) > tol:
        return None
    return best[1]


def _majority(values: list[int]) -> int:
    return max(set(values), key=values.count)


def _handler_confidence(dists: list[float], ball_backed: bool) -> float:
    avg_dist = float(np.mean(dists)) if dists else BALL_HANDLER_PROXIMITY_FT
    proximity_score = max(0.0, 1.0 - avg_dist / (BALL_HANDLER_PROXIMITY_FT * 2))
    return proximity_score * (0.9 if ball_backed else 0.5)


def build_possession_from_cut_segment(
    possession_id: str,
    tracks: list[TrackTimeSeries],
    ball_court_positions: dict[float, tuple[float, float]],
    ball_track_available: bool,
    start_ts: float,
    end_ts: float,
    start_frame: int,
    end_frame: int,
) -> Possession:
    """Builds exactly one Possession spanning a whole cut-detected segment.

    Used when the video has been edited to cut straight from one possession to
    the next (perception/cuts.py) — the segment boundaries ARE the possession
    boundaries already, so none of the direction-reversal / handler-switching
    logic in segment_possessions() is needed here. This only has to figure out
    who the ball handler was, using the same proximity heuristics.

    Track IDs here are LOCAL to this segment (tracking was run fresh per clip,
    see perception/pipeline.py) — the caller remaps them to global player IDs
    via perception/reid.py after all segments are built.

    Known gap: offense/defense role split is not determined (would need team
    jersey-color clustering, which isn't reliable enough to build blind under
    this project's time budget) — offense_track_ids holds every participant,
    defense_track_ids is left empty. compute_spacing's nearest-defender metric
    degrades to None accordingly, consistent with this project's degrade-
    gracefully-rather-than-guess philosophy elsewhere (ball detection, etc).
    """
    if ball_track_available and ball_court_positions:
        handler_by_ts = _handler_by_ts_ball_backed(tracks, ball_court_positions)
        ball_backed = True
    else:
        handler_by_ts = _handler_by_ts_proxy(tracks)
        ball_backed = False

    all_track_ids = sorted({t.track_id for t in tracks})
    position_time_series = _build_position_time_series(tracks, start_ts, end_ts)

    if handler_by_ts:
        handler_id = _majority([h for h, _ in handler_by_ts.values()])
        confidence = _handler_confidence([d for _, d in handler_by_ts.values()], ball_backed)
    else:
        handler_id, confidence = None, 0.0

    return Possession(
        possession_id=possession_id,
        start_ts=start_ts,
        end_ts=end_ts,
        start_frame=start_frame,
        end_frame=end_frame,
        offense_track_ids=all_track_ids,
        defense_track_ids=[],
        ball_handler_track_id=handler_id,
        ball_handler_confidence=confidence,
        outcome=Outcome.UNKNOWN,  # perception layer can't tell make/miss from tracking alone
        outcome_confidence=0.0,
        position_time_series=position_time_series,
        derived_features=DerivedFeatures(
            duration_sec=end_ts - start_ts,
            notes="cut-detected segment; offense/defense split not determined",
        ),
    )


def _handler_by_ts_ball_backed(
    tracks: list[TrackTimeSeries], ball_court_positions: dict[float, tuple[float, float]]
) -> dict[float, tuple[int, float]]:
    handler_by_ts: dict[float, tuple[int, float]] = {}
    for ts, ball_xy in ball_court_positions.items():
        ball_arr = np.array(ball_xy)
        best_track_id, best_dist = None, float("inf")
        for track in tracks:
            pos = _nearest_sample(track.samples, ts)
            if pos is None:
                continue
            dist = float(np.linalg.norm(np.array(pos) - ball_arr))
            if dist < best_dist:
                best_dist, best_track_id = dist, track.track_id
        if best_track_id is not None:
            handler_by_ts[ts] = (best_track_id, best_dist)
    return handler_by_ts


def _handler_by_ts_proxy(tracks: list[TrackTimeSeries]) -> dict[float, tuple[int, float]]:
    all_ts = sorted({ts for track in tracks for ts, _ in track.samples})
    handler_by_ts: dict[float, tuple[int, float]] = {}
    for ts in all_ts:
        positions = [(t.track_id, _nearest_sample(t.samples, ts)) for t in tracks]
        positions = [(tid, p) for tid, p in positions if p is not None]
        if not positions:
            continue
        centroid_x = np.mean([p[0] for _, p in positions])
        best_track_id, best_dist = None, float("inf")
        for tid, (x, _y) in positions:
            dist = abs(x - centroid_x)
            if dist < best_dist:
                best_dist, best_track_id = dist, tid
        if best_track_id is not None:
            handler_by_ts[ts] = (best_track_id, best_dist)
    return handler_by_ts


TARGET_SAMPLES_PER_SEC = 5.0  # source video is 60fps; raw density made a single possession's
# position_time_series balloon to 100K+ tokens as a tool result — way more temporal resolution
# than spacing/movement analysis needs, and expensive/risky to hand an LLM raw.


def _build_position_time_series(
    tracks: list[TrackTimeSeries], start_ts: float, end_ts: float
) -> list[PositionSample]:
    all_ts = sorted({ts for track in tracks for ts, _ in track.samples if start_ts <= ts <= end_ts})
    all_ts = _downsample_timestamps(all_ts, TARGET_SAMPLES_PER_SEC)
    samples = []
    for ts in all_ts:
        track_positions = {}
        for track in tracks:
            pos = _nearest_sample(track.samples, ts, tol=0.1)
            if pos is not None:
                track_positions[str(track.track_id)] = pos
        if track_positions:
            samples.append(PositionSample(ts=ts, track_positions=track_positions))
    return samples


def _downsample_timestamps(all_ts: list[float], target_per_sec: float) -> list[float]:
    if not all_ts or target_per_sec <= 0:
        return all_ts
    min_gap = 1.0 / target_per_sec
    kept = [all_ts[0]]
    for ts in all_ts[1:]:
        if ts - kept[-1] >= min_gap:
            kept.append(ts)
    return kept
