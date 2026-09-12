"""Orchestrates the full perception pipeline: video -> game_state.json.

Supports two segmentation modes:
  - "heuristic" (original design): continuous unedited footage, possession
    boundaries inferred from ball/position heuristics (perception/possession.py
    segment_possessions()). Tracking runs once over the whole video, so track
    IDs persist naturally across possessions.
  - "cuts": the source video has been edited to cut straight from one
    possession to the next (e.g. skipping to the next possession after every
    made basket). Cuts are detected via pixel-difference discontinuity
    (perception/cuts.py). Each segment is extracted as its own clip and
    tracked independently — which means track IDs do NOT persist across
    possessions on their own, so a lightweight appearance-based
    re-identification pass (perception/reid.py) reconnects them into stable
    global player IDs afterward.

Everything here is deterministic and offline — the agent never runs any of
this itself, it only reads the game_state.json this produces.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from perception.ball import run_ball_detection
from perception.calibrate import build_calibration, pixels_to_court_batch
from perception.clips import extract_all_possession_clips, extract_clip
from perception.cuts import build_segments, detect_cuts
from perception.possession import (
    TrackTimeSeries,
    build_possession_from_cut_segment,
    segment_possessions,
)
from perception.reid import cluster_global_identities, extract_track_appearance
from perception.schemas import (
    Ball,
    Calibration,
    GameState,
    GameStateMeta,
    Roster,
    VideoMeta,
)

PIPELINE_VERSION = "0.3.0"

# A homography estimated from 4 points clustered in one small region of the
# frame (e.g. one key/lane) extrapolates poorly far outside that region.
# Detections whose projected court position lands way outside a plausible
# court footprint are almost always background people (bystanders, other
# gym-goers), not participants — this filters them out before they dilute
# the ball-handler heuristic and the re-ID clustering.
COURT_BOUNDS_Y_FT = (-5.0, 55.0)   # court width ~50ft, with margin
COURT_BOUNDS_X_FT = (-15.0, 110.0)  # generous depth margin; true court length is an estimate


def run_pipeline(
    video_path: str,
    image_points: list[tuple[float, float]],
    court_points_ft: list[tuple[float, float]],
    landmarks_used: str,
    basket_x_ft: float,
    track_id_to_name: dict[str, str],
    clips_out_dir: str,
    game_state_out_path: str,
    detector: str = "yolo11n",
    segmentation_mode: str = "heuristic",
    reid_cache_path: str | None = None,
) -> GameState:
    calibration: Calibration = build_calibration(image_points, court_points_ft, landmarks_used)

    if segmentation_mode == "cuts":
        possessions, all_track_ids = _run_cut_based(
            video_path, calibration, clips_out_dir, detector, reid_cache_path
        )
        notes = [
            "segmentation_mode=cuts: possession boundaries came from detected edit cuts, not heuristics.",
            "Player identity is reconstructed via appearance-based re-ID across cuts (perception/reid.py). "
            "On the real footage this was built against, it under-merges badly: ~27 possessions produced "
            "50+ global IDs against ~8 real players. Root cause appears to be within-possession track "
            "fragmentation (ByteTrack losing a player through contact/occlusion and assigning a new local "
            "ID), not just clothing-color ambiguity — since a cluster can never contain two IDs from the "
            "same possession, a fragmented player is structurally forced into extra global IDs. Treat "
            "cross-possession player identity and get_player_summary() aggregates as unreliable, not just "
            "approximate, until this is revisited with a stronger re-ID signal.",
            "offense/defense role split is not determined in this mode; offense_track_ids holds all "
            "participants, defense_track_ids is empty.",
            f"Detections projecting outside court bounds x{COURT_BOUNDS_X_FT} y{COURT_BOUNDS_Y_FT} ft "
            "(almost always background people, not players) are dropped before possession-building.",
        ]
    elif segmentation_mode == "heuristic":
        possessions, all_track_ids = _run_heuristic_based(video_path, calibration, basket_x_ft, detector)
        extract_all_possession_clips(video_path, possessions, clips_out_dir)
        notes = []
    else:
        raise ValueError(f"Unknown segmentation_mode: {segmentation_mode}")

    unmapped = sorted(all_track_ids - {int(k) for k in track_id_to_name.keys()})
    video_meta = _probe_video_meta(video_path)

    game_state = GameState(
        video=video_meta,
        calibration=calibration,
        roster=Roster(track_id_to_name=track_id_to_name, unmapped_track_ids=unmapped),
        ball=Ball(track_available=any(p.ball_handler_confidence > 0 for p in possessions), detections=[]),
        possessions=possessions,
        meta=GameStateMeta(
            generated_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version=PIPELINE_VERSION,
            detector=detector,
            notes=notes,
        ),
    )

    Path(game_state_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(game_state_out_path).write_text(game_state.model_dump_json(indent=2))
    return game_state


def _in_bounds(court_xy: tuple[float, float]) -> bool:
    x, y = court_xy
    return COURT_BOUNDS_X_FT[0] <= x <= COURT_BOUNDS_X_FT[1] and COURT_BOUNDS_Y_FT[0] <= y <= COURT_BOUNDS_Y_FT[1]


def _run_heuristic_based(video_path, calibration, basket_x_ft, detector):
    from perception.detect import run_detection_and_tracking

    homography = np.array(calibration.homography)

    detection_run = run_detection_and_tracking(video_path, model_name=f"{detector}.pt")
    ball_run = run_ball_detection(video_path, frame_count=detection_run.frame_count, model_name=f"{detector}.pt")

    tracks_by_id: dict[int, list[tuple[float, tuple[float, float]]]] = {}
    for box in detection_run.boxes:
        x1, y1, x2, y2 = box.bbox_px
        foot_px = np.array([(x1 + x2) / 2, y2])
        court_xy = tuple(pixels_to_court_batch(homography, foot_px.reshape(1, 2))[0])
        if not _in_bounds(court_xy):
            continue
        tracks_by_id.setdefault(box.track_id, []).append((box.ts, court_xy))

    track_series = [TrackTimeSeries(track_id=tid, samples=s) for tid, s in tracks_by_id.items()]

    ball_court_positions: dict[float, tuple[float, float]] = {}
    for det in ball_run.detections:
        x1, y1, x2, y2 = det.bbox_px
        center_px = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        court_xy = tuple(pixels_to_court_batch(homography, center_px.reshape(1, 2))[0])
        ball_court_positions[det.ts] = court_xy

    possessions = segment_possessions(
        tracks=track_series,
        ball=ball_run,
        ball_court_positions=ball_court_positions,
        basket_x_ft=basket_x_ft,
        fps=detection_run.fps,
    )
    return possessions, set(tracks_by_id.keys())


def _run_cut_based(video_path, calibration, clips_out_dir, detector, reid_cache_path):
    from perception.detect import run_detection_and_tracking

    homography = np.array(calibration.homography)

    cut_timestamps = detect_cuts(video_path)
    segments = build_segments(video_path, cut_timestamps)

    possessions = []
    all_appearances = []

    for idx, seg in enumerate(segments, start=1):
        possession_id = f"p{idx:03d}"
        clip_path = str(Path(clips_out_dir) / f"{possession_id}.mp4")
        extract_clip(video_path, seg.start_ts, seg.end_ts, clip_path, pad_sec=0.0)

        detection_run = run_detection_and_tracking(clip_path, model_name=f"{detector}.pt")
        ball_run = run_ball_detection(clip_path, frame_count=detection_run.frame_count, model_name=f"{detector}.pt")

        boxes_by_track: dict[int, list] = {}
        for box in detection_run.boxes:
            boxes_by_track.setdefault(box.track_id, []).append(box)

        tracks: list[TrackTimeSeries] = []
        for track_id, boxes in boxes_by_track.items():
            samples = []
            for box in boxes:
                x1, y1, x2, y2 = box.bbox_px
                foot_px = np.array([(x1 + x2) / 2, y2])
                court_xy = tuple(pixels_to_court_batch(homography, foot_px.reshape(1, 2))[0])
                if not _in_bounds(court_xy):
                    continue
                samples.append((box.ts + seg.start_ts, court_xy))  # offset: clip-local -> global timeline

            if not samples:
                continue  # this track never had an in-bounds detection — treat as background, not a player
            tracks.append(TrackTimeSeries(track_id=track_id, samples=samples))
            all_appearances.append(extract_track_appearance(clip_path, possession_id, track_id, boxes))

        ball_court_positions: dict[float, tuple[float, float]] = {}
        for det in ball_run.detections:
            x1, y1, x2, y2 = det.bbox_px
            center_px = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            court_xy = tuple(pixels_to_court_batch(homography, center_px.reshape(1, 2))[0])
            if _in_bounds(court_xy):
                ball_court_positions[det.ts + seg.start_ts] = court_xy

        possession = build_possession_from_cut_segment(
            possession_id=possession_id,
            tracks=tracks,
            ball_court_positions=ball_court_positions,
            ball_track_available=ball_run.track_available,
            start_ts=seg.start_ts,
            end_ts=seg.end_ts,
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
        )
        possession.clip_path = clip_path
        possessions.append(possession)

    if reid_cache_path:
        Path(reid_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(reid_cache_path, "wb") as f:
            pickle.dump({"possessions": possessions, "appearances": all_appearances}, f)

    return _apply_reid(possessions, all_appearances)


def _apply_reid(possessions, all_appearances, similarity_threshold: float | None = None):
    kwargs = {} if similarity_threshold is None else {"similarity_threshold": similarity_threshold}
    global_id_map = cluster_global_identities(all_appearances, **kwargs)

    for possession in possessions:
        possession.offense_track_ids = sorted(
            {global_id_map[(possession.possession_id, tid)] for tid in possession.offense_track_ids}
        )
        if possession.ball_handler_track_id is not None:
            possession.ball_handler_track_id = global_id_map.get(
                (possession.possession_id, possession.ball_handler_track_id)
            )
        for sample in possession.position_time_series:
            sample.track_positions = {
                str(global_id_map[(possession.possession_id, int(local_tid))]): pos
                for local_tid, pos in sample.track_positions.items()
            }

    return possessions, set(global_id_map.values())


def _probe_video_meta(video_path: str) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(
        path=video_path, fps=fps, width=width, height=height,
        duration_sec=frame_count / fps if fps else 0.0,
    )
