"""Player detection + tracking via YOLO11 (ultralytics).

Deterministic, no LLM. Runs once per video (offline precompute), producing a
per-frame list of tracked player boxes. Ball detection lives in ball.py
separately since it is far less reliable and must not block this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person", used by all YOLO11 checkpoints


@dataclass
class TrackedBox:
    ts: float
    frame_idx: int
    track_id: int
    bbox_px: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


@dataclass
class DetectionRun:
    fps: float
    width: int
    height: int
    frame_count: int
    boxes: list[TrackedBox] = field(default_factory=list)


def run_detection_and_tracking(
    video_path: str,
    model_name: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    conf_threshold: float = 0.35,
) -> DetectionRun:
    """Run YOLO11 detection + tracking over the whole clip.

    Uses ultralytics' built-in `.track()` streaming API, which persists
    track IDs across frames via ByteTrack. Only the "person" class is kept —
    ball detection is handled separately in ball.py because it needs a much
    lower confidence bar and different post-processing to degrade gracefully.
    """
    model = YOLO(model_name)
    results_stream = model.track(
        source=video_path,
        classes=[PERSON_CLASS_ID],
        conf=conf_threshold,
        tracker=tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    run: DetectionRun | None = None
    for frame_idx, result in enumerate(results_stream):
        if run is None:
            h, w = result.orig_shape
            fps = float(result.speed.get("fps", 0)) or _probe_fps(video_path)
            run = DetectionRun(fps=fps, width=w, height=h, frame_count=0)

        run.frame_count += 1
        ts = frame_idx / run.fps if run.fps else 0.0

        if result.boxes is None or result.boxes.id is None:
            continue  # no confirmed tracks this frame — not an error, just skip

        for box, track_id, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.id.tolist(),
            result.boxes.conf.tolist(),
        ):
            run.boxes.append(
                TrackedBox(
                    ts=ts,
                    frame_idx=frame_idx,
                    track_id=int(track_id),
                    bbox_px=tuple(box),
                    confidence=float(conf),
                )
            )

    if run is None:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return run


def _probe_fps(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)
