"""Best-effort ball detection.

Ball detection at typical fixed-camera resolution is unreliable: the ball is
small, fast, and frequently occluded by players. This module tries, but the
rest of the pipeline (especially possession.py) is designed to degrade
gracefully when results here are sparse or empty — ball detection must never
become a blocker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ultralytics import YOLO

BALL_CLASS_ID = 32  # COCO "sports ball" — coarse; a fine-tuned checkpoint would help
MIN_USABLE_DETECTION_RATE = 0.15  # below this fraction of frames, treat ball as unavailable


@dataclass
class BallDetection:
    ts: float
    frame_idx: int
    bbox_px: tuple[float, float, float, float]
    confidence: float


@dataclass
class BallDetectionRun:
    track_available: bool
    detections: list[BallDetection] = field(default_factory=list)


def run_ball_detection(
    video_path: str,
    frame_count: int,
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.15,
) -> BallDetectionRun:
    """Best-effort ball detection; returns track_available=False if too sparse to trust."""
    model = YOLO(model_name)
    results_stream = model.predict(
        source=video_path,
        classes=[BALL_CLASS_ID],
        conf=conf_threshold,
        stream=True,
        verbose=False,
    )

    detections: list[BallDetection] = []
    fps = None
    for frame_idx, result in enumerate(results_stream):
        if fps is None:
            fps = _probe_fps(video_path)
        ts = frame_idx / fps if fps else 0.0

        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Keep only the highest-confidence ball candidate per frame — there
        # should only ever be one ball on court.
        best_idx = int(result.boxes.conf.argmax())
        detections.append(
            BallDetection(
                ts=ts,
                frame_idx=frame_idx,
                bbox_px=tuple(result.boxes.xyxy[best_idx].tolist()),
                confidence=float(result.boxes.conf[best_idx]),
            )
        )

    detection_rate = len(detections) / frame_count if frame_count else 0.0
    track_available = detection_rate >= MIN_USABLE_DETECTION_RATE
    return BallDetectionRun(track_available=track_available, detections=detections)


def _probe_fps(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)
