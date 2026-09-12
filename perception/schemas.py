"""Pydantic models mirroring game_state.json.

This is the single schema contract shared by the perception layer (which
produces game_state.json), the agent tools (which read it), and the backend
routers (which serve it). Nothing here calls an LLM or touches raw video.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    MADE_2 = "made_2"
    MADE_3 = "made_3"
    MISSED = "missed"
    TURNOVER = "turnover"
    FOUL = "foul"
    UNKNOWN = "unknown"


class VideoMeta(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    duration_sec: float


class Calibration(BaseModel):
    image_points: list[tuple[float, float]] = Field(..., min_length=4, max_length=4)
    court_points_ft: list[tuple[float, float]] = Field(..., min_length=4, max_length=4)
    homography: list[list[float]]  # 3x3
    landmarks_used: str


class Roster(BaseModel):
    track_id_to_name: dict[str, str] = Field(default_factory=dict)
    unmapped_track_ids: list[int] = Field(default_factory=list)


class BallDetection(BaseModel):
    ts: float
    bbox_px: tuple[float, float, float, float]
    court_xy_ft: Optional[tuple[float, float]] = None
    confidence: float


class Ball(BaseModel):
    track_available: bool
    detections: list[BallDetection] = Field(default_factory=list)


class PositionSample(BaseModel):
    ts: float
    track_positions: dict[str, tuple[float, float]]  # track_id (str) -> (x_ft, y_ft)
    ball_position: Optional[tuple[float, float]] = None


class DerivedFeatures(BaseModel):
    avg_spacing_ft: Optional[float] = None
    duration_sec: float
    court_side: Optional[str] = None
    notes: str = ""


class Possession(BaseModel):
    possession_id: str
    start_ts: float
    end_ts: float
    start_frame: int
    end_frame: int
    offense_track_ids: list[int]
    defense_track_ids: list[int]
    ball_handler_track_id: Optional[int] = None
    ball_handler_confidence: float = 0.0
    outcome: Outcome = Outcome.UNKNOWN
    outcome_confidence: float = 0.0
    position_time_series: list[PositionSample] = Field(default_factory=list)
    derived_features: DerivedFeatures
    clip_path: Optional[str] = None


class PossessionSummary(BaseModel):
    """Lightweight projection of Possession, returned by list_possessions().

    Deliberately excludes position_time_series so the agent's context doesn't
    balloon when surveying many possessions at once.
    """
    possession_id: str
    start_ts: float
    end_ts: float
    offense_track_ids: list[int]
    ball_handler_track_id: Optional[int] = None
    outcome: Outcome
    outcome_confidence: float


class GameStateMeta(BaseModel):
    generated_at: str
    pipeline_version: str
    detector: str
    notes: list[str] = Field(default_factory=list)


class GameState(BaseModel):
    video: VideoMeta
    calibration: Calibration
    roster: Roster
    ball: Ball
    possessions: list[Possession]
    meta: GameStateMeta
