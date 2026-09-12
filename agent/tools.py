"""Tool implementations the agent calls against game_state.json.

Every function here is a plain Python function operating on structured data
already produced by the perception layer (perception/pipeline.py). None of
them touch raw video except inspect_frame, which is the one deliberate
exception: it renders a single frame and hands it back to the agent as an
image so Claude's own vision can look at it when the structured data is
ambiguous.

Each call is logged (see ToolCallLogger) so the reasoning trail is visible
in the frontend — that trail is a large part of the demo.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from perception.schemas import GameState, Outcome, PossessionSummary


@dataclass
class ToolCallLogEntry:
    tool_name: str
    arguments: dict
    result_summary: str
    ts: float = field(default_factory=time.time)


class ToolCallLogger:
    def __init__(self) -> None:
        self.entries: list[ToolCallLogEntry] = []

    def log(self, tool_name: str, arguments: dict, result_summary: str) -> None:
        self.entries.append(ToolCallLogEntry(tool_name, arguments, result_summary))

    def as_list(self) -> list[dict]:
        return [
            {"tool_name": e.tool_name, "arguments": e.arguments, "result_summary": e.result_summary, "ts": e.ts}
            for e in self.entries
        ]


class Finding(dict):
    """A committed finding: {claim, possession_ids, confidence, category, finding_id}."""


class AgentTools:
    """Bound to one loaded GameState + one session's findings + tool-call log."""

    def __init__(self, game_state: GameState, video_path: str, logger: Optional[ToolCallLogger] = None):
        self.game_state = game_state
        self.video_path = video_path
        self.logger = logger or ToolCallLogger()
        self.findings: list[Finding] = []
        self._next_finding_id = 1

    # ---- tool: list_possessions ----------------------------------------
    def list_possessions(
        self,
        team: Optional[str] = None,
        player_track_id: Optional[int] = None,
        outcome: Optional[str] = None,
        min_duration_sec: Optional[float] = None,
        court_side: Optional[str] = None,
    ) -> list[dict]:
        results = []
        for p in self.game_state.possessions:
            if player_track_id is not None and player_track_id not in (
                p.offense_track_ids + p.defense_track_ids
            ):
                continue
            if outcome is not None and p.outcome.value != outcome:
                continue
            if min_duration_sec is not None and p.derived_features.duration_sec < min_duration_sec:
                continue
            if court_side is not None and p.derived_features.court_side != court_side:
                continue
            results.append(
                PossessionSummary(
                    possession_id=p.possession_id,
                    start_ts=p.start_ts,
                    end_ts=p.end_ts,
                    offense_track_ids=p.offense_track_ids,
                    ball_handler_track_id=p.ball_handler_track_id,
                    outcome=p.outcome,
                    outcome_confidence=p.outcome_confidence,
                ).model_dump()
            )
        call_args = {
            "team": team,
            "player_track_id": player_track_id,
            "outcome": outcome,
            "min_duration_sec": min_duration_sec,
            "court_side": court_side,
        }
        self.logger.log("list_possessions", call_args, f"{len(results)} possessions")
        return results

    # ---- tool: get_possession -------------------------------------------
    def get_possession(self, possession_id: str) -> dict:
        for p in self.game_state.possessions:
            if p.possession_id == possession_id:
                result = p.model_dump()
                self.logger.log("get_possession", {"possession_id": possession_id}, f"found {possession_id}")
                return result
        self.logger.log("get_possession", {"possession_id": possession_id}, "NOT FOUND")
        raise ValueError(f"No possession with id {possession_id}")

    # ---- tool: get_player_summary ----------------------------------------
    def get_player_summary(self, track_id: int) -> dict:
        involved = [
            p for p in self.game_state.possessions
            if track_id in (p.offense_track_ids + p.defense_track_ids)
        ]
        as_handler = [p for p in involved if p.ball_handler_track_id == track_id]
        outcomes: dict[str, int] = {}
        for p in as_handler:
            outcomes[p.outcome.value] = outcomes.get(p.outcome.value, 0) + 1
        spacings = [p.derived_features.avg_spacing_ft for p in involved if p.derived_features.avg_spacing_ft]

        summary = {
            "track_id": track_id,
            "name": self.game_state.roster.track_id_to_name.get(str(track_id)),
            "possessions_involved": [p.possession_id for p in involved],
            "touches": len(involved),
            "times_as_ball_handler": len(as_handler),
            "avg_spacing_when_on_court": sum(spacings) / len(spacings) if spacings else None,
            "outcomes_breakdown": outcomes,
        }
        self.logger.log("get_player_summary", {"track_id": track_id}, f"{len(involved)} possessions involved")
        return summary

    # ---- tool: inspect_frame ----------------------------------------------
    def inspect_frame(self, timestamp: float, possession_id: Optional[str] = None) -> dict:
        """Renders the frame nearest `timestamp` and returns it as a base64 image
        block for the agent's own (Claude) vision to look at directly."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.game_state.video.fps
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.logger.log("inspect_frame", {"timestamp": timestamp}, "FAILED to read frame")
            raise ValueError(f"Could not read frame at ts={timestamp}")

        ok, buf = cv2.imencode(".jpg", frame)
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        self.logger.log(
            "inspect_frame", {"timestamp": timestamp, "possession_id": possession_id},
            f"rendered frame at ts={timestamp:.2f}",
        )
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        }

    # ---- tool: compute_spacing --------------------------------------------
    def compute_spacing(self, possession_id: str, timestamp: Optional[float] = None) -> dict:
        possession = next(
            (p for p in self.game_state.possessions if p.possession_id == possession_id), None
        )
        if possession is None:
            raise ValueError(f"No possession with id {possession_id}")

        samples = possession.position_time_series
        if timestamp is not None:
            samples = [s for s in samples if abs(s.ts - timestamp) < 0.5] or samples[:1]

        pairwise_dists, hull_areas, nearest_defender = [], [], {}
        for sample in samples:
            offense_positions = {
                tid: pos for tid, pos in sample.track_positions.items()
                if int(tid) in possession.offense_track_ids
            }
            defense_positions = {
                tid: pos for tid, pos in sample.track_positions.items()
                if int(tid) in possession.defense_track_ids
            }
            pts = list(offense_positions.values())
            if len(pts) >= 2:
                arr = np.array(pts)
                dists = [
                    float(np.linalg.norm(arr[i] - arr[j]))
                    for i in range(len(arr)) for j in range(i + 1, len(arr))
                ]
                pairwise_dists.extend(dists)
            if len(pts) >= 3:
                hull_areas.append(_convex_hull_area(np.array(pts)))
            for off_id, off_pos in offense_positions.items():
                nearest = min(
                    (float(np.linalg.norm(np.array(off_pos) - np.array(d_pos))) for d_pos in defense_positions.values()),
                    default=None,
                )
                if nearest is not None:
                    nearest_defender.setdefault(off_id, []).append(nearest)

        result = {
            "possession_id": possession_id,
            "avg_pairwise_distance_ft": sum(pairwise_dists) / len(pairwise_dists) if pairwise_dists else None,
            "convex_hull_area_sqft": sum(hull_areas) / len(hull_areas) if hull_areas else None,
            "per_player_nearest_defender_ft": {
                tid: sum(v) / len(v) for tid, v in nearest_defender.items()
            },
        }
        self.logger.log("compute_spacing", {"possession_id": possession_id, "timestamp": timestamp}, "computed")
        return result

    # ---- tool: record_finding ----------------------------------------------
    def record_finding(
        self, claim: str, possession_ids: list[str], confidence: float, category: Optional[str] = None
    ) -> dict:
        finding_id = f"f{self._next_finding_id:03d}"
        self._next_finding_id += 1
        finding = Finding(
            finding_id=finding_id,
            claim=claim,
            possession_ids=possession_ids,
            confidence=confidence,
            category=category,
        )
        self.findings.append(finding)
        self.logger.log(
            "record_finding",
            {"claim": claim, "possession_ids": possession_ids, "confidence": confidence},
            f"recorded {finding_id}",
        )
        return {"finding_id": finding_id}


def _convex_hull_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))
