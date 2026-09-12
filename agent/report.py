"""Assembles the final report document served to the frontend.

Takes the agent loop's raw output (report_text + findings) and resolves each
finding's possession_ids into concrete timestamps, so the frontend can make
every citation clickable and seek the video player to that exact moment.
"""
from __future__ import annotations

from perception.schemas import GameState


def build_report_document(game_state: GameState, agent_result: dict) -> dict:
    possessions_by_id = {p.possession_id: p for p in game_state.possessions}

    resolved_findings = []
    for finding in agent_result["findings"]:
        citations = []
        for pid in finding["possession_ids"]:
            possession = possessions_by_id.get(pid)
            if possession is None:
                continue
            citations.append(
                {
                    "possession_id": pid,
                    "start_ts": possession.start_ts,
                    "end_ts": possession.end_ts,
                    "clip_path": possession.clip_path,
                }
            )
        resolved_findings.append(
            {
                "finding_id": finding.get("finding_id"),
                "claim": finding["claim"],
                "confidence": finding["confidence"],
                "category": finding.get("category"),
                "citations": citations,
            }
        )

    return {
        "report_text": agent_result["report_text"],
        "findings": resolved_findings,
        "tool_log": agent_result["tool_log"],
        "video_path": game_state.video.path,
    }
