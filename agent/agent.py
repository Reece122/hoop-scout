"""Claude tool-calling loop for the scouting agent.

This is the only place an LLM reasons over the game. It never sees raw video
except via the inspect_frame tool result, and every step is logged through
AgentTools.logger so the caller (backend/routers/agent.py) can stream the
reasoning trail live.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from anthropic import Anthropic

from agent.prompts import SYSTEM_PROMPT
from agent.tools import AgentTools
from perception.schemas import GameState

AGENT_MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 25

TOOL_DEFINITIONS = [
    {
        "name": "list_possessions",
        "description": "List possession summaries, optionally filtered. Use this first to survey the game.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": ["offense", "defense"]},
                "player_track_id": {"type": "integer"},
                "outcome": {"type": "string", "enum": ["made_2", "made_3", "missed", "turnover", "foul", "unknown"]},
                "min_duration_sec": {"type": "number"},
                "court_side": {"type": "string"},
            },
        },
    },
    {
        "name": "get_possession",
        "description": "Get full detail (position time series, derived features, clip path) for one possession.",
        "input_schema": {
            "type": "object",
            "properties": {"possession_id": {"type": "string"}},
            "required": ["possession_id"],
        },
    },
    {
        "name": "get_player_summary",
        "description": "Aggregate stats for one player (by track_id) across all possessions.",
        "input_schema": {
            "type": "object",
            "properties": {"track_id": {"type": "integer"}},
            "required": ["track_id"],
        },
    },
    {
        "name": "inspect_frame",
        "description": "Render the video frame nearest a timestamp and view it directly, for when structured data is ambiguous. Use sparingly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "possession_id": {"type": "string"},
            },
            "required": ["timestamp"],
        },
    },
    {
        "name": "compute_spacing",
        "description": "Compute derived spatial metrics (avg pairwise distance, convex hull area, nearest defender distance) for a possession.",
        "input_schema": {
            "type": "object",
            "properties": {
                "possession_id": {"type": "string"},
                "timestamp": {"type": "number"},
            },
            "required": ["possession_id"],
        },
    },
    {
        "name": "record_finding",
        "description": "Commit a finding to the report. Every claim you make in the final report must have a corresponding record_finding call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "possession_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "category": {"type": "string"},
            },
            "required": ["claim", "possession_ids", "confidence"],
        },
    },
]

TaskEventCallback = Optional[Callable[[dict], None]]


def run_agent_loop(
    game_state: GameState,
    video_path: str,
    on_event: TaskEventCallback = None,
    max_turns: int = MAX_TURNS,
) -> dict:
    """Runs the full survey -> hypothesize -> drill-in -> write-findings loop.

    `on_event` is called with each tool-call log entry as it happens, so the
    backend can stream it to the frontend live (this is the "reasoning trail"
    the demo relies on). Returns {report_text, findings, tool_log}.
    """
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = AgentTools(game_state=game_state, video_path=video_path)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Analyze this basketball clip and produce a scouting report. "
                "Start by surveying the possessions."
            ),
        }
    ]

    for _turn in range(max_turns):
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            report_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            # Models sometimes stop early to ask "want me to continue?" instead of finishing
            # the loop, especially after just the survey step — seen in practice with zero
            # findings recorded. This is a fully autonomous run with no one to answer that
            # question, so nudge it to keep going rather than accepting an empty result.
            if not tools.findings and _turn < max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue autonomously — there is no user to respond to questions. "
                        "You have not recorded any findings yet. Keep drilling into possessions "
                        "and players, then call record_finding for each conclusion and write the "
                        "final report, all without stopping to ask."
                    ),
                })
                continue
            return {
                "report_text": report_text,
                "findings": [dict(f) for f in tools.findings],
                "tool_log": tools.logger.as_list(),
            }

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _dispatch_tool(tools, block.name, block.input)
            if on_event is not None:
                on_event(tools.logger.entries[-1].__dict__)
            tool_results.append(_to_tool_result(block.id, result))

        messages.append({"role": "user", "content": tool_results})

    # Ran out of turns without the model stopping voluntarily — still return
    # whatever findings were recorded rather than raising, since partial
    # results with real citations are useful and this is a bounded loop.
    return {
        "report_text": "(Agent reached max turns before finalizing a written report.)",
        "findings": [dict(f) for f in tools.findings],
        "tool_log": tools.logger.as_list(),
    }


def _dispatch_tool(tools: AgentTools, name: str, arguments: dict):
    try:
        method = getattr(tools, name)
        return method(**arguments)
    except Exception as exc:  # tool errors go back to the model as data, not a crash
        return {"error": str(exc)}


def _to_tool_result(tool_use_id: str, result) -> dict:
    if isinstance(result, dict) and result.get("type") == "image":
        content = [result]
    else:
        content = [{"type": "text", "text": json.dumps(result, default=str)}]
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
