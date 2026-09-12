"""Checks how often a finding's cited possessions actually support its claim.

This is the check most submissions skip: it's not enough for the agent to
produce a plausible-sounding report, each claim has to be traceable back to
data that actually supports it. For every recorded finding, we hand an LLM
judge (a fresh model call, no access to the agent's own reasoning) the raw
possession data for every possession_id cited and the claim text, and ask a
strict yes/no: does this data support this claim?
"""
from __future__ import annotations

import argparse
import json
import os

from anthropic import Anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker for a basketball scouting report. \
You will be given a claim and the raw structured data for the possession(s) cited as support. \
Decide ONLY whether the cited data actually supports the claim as written — do not use outside \
basketball knowledge, do not give credit for plausible-sounding claims that the data doesn't \
directly back up. Respond with strict JSON: {"supported": true|false, "reason": "<one sentence>"}."""


def load_possession(game_state: dict, possession_id: str) -> dict | None:
    for p in game_state["possessions"]:
        if p["possession_id"] == possession_id:
            return p
    return None


def judge_finding(client: Anthropic, finding: dict, cited_possessions: list[dict]) -> dict:
    user_content = (
        f"CLAIM: {finding['claim']}\n\n"
        f"CITED POSSESSION DATA:\n{json.dumps(cited_possessions, indent=2)}"
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"supported": False, "reason": f"judge returned unparseable output: {text[:200]}"}


def score_findings(game_state: dict, findings: list[dict]) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
    for finding in findings:
        cited = [load_possession(game_state, pid) for pid in finding["possession_ids"]]
        cited = [c for c in cited if c is not None]
        if not cited:
            results.append({**finding, "supported": False, "reason": "cited possession_id not found in game_state"})
            continue
        verdict = judge_finding(client, finding, cited)
        results.append({**finding, **verdict})

    supported_count = sum(1 for r in results if r["supported"])
    return {
        "total_findings": len(findings),
        "supported_findings": supported_count,
        "citation_support_rate": supported_count / len(findings) if findings else 0.0,
        "details": results,
    }


def format_results_markdown(scored: dict) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Total findings | {scored['total_findings']} |",
        f"| Findings with supported citations | {scored['supported_findings']} |",
        f"| Citation support rate | {scored['citation_support_rate']:.0%} |",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-state", required=True)
    parser.add_argument("--findings", required=True, help="JSON file: list of {claim, possession_ids, confidence}")
    args = parser.parse_args()

    game_state = json.loads(open(args.game_state).read())
    findings = json.loads(open(args.findings).read())

    scored = score_findings(game_state, findings)
    print(format_results_markdown(scored))


if __name__ == "__main__":
    main()
