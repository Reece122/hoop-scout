"""Builds a demo video from real project assets (data, not screenshots).

Screen-recording automation proved unreliable in this environment, so this
composes the demo from what's actually verifiable: real eval numbers, real
agent findings, real gameplay footage. Each "card" is a rendered PNG shown
for a few seconds; one segment cuts to real footage from data/processed/clips.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
INK = (33, 26, 20)
IVORY = (247, 243, 236)
HARDWOOD = (42, 32, 25)
HARDWOOD_LINE = (74, 58, 45)
ORANGE = (232, 89, 12)
WHISTLE = (242, 183, 5)

FONT_DIR = "C:/Windows/Fonts"
OUT_DIR = Path("data/processed/demo_cards")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONT_DIR}/{'arialbd' if bold else 'arial'}.ttf", size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=f) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def base_card(bg=HARDWOOD) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H - 14, W, H], fill=ORANGE)
    return img, draw


def save(img: Image.Image, name: str) -> str:
    path = OUT_DIR / f"{name}.png"
    img.save(path)
    return str(path)


def title_card() -> str:
    img, d = base_card()
    d.text((120, 380), "HOOP-SCOUT", font=font(140, bold=True), fill=IVORY)
    d.text((124, 540), "an agentic basketball scouting analyst", font=font(46), fill=WHISTLE)
    d.text((124, 620), "deterministic perception  ->  tool-calling agent  ->  citable report",
           font=font(32), fill=(200, 190, 178))
    return save(img, "01_title")


def what_it_does_card() -> str:
    img, d = base_card()
    d.text((120, 90), "WHAT IT DOES", font=font(64, bold=True), fill=ORANGE)
    lines = [
        "Give it a short piece of fixed-camera basketball game film.",
        "",
        "It detects and tracks every player, maps their positions onto real",
        "court coordinates, and automatically splits the video into",
        "individual possessions.",
        "",
        "Then an AI agent studies that structured data — not the raw video —",
        "and writes a scouting report.",
        "",
        "Every claim in the report cites the exact possession and timestamp",
        "it's based on, so you can click a claim and jump straight to the",
        "moment in the video that backs it up.",
    ]
    y = 230
    for line in lines:
        if line:
            d.text((120, y), line, font=font(38), fill=IVORY)
        y += 56
    return save(img, "01b_what_it_does")


def architecture_card() -> str:
    img, d = base_card()
    d.text((120, 90), "ARCHITECTURE", font=font(64, bold=True), fill=ORANGE)
    lines = [
        ("1. PERCEPTION", "YOLO11 detects & tracks players. A one-time manual court", IVORY),
        ("", "calibration maps pixels to real court feet. Possession boundaries", IVORY),
        ("", "come from detecting this footage's edit cuts. No LLM involved.", IVORY),
        ("", "", IVORY),
        ("2. AGENT", "Claude gets tool-calling access to the structured game state:", IVORY),
        ("", "list_possessions, get_possession, get_player_summary,", IVORY),
        ("", "compute_spacing, inspect_frame, record_finding. It never sees", IVORY),
        ("", "raw video except when it deliberately calls inspect_frame.", IVORY),
        ("", "", IVORY),
        ("3. REPORT", "Every finding resolves to a timestamp + clip, with a", IVORY),
        ("", "confidence score, so every claim in the report is clickable.", IVORY),
    ]
    y = 210
    for label, text, color in lines:
        if label:
            d.text((120, y), label, font=font(38, bold=True), fill=WHISTLE)
        if text:
            d.text((460, y), text, font=font(34), fill=color)
        y += 58
    return save(img, "02_architecture")


def footage_caption_card() -> str:
    img, d = base_card(bg=INK)
    d.text((120, 460), "REAL GAME FOOTAGE", font=font(72, bold=True), fill=IVORY)
    d.text((124, 560), "4-on-4 half-court pickup basketball, fixed tripod angle",
           font=font(38), fill=WHISTLE)
    d.text((124, 620), "27 possessions detected via edit-cut segmentation",
           font=font(32), fill=(200, 190, 178))
    return save(img, "03_footage_caption")


def findings_card() -> str:
    findings = json.loads(Path("data/processed/findings.json").read_text())
    img, d = base_card()
    d.text((120, 70), "AGENT REASONING TRAIL -> REPORT", font=font(56, bold=True), fill=ORANGE)
    trail = [
        "> list_possessions()",
        "> get_player_summary(track_id=5)",
        "> compute_spacing(possession_id=\"p017\")",
        "> inspect_frame(possession_id=\"p009\", ts=142.3)",
        "> record_finding(...)",
    ]
    mono_path = f"{FONT_DIR}/consola.ttf"
    mono = ImageFont.truetype(mono_path, 30) if Path(mono_path).exists() else font(30)
    y = 190
    for line in trail:
        d.text((120, y), line, font=mono, fill=(120, 200, 150))
        y += 42

    y += 30
    d.line([(120, y), (W - 120, y)], fill=HARDWOOD_LINE, width=2)
    y += 40

    top = findings[0]
    d.text((120, y), f"FINDING  ·  confidence {top['confidence']:.0%}", font=font(32, bold=True), fill=WHISTLE)
    y += 50
    for line in wrap(d, top["claim"], font(36), W - 240):
        d.text((120, y), line, font=font(36), fill=IVORY)
        y += 46
    y += 10
    d.text((120, y), f"cites: {', '.join(top['possession_ids'])}", font=font(28), fill=ORANGE)
    return save(img, "04_findings")


def eval_card() -> str:
    img, d = base_card(bg=INK)
    d.text((120, 70), "EVAL HARNESS — REAL NUMBERS", font=font(56, bold=True), fill=ORANGE)
    rows = [
        ("Ground-truth possessions", "27"),
        ("Matched (IoU >= 0.5)", "27"),
        ("Mean boundary IoU", "1.00"),
        ("Ball-handler accuracy", "N/A (no independent labels)"),
        ("Outcome accuracy", "7% (expected — layer never guesses)"),
        ("Findings with supported citations", "2 / 5  (40%)"),
    ]
    y = 220
    for label, val in rows:
        d.text((120, y), label, font=font(38), fill=IVORY)
        d.text((1140, y), val, font=font(38, bold=True), fill=WHISTLE)
        y += 80
    return save(img, "05_eval")


def limitations_card() -> str:
    img, d = base_card()
    d.text((120, 90), "HONEST LIMITATIONS", font=font(60, bold=True), fill=ORANGE)
    paras = [
        "Ground truth started as Claude-drafted labels, then 15 of 27 possessions "
        "were human-verified against the real clips — catching two real errors the "
        "assumption-based labels got wrong (a non-shot possession, a missed shot).",
        "Cross-possession player re-identification uses color-histogram clustering, "
        "a deliberately simple heuristic — documented as a known limitation, not a "
        "silent assumption.",
        "The citation-support eval caught a real design gap: some findings cite a "
        "possession but the judge can't see the compute_spacing() output that "
        "actually backed the claim. That's the honest 40%, not sloppiness.",
    ]
    y = 230
    for p in paras:
        for line in wrap(d, p, font(36), W - 240):
            d.text((120, y), line, font=font(36), fill=IVORY)
            y += 46
        y += 40
    return save(img, "06_limitations")


def closing_card() -> str:
    img, d = base_card()
    d.text((120, 420), "hoop-scout", font=font(110, bold=True), fill=IVORY)
    d.text((124, 560), "github.com/Reece122/hoop-scout", font=font(44), fill=WHISTLE)
    return save(img, "07_closing")


if __name__ == "__main__":
    cards = [
        (title_card(), 4),
        (what_it_does_card(), 9),
        (architecture_card(), 9),
        (footage_caption_card(), 3),
        (findings_card(), 9),
        (eval_card(), 9),
        (limitations_card(), 9),
        (closing_card(), 4),
    ]
    print(json.dumps(cards))
