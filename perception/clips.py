"""Extracts per-possession video segments for the frontend clip player."""
from __future__ import annotations

import subprocess
from pathlib import Path


def extract_clip(
    video_path: str,
    start_ts: float,
    end_ts: float,
    out_path: str,
    pad_sec: float = 0.5,
) -> str:
    """Cut [start_ts - pad, end_ts + pad] out of video_path into out_path.

    Uses ffmpeg with stream copy where possible for speed; falls back to
    re-encoding is not attempted here — if this fails, it's a missing-ffmpeg
    problem to surface directly rather than silently degrade, since a broken
    clip breaks the citation-seeking feature in the frontend.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clip_start = max(0.0, start_ts - pad_sec)
    duration = (end_ts + pad_sec) - clip_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting clip {out_path}: {result.stderr}")
    return out_path


def extract_all_possession_clips(
    video_path: str,
    possessions: list,  # list[perception.schemas.Possession]
    out_dir: str,
) -> None:
    for possession in possessions:
        out_path = str(Path(out_dir) / f"{possession.possession_id}.mp4")
        extract_clip(video_path, possession.start_ts, possession.end_ts, out_path)
        possession.clip_path = out_path
