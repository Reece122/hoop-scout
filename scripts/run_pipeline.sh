#!/usr/bin/env bash
# Runs the offline perception pipeline: video -> game_state.json + clips.
# Calibration points come from calibration.json, produced once via calibration_ui.
set -euo pipefail

VIDEO_PATH="${1:?Usage: run_pipeline.sh <video_path> <calibration.json> <roster.json>}"
CALIBRATION_JSON="${2:?Usage: run_pipeline.sh <video_path> <calibration.json> <roster.json>}"
ROSTER_JSON="${3:?Usage: run_pipeline.sh <video_path> <calibration.json> <roster.json>}"

python -c "
import json
from perception.pipeline import run_pipeline

calibration = json.load(open('$CALIBRATION_JSON'))
roster = json.load(open('$ROSTER_JSON'))

run_pipeline(
    video_path='$VIDEO_PATH',
    image_points=calibration['image_points'],
    court_points_ft=calibration['court_points_ft'],
    landmarks_used=calibration['landmarks_used'],
    basket_x_ft=calibration['basket_x_ft'],
    track_id_to_name=roster.get('track_id_to_name', {}),
    clips_out_dir='data/processed/clips',
    game_state_out_path='data/processed/game_state.json',
    segmentation_mode=calibration.get('segmentation_mode', 'heuristic'),
)
print('game_state.json written to data/processed/game_state.json')
"
