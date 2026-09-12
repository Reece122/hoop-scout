"""Scores the perception pipeline's output against hand-labeled ground truth.

Three metrics, matching the spec:
  1. Possession boundary accuracy — temporal IoU between predicted and
     ground-truth possession spans, matched greedily by overlap.
  2. Ball-handler attribution accuracy — % of matched possessions where the
     predicted ball_handler_track_id equals ground truth.
  3. Outcome accuracy — % of matched possessions where predicted outcome
     equals ground truth exactly.

Usage:
    python -m eval.score_perception --game-state data/processed/game_state.json \
        --ground-truth eval/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

IOU_MATCH_THRESHOLD = 0.5


@dataclass
class ScoreResult:
    num_ground_truth: int
    num_matched: int
    boundary_iou_mean: float
    ball_handler_accuracy: float | None  # None means no ground-truth labels were available to check against
    outcome_accuracy: float


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def match_possessions(predicted: list[dict], ground_truth: list[dict]) -> list[tuple[dict, dict, float]]:
    """Greedy best-IoU matching, one predicted possession per ground-truth possession."""
    matches = []
    used_pred_ids = set()
    for gt in ground_truth:
        best_pred, best_iou = None, 0.0
        for pred in predicted:
            if pred["possession_id"] in used_pred_ids:
                continue
            iou = temporal_iou(pred["start_ts"], pred["end_ts"], gt["start_ts"], gt["end_ts"])
            if iou > best_iou:
                best_iou, best_pred = iou, pred
        if best_pred is not None and best_iou >= IOU_MATCH_THRESHOLD:
            used_pred_ids.add(best_pred["possession_id"])
            matches.append((best_pred, gt, best_iou))
    return matches


def score(game_state: dict, ground_truth: dict) -> ScoreResult:
    predicted = game_state["possessions"]
    gt_possessions = ground_truth["possessions"]
    matches = match_possessions(predicted, gt_possessions)

    if not matches:
        return ScoreResult(len(gt_possessions), 0, 0.0, None, 0.0)

    ious = [iou for _, _, iou in matches]
    handler_labeled = [
        (pred, gt) for pred, gt, _ in matches if gt.get("ball_handler_track_id") is not None
    ]
    handler_correct = sum(
        1 for pred, gt in handler_labeled if pred.get("ball_handler_track_id") == gt["ball_handler_track_id"]
    )
    outcome_correct = sum(
        1 for pred, gt, _ in matches if pred.get("outcome") == gt.get("outcome")
    )

    return ScoreResult(
        num_ground_truth=len(gt_possessions),
        num_matched=len(matches),
        boundary_iou_mean=sum(ious) / len(ious),
        ball_handler_accuracy=(handler_correct / len(handler_labeled)) if handler_labeled else None,
        outcome_accuracy=outcome_correct / len(matches),
    )


def format_results_markdown(result: ScoreResult) -> str:
    handler_str = f"{result.ball_handler_accuracy:.0%}" if result.ball_handler_accuracy is not None else "N/A (no labels)"
    return (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Ground-truth possessions | {result.num_ground_truth} |\n"
        f"| Matched (IoU >= {IOU_MATCH_THRESHOLD}) | {result.num_matched} |\n"
        f"| Mean boundary IoU | {result.boundary_iou_mean:.2f} |\n"
        f"| Ball-handler attribution accuracy | {handler_str} |\n"
        f"| Outcome accuracy | {result.outcome_accuracy:.0%} |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-state", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    game_state = json.loads(open(args.game_state).read())
    ground_truth = json.loads(open(args.ground_truth).read())

    result = score(game_state, ground_truth)
    print(format_results_markdown(result))


if __name__ == "__main__":
    main()
