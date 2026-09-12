#!/usr/bin/env bash
# Runs both eval checks and writes combined results into eval/results.md.
set -euo pipefail

GAME_STATE="${1:-data/processed/game_state.json}"
GROUND_TRUTH="${2:-eval/ground_truth.json}"
FINDINGS="${3:-data/processed/findings.json}"

{
  echo "## Perception accuracy"
  echo
  python -m eval.score_perception --game-state "$GAME_STATE" --ground-truth "$GROUND_TRUTH"
  echo "## Finding citation support"
  echo
  python -m eval.score_findings --game-state "$GAME_STATE" --findings "$FINDINGS"
} > eval/results.md

cat eval/results.md
