# hoop-scout

An agentic basketball scouting analyst. It ingests a short piece of fixed-camera
game film and produces a written scouting report where every claim cites a
specific timestamped possession and carries an explicit confidence score.

## Architecture — this is the whole point of the project

Language models are not asked to "watch" video or reason over raw pixels
across time — that doesn't work reliably. Instead:

```
video ──▶ [1] Perception layer ──▶ game_state.json ──▶ [2] Agent layer ──▶ [3] Report layer
          (deterministic,          (structured game-      (Claude, tool-      (citable findings,
           no LLM)                  state database)         calling)          confidence-scored)
```

1. **Perception layer** (`perception/`, plain Python + YOLO11, no LLM). Detects
   and tracks players, maps pixel positions to real court coordinates via a
   one-time manual homography calibration, and segments the clip into discrete
   possessions using position, direction of play, and ball-handler proximity
   heuristics. Ball detection is attempted but the pipeline degrades
   gracefully when it's unreliable — see `perception/possession.py`.
2. **Agent layer** (`agent/`). A Claude tool-calling loop reasons over the
   structured `game_state.json` — it never sees raw video except when it
   deliberately calls `inspect_frame` to resolve a specific ambiguity. It
   surveys possessions, forms hypotheses, drills into specific possessions,
   and commits findings via `record_finding`, each with citations and a
   confidence score. Every tool call is logged — that trail is rendered live
   in the frontend.
3. **Report layer** (`agent/report.py`). Resolves each finding's cited
   possession IDs into timestamps and clip paths so the frontend can make
   every citation clickable and seek the video to that exact moment.

## A note on edited source footage

Some source clips are edited to skip dead time between possessions (cutting
straight to the next possession after a made basket) rather than being one
continuous unedited take. Standard scene-change detection can't see these
cuts — the camera and background are identical on both sides, so overall
frame content barely changes. `perception/cuts.py` detects them via
frame-to-frame pixel-difference outliers instead (real human motion at 60fps
changes a tiny fraction of the frame; a hard cut changes far more, since 6-8
people's positions and poses change all at once).

Each cut segment is tracked independently, which means the same real player
gets a new track ID in every possession — there's no continuous motion to
carry an ID across a cut. `perception/reid.py` reconnects these into stable
global player IDs after the fact via appearance clustering (torso color),
so roster mapping and cross-possession player stats still work. This is a
deliberately simple heuristic, not real re-identification — flagged as a
known limitation in `game_state.json`'s `meta.notes`, not a silent
assumption. Pick `segmentation_mode` (`"cuts"` or `"heuristic"`) per video in
`calibration_ui`.

## Repo structure

```
perception/       detection, calibration, possession segmentation, cut detection, re-ID, clip extraction
calibration_ui/   standalone browser tool: click 4 court landmarks once, map track IDs to names
agent/            tool definitions, the Claude tool-calling loop, report assembly
backend/          FastAPI: serves game_state.json + clips, runs the agent loop, streams its trail
frontend/         Next.js: video player, possession timeline, live tool-call trail, report view
data/             raw video, processed game_state.json + clips, calibration output (gitignored)
eval/             ground-truth labels + scoring scripts (see Evaluation below)
scripts/          run_pipeline.sh, run_eval.sh
```

## Scope

One fixed-camera clip, 2-5 minutes, half court. Not a general video-analysis
system. No auth, no multi-user, no database beyond files.

## Setup

```bash
pip install -r requirements.txt
cd frontend && npm install
```

Requires `ffmpeg` on PATH (clip extraction) and an `ANTHROPIC_API_KEY` env var
(agent loop + eval citation-support judge, both use Claude Haiku).

## Running the pipeline

1. Record or obtain a fixed-camera half-court clip and place it under `data/raw/`.
2. Export a still frame (`ffmpeg -i data/raw/clip.mp4 -frames:v 1 frame.jpg`) and
   open `calibration_ui/index.html` in a browser to click the 4 court landmarks
   and export `calibration.json`. Run the pipeline once without a roster map to
   see the assigned track IDs, then fill in `roster.json` and re-run if you
   want names instead of raw track IDs in the report.
3. Run the perception pipeline:
   ```bash
   ./scripts/run_pipeline.sh data/raw/clip.mp4 calibration.json roster.json
   ```
4. Start the backend and frontend, then trigger an agent run from the UI:
   ```bash
   uvicorn backend.main:app --reload
   cd frontend && npm run dev
   ```

## Evaluation

Required, not optional — a plausible-sounding report is not the same as a
correct one. `eval/` contains:

- `ground_truth.template.json` — copy to `ground_truth.json` and hand-label
  ~20 possessions (boundaries, ball handler, outcome) from the real clip.
- `score_perception.py` — scores the pipeline's possession boundaries
  (temporal IoU), ball-handler attribution, and outcome accuracy against
  ground truth.
- `score_findings.py` — an independent LLM judge checks, for every finding
  the agent recorded, whether the possession data it cited actually supports
  the claim made about it.

Run both and regenerate the table below with:

```bash
./scripts/run_eval.sh
```

### Results

Run against the real clip (`data/raw/freeflowbasketball_oxnard.mp4`, 27 possessions,
`segmentation_mode=cuts`). **Methodology note:** possessions p001-p015 were human-verified
— the project owner watched each clip in `data/processed/clips/` directly and confirmed
or corrected the outcome. That review caught two real errors in the initial Claude-drafted
labels (p007 and p013 were both labeled `made_2` from the "editor cuts after every make"
assumption, but are actually a possession that ends on a normal pass and a missed shot,
respectively) — a concrete example of that assumption not holding universally, not just
a hypothetical caveat. Possessions p016-p027 remain Claude-drafted from that same
assumption and have not been independently reviewed; `ball_handler_track_id` was left
unlabeled throughout, since verifying it needs either ID-overlay video or manual
position-matching, neither of which was feasible in the time available. See
`eval/ground_truth.json`'s `_labeling_method` field for the full breakdown.

| Metric | Value |
|---|---|
| Ground-truth possessions | 27 |
| Matched (IoU >= 0.5) | 27 |
| Mean boundary IoU | 1.00 |
| Ball-handler attribution accuracy | N/A (no independent labels — see note above) |
| Outcome accuracy | 7% (expected: the perception layer never guesses an outcome by design, so it only scores a "hit" on the 2 possessions ground truth also marks "unknown"; every real outcome label scores as a mismatch) |
| Findings with supported citations | 2 / 5 |
| Citation support rate | 40% |

**On the 40%:** this check is doing its job — it caught a real design gap, not just
sloppy agent claims. Findings about spatial structure (spacing, convex hull area)
cite a `possession_id`, but the judge is only shown that possession's raw stored
data, not the `compute_spacing()` tool's on-the-fly output that actually backed the
claim — that metric isn't persisted back onto the possession object. So a
well-supported claim can still fail this check because the *evidence* it cites isn't
visible to the judge, not because the claim was fabricated. Worth fixing next: pass
compute_spacing results into the judge's context, or record which tool call backed
each finding, not just which possession.
