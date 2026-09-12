# Demo video script (~3 min)

Record your screen with the frontend running (`http://localhost:3000`, backend on
`:8000`). Read this roughly verbatim — timestamps are targets, not hard cuts.

---

**[0:00-0:20] The problem**

"This is hoop-scout — an agentic basketball scouting analyst. It watches a short
piece of game film and writes a scouting report where every claim cites a specific
timestamped possession with a confidence score. The core idea: I don't ask an LLM
to watch raw video. That doesn't work reliably. Instead there's a three-layer
pipeline."

**[0:20-0:50] Architecture (screen: README architecture diagram, or just talk over
the repo file tree)**

"First, a deterministic perception layer — no LLM at all. YOLO11 detects and tracks
players, a one-time manual court calibration maps pixels to real court coordinates,
and possession boundaries come from detecting the video's edit cuts — this footage
was cut to skip straight to the next possession after every made basket, so I built
a detector for that specific kind of edit, since normal scene-detection doesn't see
it. That produces a structured game_state.json.

Second, an agent layer. Claude gets tool-calling access to that structured data —
list possessions, get one possession's full detail, get a player's summary, compute
spacing, inspect a specific video frame when it's genuinely unsure, and record a
finding. It never sees raw video except when it deliberately asks to.

Third, a report layer that turns the agent's findings into a report where every
citation is clickable and seeks the video to that exact moment."

**[0:50-1:30] Live demo — the timeline and video (screen: frontend, localhost:3000)**

"Here's the actual app, running against real footage of a 4-on-4 pickup game. This
timeline is every detected possession — [click one] — clicking it seeks the video
right to that possession's start."

[click 2-3 different possession blocks, let each play a moment]

**[1:30-2:30] Live demo — running the agent**

"Now let's run the agent." [click Run scouting agent]

"This panel on the right is its live reasoning trail — every tool call, streamed as
it happens. You can see it surveying all the possessions first, then drilling into
specific ones, pulling player summaries, computing spacing — this is genuinely
multi-step, not one prompt dumping all the data and asking for a report."

[let it run 20-30s, narrate over it]

"And here's the report." [scroll to report] "Every finding has a confidence score
and clickable citations back to the exact possession." [click one citation] "That
seeks the video straight to it."

**[2:30-3:00] Eval + honesty about limitations**

"The eval harness — this is the part most submissions skip. It checks possession
boundary accuracy, and independently checks whether the agent's cited possessions
actually support the claims it made, using a separate LLM judge. I'll be upfront:
under time pressure I labeled ground truth myself using a known fact about the
footage rather than a full independent human review — that's documented honestly
in the README, along with a real limitation in cross-possession player identity
that I found and documented rather than hid. Knowing whether the agent is actually
right is the hard part of this project, and I wanted that to be visible, not
papered over."

**[3:00] Close**

"That's hoop-scout — deterministic perception, an agentic report layer with real
tool use, and an eval harness that checks its own honesty. Repo's linked below."

---

## Recording checklist
- [ ] Backend running: `uvicorn backend.main:app --reload`
- [ ] Frontend running: `npm run dev` (in `frontend/`)
- [ ] `.env` has your API key (agent run costs a small amount of real API usage)
- [ ] Do a dry run of clicking "Run scouting agent" once before recording so you know how long it takes
- [ ] Keep total under 4 minutes
