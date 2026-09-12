"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import VideoPlayer, { VideoPlayerHandle } from "@/components/VideoPlayer";
import PossessionTimeline from "@/components/PossessionTimeline";
import AgentTrail from "@/components/AgentTrail";
import ReportView from "@/components/ReportView";
import { getGameState, videoUrl, startAgentRun, streamAgentRun } from "@/lib/api";
import type { GameState, ToolCallEntry, ReportDocument } from "@/lib/types";

export default function Home() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activePossessionId, setActivePossessionId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<ToolCallEntry[]>([]);
  const [report, setReport] = useState<ReportDocument | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const videoRef = useRef<VideoPlayerHandle>(null);

  useEffect(() => {
    getGameState()
      .then(setGameState)
      .catch((e) => setLoadError(String(e)));
  }, []);

  const handleSeek = useCallback((ts: number, possessionId: string) => {
    videoRef.current?.seekTo(ts);
    setActivePossessionId(possessionId);
  }, []);

  const handleRunAgent = useCallback(async () => {
    setRunning(true);
    setEvents([]);
    setReport(null);
    setRunError(null);
    try {
      const { run_id } = await startAgentRun();
      streamAgentRun(
        run_id,
        (entry) => setEvents((prev) => [...prev, entry]),
        (status, payload) => {
          setRunning(false);
          if (status === "done") setReport(payload as ReportDocument);
          else setRunError(typeof payload === "string" ? payload : JSON.stringify(payload));
        }
      );
    } catch (e) {
      setRunning(false);
      setRunError(String(e));
    }
  }, []);

  if (loadError) {
    return (
      <main className="mx-auto max-w-2xl p-8">
        <h1 className="font-display text-xl font-semibold text-court-orange-dark">
          Couldn&apos;t load game state
        </h1>
        <p className="mt-2 text-sm text-ink/70">{loadError}</p>
        <p className="mt-2 text-sm text-ink/50">
          Is the backend running? <code>uvicorn backend.main:app --reload</code>, and has the
          perception pipeline produced <code>data/processed/game_state.json</code>?
        </p>
      </main>
    );
  }

  if (!gameState) {
    return <main className="p-8 text-sm text-ink/50">Loading game state…</main>;
  }

  return (
    <>
      <header className="border-b-4 border-court-orange bg-hardwood px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-baseline gap-3">
          <h1 className="font-display text-2xl font-semibold uppercase tracking-wide text-court-ivory">
            hoop-scout
          </h1>
          <span className="text-xs uppercase tracking-widest text-court-orange">
            scouting report · {gameState.possessions.length} possessions
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-7xl p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <VideoPlayer ref={videoRef} src={videoUrl()} />
            <PossessionTimeline
              possessions={gameState.possessions}
              duration={gameState.video.duration_sec}
              onSeek={handleSeek}
              activeId={activePossessionId}
            />

            <div className="flex items-center gap-3">
              <button
                onClick={handleRunAgent}
                disabled={running}
                className="bg-court-orange px-5 py-2.5 text-sm font-semibold uppercase tracking-wide text-white transition-colors hover:bg-court-orange-dark disabled:cursor-not-allowed disabled:opacity-50"
              >
                {running ? "Agent running…" : "Run scouting agent"}
              </button>
              {runError && <span className="text-sm text-court-orange-dark">{runError}</span>}
            </div>

            {report && <ReportView report={report} onCiteClick={handleSeek} />}
          </div>

          <div className="h-[70vh] lg:h-auto">
            <AgentTrail events={events} running={running} />
          </div>
        </div>
      </main>
    </>
  );
}
