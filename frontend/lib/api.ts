import type { GameState, AgentRun, ToolCallEntry } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function getGameState(): Promise<GameState> {
  const res = await fetch(`${API_BASE}/api/game-state`);
  if (!res.ok) throw new Error(`game-state fetch failed: ${res.status}`);
  return res.json();
}

export function videoUrl(): string {
  return `${API_BASE}/api/video`;
}

export function clipUrl(possessionId: string): string {
  return `${API_BASE}/api/clips/${possessionId}`;
}

export async function startAgentRun(): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/api/agent/runs`, { method: "POST" });
  if (!res.ok) throw new Error(`start agent run failed: ${res.status}`);
  return res.json();
}

export async function getAgentRun(runId: string): Promise<AgentRun> {
  const res = await fetch(`${API_BASE}/api/agent/runs/${runId}`);
  if (!res.ok) throw new Error(`get agent run failed: ${res.status}`);
  return res.json();
}

/** Streams the tool-call trail live via SSE; calls onEvent per tool call and
 * onDone once the run finishes (with the final result or error). */
export function streamAgentRun(
  runId: string,
  onEvent: (entry: ToolCallEntry) => void,
  onDone: (status: "done" | "error", payload: unknown) => void
): () => void {
  const source = new EventSource(`${API_BASE}/api/agent/runs/${runId}/stream`);

  source.addEventListener("tool_call", (e) => {
    onEvent(JSON.parse((e as MessageEvent).data));
  });
  source.addEventListener("done", (e) => {
    onDone("done", JSON.parse((e as MessageEvent).data));
    source.close();
  });
  source.addEventListener("error", (e) => {
    const msgEvent = e as MessageEvent;
    onDone("error", msgEvent.data ? JSON.parse(msgEvent.data) : "stream error");
    source.close();
  });

  return () => source.close();
}
