"use client";

import type { ToolCallEntry } from "@/lib/types";

export default function AgentTrail({
  events,
  running,
}: {
  events: ToolCallEntry[];
  running: boolean;
}) {
  return (
    <div className="flex h-full flex-col border border-hardwood-line/40 bg-hardwood">
      <div className="flex items-center gap-2 border-b border-hardwood-line/40 px-3 py-2 font-display text-xs font-semibold uppercase tracking-widest text-court-ivory/80">
        <span className={`inline-block h-1.5 w-1.5 rounded-full ${running ? "animate-pulse bg-whistle" : "bg-court-ivory/30"}`} />
        Reasoning trail
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-3 font-mono text-[11px] leading-relaxed">
        {events.length === 0 && !running && (
          <p className="text-court-ivory/40">Run the agent to see its tool calls here.</p>
        )}
        {events.map((e, i) => (
          <div key={i} className="border-l-2 border-court-orange/60 pl-2">
            <span className="font-semibold text-whistle">{e.tool_name}</span>
            {Object.keys(e.arguments).length > 0 && (
              <span className="text-court-ivory/40"> {JSON.stringify(e.arguments)}</span>
            )}
            <div className="text-court-ivory/70">→ {e.result_summary}</div>
          </div>
        ))}
        {running && <div className="animate-pulse text-court-ivory/40">…</div>}
      </div>
    </div>
  );
}
