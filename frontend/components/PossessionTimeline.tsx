"use client";

import type { Possession } from "@/lib/types";

const OUTCOME_COLOR: Record<string, string> = {
  made_2: "bg-green-500",
  made_3: "bg-emerald-400",
  missed: "bg-red-400",
  turnover: "bg-orange-400",
  foul: "bg-yellow-400",
  unknown: "bg-gray-400",
};

export default function PossessionTimeline({
  possessions,
  duration,
  onSeek,
  activeId,
}: {
  possessions: Possession[];
  duration: number;
  onSeek: (ts: number, possessionId: string) => void;
  activeId: string | null;
}) {
  return (
    <div className="w-full">
      <div className="relative h-8 w-full border border-hardwood-line/30 bg-hardwood">
        {possessions.map((p) => {
          const leftPct = (p.start_ts / duration) * 100;
          const widthPct = Math.max(((p.end_ts - p.start_ts) / duration) * 100, 0.3);
          return (
            <button
              key={p.possession_id}
              title={`${p.possession_id}: ${p.start_ts.toFixed(1)}s-${p.end_ts.toFixed(1)}s (${p.outcome})`}
              onClick={() => onSeek(p.start_ts, p.possession_id)}
              className={`absolute top-0 h-full border-r border-hardwood/60 transition-opacity hover:opacity-80 ${
                OUTCOME_COLOR[p.outcome] ?? "bg-gray-400"
              } ${activeId === p.possession_id ? "ring-2 ring-inset ring-whistle" : ""}`}
              style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
            />
          );
        })}
      </div>
      <div className="mt-1.5 flex gap-3 text-[11px] uppercase tracking-wide text-ink/50">
        {Object.entries(OUTCOME_COLOR).map(([outcome, color]) => (
          <span key={outcome} className="flex items-center gap-1">
            <span className={`inline-block h-2 w-2 ${color}`} />
            {outcome}
          </span>
        ))}
      </div>
    </div>
  );
}
