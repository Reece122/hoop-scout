"use client";

import type { ReportDocument } from "@/lib/types";

function confidenceStyle(c: number): string {
  if (c >= 0.75) return "border-emerald-600 text-emerald-700";
  if (c >= 0.4) return "border-whistle text-amber-700";
  return "border-court-orange-dark text-court-orange-dark";
}

export default function ReportView({
  report,
  onCiteClick,
}: {
  report: ReportDocument;
  onCiteClick: (ts: number, possessionId: string) => void;
}) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="font-display mb-2 text-lg font-semibold uppercase tracking-wide text-ink">
          Scouting Report
        </h2>
        <div className="whitespace-pre-wrap border-l-4 border-hardwood-line/30 py-1 pl-4 text-sm leading-relaxed text-ink/90">
          {report.report_text}
        </div>
      </div>

      <div>
        <h3 className="font-display mb-3 text-base font-semibold uppercase tracking-wide text-ink">
          Findings <span className="text-ink/40">({report.findings.length})</span>
        </h3>
        <div className="space-y-3">
          {report.findings.map((f) => (
            <div key={f.finding_id} className="border-l-4 border-hardwood-line/30 py-1 pl-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-ink/90">{f.claim}</p>
                <span
                  className={`shrink-0 border px-1.5 py-0.5 font-mono text-[11px] ${confidenceStyle(
                    f.confidence
                  )}`}
                >
                  {(f.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {f.citations.map((c) => (
                  <button
                    key={c.possession_id}
                    onClick={() => onCiteClick(c.start_ts, c.possession_id)}
                    className="border border-court-orange/40 bg-court-orange/5 px-2 py-0.5 font-mono text-xs text-court-orange-dark hover:bg-court-orange/15"
                  >
                    {c.possession_id} @ {c.start_ts.toFixed(1)}s
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
