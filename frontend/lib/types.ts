export type Outcome = "made_2" | "made_3" | "missed" | "turnover" | "foul" | "unknown";

export interface PositionSample {
  ts: number;
  track_positions: Record<string, [number, number]>;
  ball_position: [number, number] | null;
}

export interface DerivedFeatures {
  avg_spacing_ft: number | null;
  duration_sec: number;
  court_side: string | null;
  notes: string;
}

export interface Possession {
  possession_id: string;
  start_ts: number;
  end_ts: number;
  start_frame: number;
  end_frame: number;
  offense_track_ids: number[];
  defense_track_ids: number[];
  ball_handler_track_id: number | null;
  ball_handler_confidence: number;
  outcome: Outcome;
  outcome_confidence: number;
  position_time_series: PositionSample[];
  derived_features: DerivedFeatures;
  clip_path: string | null;
}

export interface GameState {
  video: { path: string; fps: number; width: number; height: number; duration_sec: number };
  calibration: unknown;
  roster: { track_id_to_name: Record<string, string>; unmapped_track_ids: number[] };
  ball: { track_available: boolean; detections: unknown[] };
  possessions: Possession[];
  meta: { generated_at: string; pipeline_version: string; detector: string; notes: string[] };
}

export interface ToolCallEntry {
  tool_name: string;
  arguments: Record<string, unknown>;
  result_summary: string;
  ts: number;
}

export interface Citation {
  possession_id: string;
  start_ts: number;
  end_ts: number;
  clip_path: string | null;
}

export interface Finding {
  finding_id: string;
  claim: string;
  confidence: number;
  category: string | null;
  citations: Citation[];
}

export interface ReportDocument {
  report_text: string;
  findings: Finding[];
  tool_log: ToolCallEntry[];
  video_path: string;
}

export interface AgentRun {
  status: "running" | "done" | "error";
  events: ToolCallEntry[];
  result: ReportDocument | null;
  error?: string;
}
