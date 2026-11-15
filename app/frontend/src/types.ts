export type Mode = "pipeline" | "liveness" | "recognition" | "emotion" | "register_check";

export interface Liveness {
  label: "live" | "spoof";
  prob: number;
  raw: number;
  threshold: number;
  fresh: boolean;
}

export interface Identity {
  person_id: number | null;
  name: string | null;
  score: number;
  known: boolean;
  threshold: number;
  top?: { person_id: number; name: string; score: number }[];
}

export interface Emotion {
  label: string;
  probs: Record<string, number>;
}

export interface Face {
  track_id: number;
  box: [number, number, number, number];
  liveness: Liveness | null;
  identity: Identity | null;
  emotion: Emotion | null;
  logged: boolean;
  quality?: { ok: boolean; reasons: string[]; face_pixels: number };
}

export interface StreamResult {
  type: "result";
  frame_id: number;
  mode: Mode;
  frame_size: [number, number];
  latency_ms: Record<string, number>;
  faces: Face[];
}

export interface AttendanceEvent {
  type: "attendance_logged";
  session_id: number;
  person_id: number;
  name: string;
  time: string;
  emotion: string | null;
  match_score: number;
}

export interface SpoofEvent {
  type: "spoof_blocked";
  session_id: number;
  time: string;
  live_prob: number;
  snapshot?: string;
}

export type StreamEvent = AttendanceEvent | SpoofEvent;

export interface Health {
  ok: boolean;
  models: Record<string, Record<string, unknown>> & { load_seconds?: number };
  thresholds: { recognition: number; liveness: number };
  gallery_size: number;
  embedding_dim: number;
  active_session_id: number;
  mock_mode: boolean;
}

export interface Session {
  id: number;
  name: string;
  started_at: string;
  ended_at: string | null;
  active: boolean;
}

export interface AttendanceRow {
  person_id: number;
  name: string;
  student_id: string | null;
  first_seen: string;
  live_prob: number;
  match_score: number;
  emotion: string | null;
}

export interface AttendanceReport {
  session: Session;
  stats: {
    registered: number;
    present: number;
    spoof_blocked: number;
    emotions: Record<string, number>;
  };
  present: AttendanceRow[];
  absent: { person_id: number; name: string; student_id: string | null }[];
}

export interface SpoofRow {
  id: number;
  session_id: number;
  ts: string;
  live_prob: number;
  snapshot_path: string | null;
}

export interface Person {
  id: number;
  name: string;
  student_id: string | null;
  created_at: string;
  embedding_ok: boolean;
}

export interface Settings {
  thresholds: { recognition: number | null; liveness: number | null };
  pipeline: Record<string, number>;
  features: Record<string, boolean>;
  effective: { recognition: number; liveness: number };
}
