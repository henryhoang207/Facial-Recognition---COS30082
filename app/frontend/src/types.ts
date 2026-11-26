export type IndividualMode = "challenge" | "gesture" | "head_pose" | "face_parsing";
export type Mode =
  | "pipeline" | "liveness" | "recognition" | "emotion" | "register_check" | "mask" | "attendance_mlp"
  | IndividualMode;

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

export interface MaskResult {
  label: "mask" | "no_mask";
  prob: number; // P(mask), trung bình trong cửa sổ làm mượt
  raw: number;
  threshold: number;
}

export interface ChallengeResult {
  state: "checking" | "challenge" | "passed" | "failed";
  reason: "live" | "challenge" | "timeout" | null;
  live_prob: number;
  raw: number;
  threshold: number;
  frames: number;
  required_frames: number;
  yaw: number | null; // độ, > 0 = người quay sang trái của họ
  turn_deg: number;
  steps: ("left" | "right")[];
  step_index: number;
  step: "left" | "right" | null;
  hold: number;
  hold_frames: number;
  seconds_left: number | null;
  retry_in: number | null;
  identity: Identity | null;
}

export interface HeadPoseResult {
  yaw: number;
  pitch: number;
  roll: number;
  direction: "frontal" | "left" | "right" | "up" | "down";
  landmarks: [number, number][] | null;
  source: "model" | "detector" | "starting"; // detector = landmark hỏng; starting = chưa có kết quả model
}

export interface FaceParsingResult {
  overlay: string; // PNG data URL phủ lên vùng `region`
  region: [number, number, number, number];
  parts: { name: string; color: string; share: number }[];
}

export interface GestureResult {
  status: "no_hand" | "collecting" | "detected" | "other";
  label: string | null;
  confidence: number | null;
  top: { label: string; prob: number }[];
  progress: number;
  handedness: string | null;
  points: [number, number][] | null; // 21 điểm, toạ độ chuẩn hoá 0..1 theo frame
}

export interface Face {
  track_id: number;
  box: [number, number, number, number];
  liveness: Liveness | null;
  identity: Identity | null;
  emotion: Emotion | null;
  logged: boolean;
  quality?: { ok: boolean; reasons: string[]; face_pixels: number };
  mask?: MaskResult | null;
  attendance_mlp?: {
    state: "needs_gallery" | "collecting" | "accepted" | "rejected";
    frames: number;
    required_frames: number;
    gallery_size: number;
    required_gallery_size: number | null;
    score: number | null;
    threshold: number;
    candidate_name: string | null;
  };
  challenge?: ChallengeResult | null;
  head_pose?: HeadPoseResult | null;
  face_parsing?: FaceParsingResult | null;
}

/** Chữ cái ASL tĩnh trên trang Live (mode "pipeline"). */
export interface AlphabetResult {
  status: "detected" | "no_hand" | "unavailable";
  label: string | null;
  confidence: number | null;
}

export interface StreamResult {
  type: "result";
  frame_id: number;
  mode: Mode;
  frame_size: [number, number];
  latency_ms: Record<string, number>;
  faces: Face[];
  // mode "pipeline": AlphabetResult; mode "gesture" (tab Đức Anh): GestureResult
  gesture?: AlphabetResult | GestureResult;
  face_parsing?: { error: string | null };
  head_pose?: { error: string | null };
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

export interface IndividualInfo {
  type: string;
  path: string | null;
  available: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface Health {
  ok: boolean;
  models: Record<string, Record<string, unknown>> & {
    load_seconds?: number;
    individual?: Partial<Record<IndividualMode, IndividualInfo>>;
  };
  thresholds: { recognition: number; liveness: number; mask: number | null };
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
  thresholds: { recognition: number | null; liveness: number | null; mask: number | null };
  pipeline: Record<string, number>;
  features: Record<string, boolean>;
  effective: { recognition: number; liveness: number; mask: number | null };
}
