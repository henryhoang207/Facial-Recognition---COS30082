import type {
  AttendanceReport,
  Health,
  Person,
  Session,
  Settings,
  SpoofRow,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : detailMessage(detail) ?? `HTTP ${status}`);
  }
}

function detailMessage(detail: unknown): string | undefined {
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return undefined;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail: unknown = response.statusText;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* body không phải JSON */
    }
    throw new ApiError(response.status, detail);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  settings: () => request<Settings>("/api/settings"),
  updateSettings: (patch: Partial<Omit<Settings, "effective">>) =>
    request<Settings>("/api/settings", json("PUT", patch)),
  reloadModels: () => request<Health>("/api/models/reload", { method: "POST" }),
  persons: () => request<Person[]>("/api/persons"),
  deletePerson: (id: number) => request<void>(`/api/persons/${id}`, { method: "DELETE" }),
  register: (name: string, studentId: string, frames: Blob[], force = false) => {
    const form = new FormData();
    form.append("name", name);
    if (studentId) form.append("student_id", studentId);
    form.append("force", String(force));
    frames.forEach((frame, i) => form.append("frames", frame, `frame${i}.jpg`));
    return request<{ person_id: number; name: string; used_frames: number }>(
      "/api/persons/register",
      { method: "POST", body: form },
    );
  },
  sessions: () => request<Session[]>("/api/sessions"),
  startSession: (name: string) => request<Session>("/api/sessions", json("POST", { name })),
  endSession: (id: number) => request<Session>(`/api/sessions/${id}/end`, { method: "PATCH" }),
  attendance: (id: number) => request<AttendanceReport>(`/api/sessions/${id}/attendance`),
  spoofEvents: (id: number) => request<SpoofRow[]>(`/api/sessions/${id}/spoof-events`),
  exportUrl: (id: number) => `/api/sessions/${id}/export.csv`,
  snapshotUrl: (path: string) => `/api/snapshots/${path}`,
};

export function wsUrl(mode: string): string {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/ws/stream?mode=${mode}`;
}

export const timeOf = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

export const dateOf = (iso: string) =>
  new Date(iso).toLocaleDateString([], { day: "numeric", month: "short" });
