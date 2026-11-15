import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { useApp } from "../App";
import { CameraView } from "../components/CameraView";
import { useToast } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";

const FRAMES = 10;
const CAPTURE_GAP_MS = 220;
const CAPTURE_TIMEOUT_MS = 20000;

const REASONS: Record<string, string> = {
  more_than_one_face: "Only one person in view",
  face_too_small: "Move closer",
  not_live: "Live face",
  no_face: "Face in view",
};

type Phase = "idle" | "capturing" | "saving";

export function Register() {
  const { refresh, go } = useApp();
  const camera = useCamera();
  const toast = useToast();
  const stream = useFaceStream(camera.capture, "register_check", camera.state === "on");
  const [name, setName] = useState("");
  const [studentId, setStudentId] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [shots, setShots] = useState<string[]>([]);
  const [flash, setFlash] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [duplicate, setDuplicate] = useState<{ name: string; score: number } | null>(null);
  const framesRef = useRef<Blob[]>([]);
  const qualityRef = useRef(false);

  const faces = stream.result?.faces ?? [];
  const face = faces.length === 1 ? faces[0] : null;
  const checks = [
    { id: "face", label: "Face in view", ok: faces.length >= 1 },
    { id: "one", label: "Only one person in view", ok: faces.length <= 1 },
    { id: "size", label: "Close enough", ok: !!face && !face.quality?.reasons.includes("face_too_small") },
    { id: "live", label: "Live face", ok: !!face && face.liveness?.label === "live" },
  ];
  const ready = checks.every((c) => c.ok);
  qualityRef.current = ready;

  const shotsRef = useRef<string[]>([]);
  shotsRef.current = shots;
  useEffect(() => () => shotsRef.current.forEach((url) => URL.revokeObjectURL(url)), []);

  const submit = async (blobs: Blob[], force = false) => {
    setPhase("saving");
    setError(null);
    try {
      const result = await api.register(name.trim(), studentId.trim(), blobs, force);
      toast({ tone: "live", title: `${result.name} registered`, body: `Saved from ${result.used_frames} frames.` });
      refresh();
      setShots([]);
      setName("");
      setStudentId("");
      setDuplicate(null);
      go("live");
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        const d = (err.detail as { duplicate_of: { name: string; score: number } }).duplicate_of;
        setDuplicate(d);
      } else if (err instanceof ApiError && err.status === 422) {
        const detail = err.detail as { checks?: { reason: string | null }[] };
        const reasons = [...new Set((detail.checks ?? []).map((c) => c.reason).filter(Boolean))] as string[];
        setError(
          `Not enough usable frames. ${reasons.map((r) => REASONS[r] ?? r).join(", ").toLowerCase() || "Check the camera"} and try again.`,
        );
      } else {
        setError(err instanceof Error ? err.message : "Registration failed. Try again.");
      }
    } finally {
      setPhase("idle");
    }
  };

  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    setPhase("capturing");
    setError(null);
    setDuplicate(null);
    setShots([]);
    framesRef.current = [];
    const started = performance.now();
    // chỉ chụp khi đủ điều kiện, lặp tới khi đủ 10 frame hoặc hết thời gian
    while (framesRef.current.length < FRAMES && performance.now() - started < CAPTURE_TIMEOUT_MS) {
      await new Promise((r) => setTimeout(r, CAPTURE_GAP_MS));
      if (!qualityRef.current) continue;
      const blob = await camera.capture(0.92);
      if (!blob) continue;
      framesRef.current.push(blob);
      setShots((list) => [...list, URL.createObjectURL(blob)]);
      setFlash((n) => n + 1);
    }
    if (framesRef.current.length < FRAMES) {
      setPhase("idle");
      setError("Capture timed out. Follow the checklist under the camera, then try again.");
      return;
    }
    await submit(framesRef.current);
  };

  const busy = phase !== "idle";

  return (
    <div className="register-layout">
      <section className="stage" aria-label="Camera">
        <CameraView videoRef={camera.videoRef} state={camera.state} flashKey={flash}>
          <div className={`guide-oval ${ready ? "is-ready" : ""}`} aria-hidden="true" />
        </CameraView>
        <ul className="checklist" aria-label="Before capturing">
          {checks.map((c) => (
            <li key={c.id} className={c.ok ? "ok" : ""}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                {c.ok ? <path d="M5 12.5l4.5 4.5L19 7.5" /> : <circle cx="12" cy="12" r="6" />}
              </svg>
              {c.label}
            </li>
          ))}
        </ul>
      </section>

      <aside className="side">
        <h1 className="page-title">Register a person</h1>
        <p className="muted">
          Look at the camera and turn your head slightly left and right while {FRAMES} frames are taken.
        </p>
        <form className="form" onSubmit={start}>
          <label className="field">
            <span>Full name</span>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} required disabled={busy} autoComplete="off" />
          </label>
          <label className="field">
            <span>Student ID <span className="muted">(optional)</span></span>
            <input className="input" value={studentId} onChange={(e) => setStudentId(e.target.value)} disabled={busy} autoComplete="off" />
          </label>

          <div className="shots" aria-label={`${shots.length} of ${FRAMES} frames captured`}>
            {Array.from({ length: FRAMES }, (_, i) => (
              <div key={i} className={shots[i] ? "shot filled" : "shot"}>
                {shots[i] && <img src={shots[i]} alt="" />}
              </div>
            ))}
          </div>
          <div className="progress" aria-hidden="true">
            <div className="progress-fill" style={{ width: `${(shots.length / FRAMES) * 100}%` }} />
          </div>

          {error && <p className="form-error" role="alert">{error}</p>}
          {duplicate && (
            <div className="notice" role="alert">
              This face already matches <strong>{duplicate.name}</strong> (score {duplicate.score.toFixed(2)}).
              <div className="notice-actions">
                <button type="button" className="button" disabled={busy} onClick={() => submit(framesRef.current, true)}>
                  Register anyway
                </button>
                <button type="button" className="button button-ghost" onClick={() => setDuplicate(null)}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          <button type="submit" className="button button-primary" disabled={busy || !name.trim() || camera.state !== "on"}>
            {phase === "capturing" ? `Capturing ${shots.length}/${FRAMES}…` : phase === "saving" ? "Saving…" : "Capture and register"}
          </button>
        </form>
      </aside>
    </div>
  );
}
