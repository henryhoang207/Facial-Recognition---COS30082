import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { useApp } from "../App";
import { CameraView } from "../components/CameraView";
import { useToast } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";

const FRAMES = 10;

// gợi ý tư thế cho từng ảnh để embedding đa dạng góc mặt
const POSES = [
  "Look straight at the camera",
  "Look straight, neutral face",
  "Turn your head slightly left",
  "Turn a little more to the left",
  "Turn your head slightly right",
  "Turn a little more to the right",
  "Tilt your chin up a little",
  "Tilt your chin down a little",
  "Look straight and smile",
  "Look straight, last one",
];

const REASONS: Record<string, string> = {
  more_than_one_face: "Only one person in view",
  face_too_small: "Move closer",
  not_live: "Live face",
  no_face: "Face in view",
};

interface Shot {
  blob: Blob;
  url: string;
}

type Phase = "idle" | "saving";

export function Register() {
  const { refresh, go } = useApp();
  const camera = useCamera();
  const toast = useToast();
  const stream = useFaceStream(camera.capture, "register_check", camera.state === "on");
  const [name, setName] = useState("");
  const [studentId, setStudentId] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [shots, setShots] = useState<Shot[]>([]);
  const [flash, setFlash] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [duplicate, setDuplicate] = useState<{ name: string; score: number } | null>(null);

  const faces = stream.result?.faces ?? [];
  const face = faces.length === 1 ? faces[0] : null;
  const checks = [
    { id: "face", label: "Face in view", ok: faces.length >= 1 },
    { id: "one", label: "Only one person in view", ok: faces.length <= 1 },
    { id: "size", label: "Close enough", ok: !!face && !face.quality?.reasons.includes("face_too_small") },
    { id: "live", label: "Live face", ok: !!face && face.liveness?.label === "live" },
  ];
  const ready = checks.every((c) => c.ok);
  const full = shots.length >= FRAMES;
  const busy = phase !== "idle";
  const canShoot = ready && !full && !busy && camera.state === "on";

  const shotsRef = useRef<Shot[]>([]);
  shotsRef.current = shots;
  useEffect(() => () => shotsRef.current.forEach((s) => URL.revokeObjectURL(s.url)), []);

  const takePhoto = useCallback(async () => {
    if (!canShoot) return;
    const blob = await camera.capture(0.92);
    if (!blob) return;
    setShots((list) => (list.length >= FRAMES ? list : [...list, { blob, url: URL.createObjectURL(blob) }]));
    setFlash((n) => n + 1);
    setError(null);
    setDuplicate(null);
  }, [canShoot, camera]);

  // phím Space để chụp, trừ khi đang gõ trong ô nhập
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.code !== "Space" || target.closest("input, textarea, select, button")) return;
      event.preventDefault();
      takePhoto();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [takePhoto]);

  const removeShot = (index: number) => {
    setShots((list) => {
      URL.revokeObjectURL(list[index].url);
      return list.filter((_, i) => i !== index);
    });
    setDuplicate(null);
  };

  const clearShots = () => {
    shots.forEach((s) => URL.revokeObjectURL(s.url));
    setShots([]);
    setError(null);
    setDuplicate(null);
  };

  const submit = async (force = false) => {
    setPhase("saving");
    setError(null);
    try {
      const result = await api.register(name.trim(), studentId.trim(), shots.map((s) => s.blob), force);
      toast({ tone: "live", title: `${result.name} registered`, body: `Saved from ${result.used_frames} photos.` });
      refresh();
      clearShots();
      setName("");
      setStudentId("");
      go("live");
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setDuplicate((err.detail as { duplicate_of: { name: string; score: number } }).duplicate_of);
      } else if (err instanceof ApiError && err.status === 422) {
        const detail = err.detail as { checks?: { reason: string | null }[] };
        const reasons = [...new Set((detail.checks ?? []).map((c) => c.reason).filter(Boolean))] as string[];
        setError(
          `Not enough usable photos. ${reasons.map((r) => REASONS[r] ?? r).join(", ").toLowerCase() || "Check the camera"}, then retake the bad ones.`,
        );
      } else {
        setError(err instanceof Error ? err.message : "Registration failed. Try again.");
      }
    } finally {
      setPhase("idle");
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (name.trim() && full) submit();
  };

  const blocker = checks.find((c) => !c.ok);
  const shootHint = full
    ? "All photos taken. Remove one to retake it."
    : camera.state !== "on"
      ? "Waiting for the camera…"
      : blocker
        ? `Not ready: ${blocker.label.toLowerCase()}`
        : "Ready. Click the button or press Space.";

  return (
    <div className="register-layout">
      <section className="stage" aria-label="Camera">
        <CameraView videoRef={camera.videoRef} state={camera.state} flashKey={flash}>
          <div className={`guide-oval ${ready ? "is-ready" : ""}`} aria-hidden="true" />
          {!full && camera.state === "on" && (
            <div className="pose-hint" key={shots.length}>
              <span className="num">{shots.length + 1}/{FRAMES}</span>
              {POSES[shots.length]}
            </div>
          )}
        </CameraView>

        <div className="capture-bar">
          <button
            type="button"
            className="button button-primary capture-button"
            onClick={takePhoto}
            disabled={!canShoot}
            aria-describedby="shoot-hint"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1.5-2h6l1.5 2h2A1.5 1.5 0 0 1 20 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z" />
              <circle cx="12" cy="13" r="3.5" />
            </svg>
            Take photo
            <span className="num capture-count">{shots.length}/{FRAMES}</span>
          </button>
          <p id="shoot-hint" className={`small ${ready || full ? "muted" : "warn"}`}>{shootHint}</p>
        </div>

        <ul className="checklist" aria-label="Before taking a photo">
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
          Take {FRAMES} photos yourself, following the hint on the camera: straight, then slightly left, right, up and down.
        </p>
        <form className="form" onSubmit={onSubmit}>
          <label className="field">
            <span>Full name</span>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} required disabled={busy} autoComplete="off" />
          </label>
          <label className="field">
            <span>Student ID <span className="muted">(optional)</span></span>
            <input className="input" value={studentId} onChange={(e) => setStudentId(e.target.value)} disabled={busy} autoComplete="off" />
          </label>

          <div className="shots-head">
            <span className="small strong">Photos</span>
            {shots.length > 0 && (
              <button type="button" className="link-button" onClick={clearShots} disabled={busy}>
                Retake all
              </button>
            )}
          </div>
          <div className="shots" aria-label={`${shots.length} of ${FRAMES} photos taken`}>
            {Array.from({ length: FRAMES }, (_, i) => (
              <div key={shots[i]?.url ?? `empty-${i}`} className={shots[i] ? "shot filled" : i === shots.length ? "shot next" : "shot"}>
                {shots[i] ? (
                  <>
                    <img src={shots[i].url} alt={`Photo ${i + 1}`} />
                    <button type="button" className="shot-remove" onClick={() => removeShot(i)} disabled={busy} aria-label={`Remove photo ${i + 1}`}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" aria-hidden="true">
                        <path d="M6 6l12 12M18 6L6 18" />
                      </svg>
                    </button>
                  </>
                ) : (
                  <span className="num shot-index">{i + 1}</span>
                )}
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
                <button type="button" className="button" disabled={busy} onClick={() => submit(true)}>
                  Register anyway
                </button>
                <button type="button" className="button button-ghost" onClick={() => setDuplicate(null)}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          <button type="submit" className="button button-primary" disabled={busy || !name.trim() || !full}>
            {phase === "saving" ? "Saving…" : full ? "Register" : `Register (${FRAMES - shots.length} more photo${FRAMES - shots.length === 1 ? "" : "s"})`}
          </button>
        </form>
      </aside>
    </div>
  );
}
