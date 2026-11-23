import { useCallback, useEffect, useState } from "react";
import { api, timeOf } from "../api";
import { useApp } from "../App";
import { CameraView } from "../components/CameraView";
import { EmotionIcon, Empty, Meter, useToast } from "../components/ui";
import { type CameraState, useCamera } from "../hooks/useCamera";
import { type StreamStatus, useFaceStream } from "../hooks/useFaceStream";
import type { AttendanceRow, Face, StreamEvent, StreamResult } from "../types";

const STAGES = [
  { key: "detect", label: "Detect" },
  { key: "liveness", label: "Anti-spoofing" },
  { key: "embed", label: "Recognition" },
  { key: "emotion", label: "Emotion" },
  { key: "gesture", label: "ASL" },
];

export function Live() {
  const { session, health, go, refresh } = useApp();
  const camera = useCamera();
  const toast = useToast();
  const [flash, setFlash] = useState(0);
  const [recent, setRecent] = useState<AttendanceRow[]>([]);
  const [freshId, setFreshId] = useState<number | null>(null);
  const [pickedTrack, setPickedTrack] = useState<number | null>(null);

  const loadRecent = useCallback(() => {
    if (!session) return;
    api.attendance(session.id).then((r) => setRecent([...r.present].reverse())).catch(() => undefined);
  }, [session]);
  useEffect(loadRecent, [loadRecent]);

  const onEvent = useCallback(
    (event: StreamEvent) => {
      if (event.type === "attendance_logged") {
        setFlash((n) => n + 1);
        setFreshId(event.person_id);
        toast({
          tone: "live",
          title: `${event.name} checked in`,
          body: event.emotion ? `${event.time}, looking ${event.emotion.toLowerCase()}` : event.time,
        });
        loadRecent();
        if (event.session_id !== session?.id) refresh();
      } else {
        toast({ tone: "spoof", title: "Spoof blocked", body: `Not a live face (${(event.live_prob * 100).toFixed(0)}% live). Nothing was logged.` });
      }
    },
    [toast, loadRecent, refresh, session?.id],
  );

  const stream = useFaceStream(camera.capture, "pipeline", camera.state === "on", onEvent);
  const faces = stream.result?.faces ?? [];
  const latency = stream.result?.latency_ms ?? {};
  // mặc định chọn mặt lớn nhất (server trả về theo thứ tự diện tích giảm dần)
  const selected = faces.find((f) => f.track_id === pickedTrack) ?? faces[0];

  return (
    <div className="live-layout">
      <section className="stage" aria-label="Camera">
        <CameraView
          videoRef={camera.videoRef}
          state={camera.state}
          faces={faces}
          frameSize={stream.result?.frame_size}
          flashKey={flash}
          selectedTrack={faces.length > 1 ? selected?.track_id : undefined}
          onSelect={setPickedTrack}
          tone={(f) => (f.liveness?.label === "spoof" ? "spoof" : f.identity?.known ? "live" : "neutral")}
          renderLabel={(f) => <FaceLabel face={f} />}
        />
        <div className="stage-bar">
          <StreamDot status={stream.status} />
          <span className="num">{stream.fps.toFixed(1)} fps</span>
          <div className="pipeline-strip" aria-label="Time per step for the last frame">
            {STAGES.map((s) => (
              <span key={s.key} className={latency[s.key] !== undefined ? "step on" : "step"}>
                {s.label}
                <span className="num">{latency[s.key] !== undefined ? `${Math.round(latency[s.key])} ms` : "–"}</span>
              </span>
            ))}
          </div>
          {camera.devices.length > 1 && (
            <select className="select" aria-label="Camera" value={camera.deviceId} onChange={(e) => camera.setDeviceId(e.target.value)}>
              {camera.devices.map((d, i) => (
                <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>
              ))}
            </select>
          )}
        </div>

      </section>

      <aside className="side analysis" aria-label="Model results">
        <GestureAnalysis camera={camera.state} status={stream.status} result={stream.result} />
        {faces.length > 1 && (
          <div className="face-picker" role="tablist" aria-label="Face">
            {faces.map((f, i) => (
              <button
                key={f.track_id}
                type="button"
                role="tab"
                aria-selected={f.track_id === selected?.track_id}
                className={`face-pick tone-${faceTone(f)}`}
                onClick={() => setPickedTrack(f.track_id)}
              >
                {faceName(f) ?? `Face ${i + 1}`}
              </button>
            ))}
          </div>
        )}
        {!selected ? (
          <div className="analysis-empty">
            <Empty>No face in view. Step in front of the camera to see all three models at work.</Empty>
          </div>
        ) : (
          <FaceAnalysis
            key={selected.track_id}
            face={selected}
            models={health?.models}
            onRegister={() => go("register")}
          />
        )}
      </aside>

      <section className="checkins" aria-label="Check-ins">
        <div className="side-head">
          <h2 className="side-title">Checked in this session</h2>
          <a href="#/attendance" className="link">All attendance</a>
        </div>
        {recent.length === 0 ? (
          <Empty>Nobody has checked in to this session yet.</Empty>
        ) : (
          <ul className="checkin-strip">
            {recent.slice(0, 12).map((row) => (
              <li key={row.person_id} className={row.person_id === freshId ? "checkin-pill fresh" : "checkin-pill"}>
                {row.emotion && <EmotionIcon label={row.emotion} size={15} />}
                <span className="checkin-name">{row.name}</span>
                <span className="num muted">{timeOf(row.first_seen)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function GestureAnalysis({
  camera,
  status,
  result,
}: {
  camera: CameraState;
  status: StreamStatus;
  result: StreamResult | null;
}) {
  const gesture = camera === "on" && status === "live" ? result?.gesture : undefined;
  const label = gesture?.status === "detected" ? gesture.label : null;
  const confidence = label ? gesture?.confidence : null;

  let message = "Show one hand to the camera.";
  if (camera === "starting") message = "Starting camera…";
  else if (camera === "denied") message = "Allow camera access to recognize ASL letters.";
  else if (camera !== "on") message = "Connect a camera to recognize ASL letters.";
  else if (status === "reconnecting") message = "Reconnecting to the live stream…";
  else if (status !== "live" || !result) message = "Waiting for the live stream…";
  else if (!gesture || gesture.status === "unavailable") message = "ASL recognition is unavailable.";

  return (
    <section className="analysis-card gesture-card" aria-labelledby="gesture-title">
      <h2 className="analysis-name" id="gesture-title">ASL letter</h2>
      <div className="gesture-result">
        <div aria-live="polite" aria-atomic="true">
          {label ? <p className="gesture-letter">{label}</p> : <p className="muted small">{message}</p>}
        </div>
        {confidence != null && (
          <Meter label="Confidence" value={confidence} detail={`${Math.round(confidence * 100)}%`} />
        )}
      </div>
    </section>
  );
}

const faceTone = (f: Face) => (f.liveness?.label === "spoof" ? "spoof" : f.identity?.known ? "live" : "neutral");
const faceName = (f: Face) =>
  f.liveness?.label === "spoof" ? "Spoof" : f.identity?.known ? f.identity.name : null;

function FaceLabel({ face }: { face: Face }) {
  if (face.liveness?.label === "spoof") return <span className="tag tag-spoof">Spoof blocked</span>;
  const name = face.identity?.known ? face.identity.name : "Not registered";
  return (
    <span className={face.identity?.known ? "tag tag-live" : "tag"}>
      {name}
      {face.emotion && <EmotionIcon label={face.emotion.label} size={15} />}
    </span>
  );
}

/** Kết quả của cả 3 model cho một khuôn mặt, theo đúng thứ tự pipeline. */
function FaceAnalysis({
  face,
  models,
  onRegister,
}: {
  face: Face;
  models?: Record<string, Record<string, unknown>>;
  onRegister: () => void;
}) {
  const live = face.liveness;
  const spoof = live?.label === "spoof";
  const identity = face.identity;
  const emotion = face.emotion;
  const emotionRows = emotion ? Object.entries(emotion.probs).sort((a, b) => b[1] - a[1]) : [];
  const skipped = <p className="skipped">Skipped: the face did not pass anti-spoofing.</p>;

  return (
    <div className={`analysis-card ${spoof ? "is-spoof" : ""}`}>
      <header className="analysis-head">
        <div>
          <p className="analysis-name">{spoof ? "Spoof attempt" : identity?.known ? identity.name : "Not registered"}</p>
          <p className="muted small">
            {spoof ? "Blocked before recognition. Nothing is logged." : face.logged ? "Checked in for this session." : identity?.known ? "Hold still to check in." : "This face is not in the gallery."}
          </p>
        </div>
        {face.logged && !spoof && <span className="badge badge-live">Checked in</span>}
      </header>

      <section className="model-block">
        <ModelHead step={1} title="Anti-spoofing" detail={modelFile(models?.liveness)} />
        {live && (
          <>
            <p className={`verdict ${spoof ? "warn" : "ok"}`}>{spoof ? "Spoof" : "Live face"}</p>
            <Meter label="This frame" value={live.raw} threshold={live.threshold} tone={live.raw < live.threshold ? "warn" : "accent"} />
            <Meter label="Smoothed" value={live.prob} threshold={live.threshold} tone={spoof ? "warn" : "accent"} />
          </>
        )}
      </section>

      <section className={`model-block ${spoof ? "is-skipped" : ""}`}>
        <ModelHead step={2} title="Face recognition" detail={modelFile(models?.embedder)} />
        {spoof || !identity ? (
          skipped
        ) : (
          <>
            <p className={`verdict ${identity.known ? "ok" : ""}`}>{identity.known ? identity.name : "No match"}</p>
            {(identity.top ?? []).length === 0 ? (
              <p className="muted small">No one is registered yet.</p>
            ) : (
              (identity.top ?? []).map((m) => (
                <Meter
                  key={m.person_id}
                  label={m.name}
                  value={Math.max(0, m.score)}
                  threshold={identity.threshold}
                  tone={m.score >= identity.threshold ? "accent" : "quiet"}
                />
              ))
            )}
            {!identity.known && (
              <button type="button" className="button button-small" onClick={onRegister}>
                Register this person
              </button>
            )}
          </>
        )}
      </section>

      <section className={`model-block ${spoof ? "is-skipped" : ""}`}>
        <ModelHead step={3} title="Emotion" detail={modelFile(models?.emotion)} />
        {spoof || !emotion ? (
          spoof ? skipped : <p className="muted small">Waiting for the first prediction…</p>
        ) : (
          <>
            <p className="verdict ok emotion-verdict">
              <EmotionIcon label={emotion.label} size={24} /> {emotion.label}
            </p>
            <div className="emotion-bars">
              {emotionRows.map(([label, p]) => (
                <div key={label} className={label === emotion.label ? "emotion-bar top" : "emotion-bar"}>
                  <span>{label}</span>
                  <div className="meter-track">
                    <div className={`meter-fill ${label === emotion.label ? "tone-accent" : "tone-quiet"}`} style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="num muted">{Math.round(p * 100)}%</span>
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function ModelHead({ step, title, detail }: { step: number; title: string; detail?: string }) {
  return (
    <div className="model-head">
      <span className="model-step" aria-hidden="true">{step}</span>
      <h3>{title}</h3>
      {detail && <span className="model-file" title={detail}>{detail.split("/").pop()}</span>}
    </div>
  );
}

const modelFile = (info?: Record<string, unknown>) => (info?.path ? String(info.path) : info?.type ? String(info.type) : undefined);

export function StreamDot({ status }: { status: string }) {
  const text = { live: "Live", connecting: "Connecting", reconnecting: "Reconnecting", paused: "Paused" }[status] ?? status;
  return (
    <span className={`stream-dot is-${status}`}>
      <span className="dot" aria-hidden="true" />
      {text}
    </span>
  );
}
