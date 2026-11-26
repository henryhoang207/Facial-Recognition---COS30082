import { useEffect, useRef, useState } from "react";
import { CameraView } from "../components/CameraView";
import { Empty, Meter } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import type { GestureResult, IndividualInfo } from "../types";
import { FeatureUnavailable, StageBar, useFeatureInfo } from "./shared";

const STAGES = [
  { key: "hand", label: "Hand tracker" },
  { key: "gesture", label: "MLP + GRU" },
];

// các cặp điểm nối thành khung bàn tay (21 điểm MediaPipe)
const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const STABLE_RESULTS = 3; // cùng một ký hiệu bấy nhiêu lần liền mới ghi vào dòng chữ
const MIN_CONFIDENCE = 0.6;

const pretty = (label: string) => label.replace(/_/g, " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());

export function HandGesture() {
  const { loading, info } = useFeatureInfo("gesture");
  if (loading) return <Empty>Loading…</Empty>;
  if (!info?.available) return <FeatureUnavailable mode="gesture" info={info} />;
  return <GestureLive info={info} />;
}

function useTranscript(gesture?: GestureResult) {
  const [words, setWords] = useState<{ id: number; label: string }[]>([]);
  const streak = useRef<{ label: string | null; count: number; written: boolean }>({ label: null, count: 0, written: false });
  const nextId = useRef(0);
  useEffect(() => {
    const s = streak.current;
    const label = gesture?.status === "detected" && (gesture.confidence ?? 0) >= MIN_CONFIDENCE ? gesture.label : null;
    if (label !== s.label) {
      streak.current = { label, count: label ? 1 : 0, written: false };
      return;
    }
    if (!label) return;
    s.count += 1;
    if (s.count >= STABLE_RESULTS && !s.written) {
      s.written = true;
      setWords((list) => [...list.slice(-11), { id: nextId.current++, label }]);
    }
  }, [gesture]);
  return { words, clear: () => setWords([]) };
}

function GestureLive({ info }: { info: IndividualInfo }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "gesture", camera.state === "on");
  const gesture = stream.result?.gesture;
  const { words, clear } = useTranscript(gesture);
  const classes = (Array.isArray(info.classes) ? (info.classes as string[]) : []).filter((c) => c !== "OTHER");
  const detected = gesture?.status === "detected" ? gesture : null;

  return (
    <div className="live-layout">
      <section className="stage" aria-label="Camera">
        <CameraView
          videoRef={camera.videoRef}
          state={camera.state}
          frameSize={stream.result?.frame_size}
          renderOverlay={(scale, [width, height]) =>
            gesture?.points && (
              <svg className="hand-skeleton" width={width * scale} height={height * scale} aria-hidden="true">
                {BONES.map(([a, b]) => (
                  <line key={`${a}-${b}`}
                    x1={(1 - gesture.points![a][0]) * width * scale} y1={gesture.points![a][1] * height * scale}
                    x2={(1 - gesture.points![b][0]) * width * scale} y2={gesture.points![b][1] * height * scale} />
                ))}
                {gesture.points.map(([x, y], i) => (
                  <circle key={i} cx={(1 - x) * width * scale} cy={y * height * scale} r={i % 4 === 0 ? 5 : 3.5} />
                ))}
              </svg>
            )
          }
        >
          {detected && (
            <div className="sign-bubble" key={detected.label} role="status">
              {pretty(detected.label!)}
              <span className="num">{Math.round((detected.confidence ?? 0) * 100)}%</span>
            </div>
          )}
        </CameraView>
        <StageBar camera={camera} stream={stream} stages={STAGES} />
      </section>

      <aside className="side analysis" aria-label="Hand sign">
        <div className="analysis-card">
          <div className="model-block">
            <p className="muted small">Current sign</p>
            <p className={`sign-now ${detected ? "" : "is-idle"}`}>{detected ? pretty(detected.label!) : "–"}</p>
            <p className="muted small">{statusText(gesture)}</p>
            {gesture?.status === "collecting" && <Meter label="Motion collected" value={gesture.progress} tone="quiet" detail={`${Math.round(gesture.progress * 100)}%`} />}
          </div>
          {gesture && gesture.top.length > 0 && (
            <div className="model-block">
              {gesture.top.map((t) => (
                <Meter key={t.label} label={pretty(t.label)} value={t.prob} tone={t.label === gesture.label ? "accent" : "quiet"} detail={`${Math.round(t.prob * 100)}%`} />
              ))}
            </div>
          )}
        </div>

        <div className="analysis-card">
          <div className="model-block">
            <div className="model-head">
              <h3>Signed so far</h3>
              {words.length > 0 && <button type="button" className="link-button" onClick={clear}>Clear</button>}
            </div>
            {words.length === 0 ? (
              <p className="muted small">A sign is added once it is held steady for a moment.</p>
            ) : (
              <p className="transcript">
                {words.map((w) => <span key={w.id} className="word">{pretty(w.label)}</span>)}
              </p>
            )}
          </div>
          <div className="model-block">
            <p className="muted small">Signs this model knows</p>
            <div className="chips">
              {classes.map((c) => <span key={c} className={`chip ${detected?.label === c ? "is-on" : ""}`}>{pretty(c)}</span>)}
            </div>
            <p className="muted small">
              MediaPipe hand landmarks over the last {Number(info.window_s)} s, resampled to {Number(info.sequence_length)} steps,
              then an MLP + GRU.
              {typeof info.val_accuracy === "number" && <> Validation accuracy {(info.val_accuracy * 100).toFixed(1)}%</>}
              {typeof info.val_clip_macro_f1 === "number" && <>, clip macro F1 {info.val_clip_macro_f1.toFixed(2)}.</>}
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}

function statusText(gesture?: GestureResult) {
  if (!gesture) return "Waiting for the camera…";
  if (gesture.status === "no_hand") return "Show one hand to the camera.";
  if (gesture.status === "collecting") return "Keep signing: the model needs about one second of movement.";
  if (gesture.status === "other") return "No known sign right now.";
  return `Hand found · ${Math.round((gesture.confidence ?? 0) * 100)}% sure`;
}
