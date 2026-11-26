import { useState } from "react";
import { CameraView } from "../components/CameraView";
import { Empty } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import type { HeadPoseResult, IndividualInfo } from "../types";
import { AngleGauge, FeatureUnavailable, StageBar, useFeatureInfo } from "./shared";

const STAGES = [
  { key: "detect", label: "Detect" },
  { key: "head_pose", label: "Head pose (background)" },
];

const CONTRACT = "predict_frame(frame_bgr, box) -> {'yaw', 'pitch', 'roll'} in degrees, optional 'landmarks'";

const DIRECTION_TEXT: Record<HeadPoseResult["direction"], string> = {
  frontal: "Facing the camera",
  left: "Turned left",
  right: "Turned right",
  up: "Looking up",
  down: "Looking down",
};

export function HeadPose() {
  const { loading, info } = useFeatureInfo("head_pose");
  if (loading) return <Empty>Loading…</Empty>;
  if (!info?.available) return <FeatureUnavailable mode="head_pose" info={info} contract={CONTRACT} />;
  return <HeadPoseLive info={info} />;
}

function HeadPoseLive({ info }: { info: IndividualInfo }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "head_pose", camera.state === "on");
  const [picked, setPicked] = useState<number | null>(null);
  const faces = stream.result?.faces ?? [];
  const selected = faces.find((f) => f.track_id === picked) ?? faces[0];
  const pose = selected?.head_pose ?? null;
  const mock = info.type === "geometry";

  return (
    <div className="live-layout">
      <section className="stage" aria-label="Camera">
        <CameraView
          videoRef={camera.videoRef}
          state={camera.state}
          faces={faces}
          frameSize={stream.result?.frame_size}
          selectedTrack={faces.length > 1 ? selected?.track_id : undefined}
          onSelect={setPicked}
          tone={(f) => (f.head_pose?.direction === "frontal" ? "live" : "neutral")}
          renderLabel={(f) => f.head_pose && <span className="tag">{DIRECTION_TEXT[f.head_pose.direction]}</span>}
          renderOverlay={(scale, [width]) =>
            faces.some((f) => f.head_pose?.landmarks) && (
              <svg className="pose-points" aria-hidden="true">
                {faces.flatMap((f) =>
                  (f.head_pose?.landmarks ?? []).map(([x, y], i) => (
                    <circle key={`${f.track_id}-${i}`} cx={(width - x) * scale} cy={y * scale} r={2.2} />
                  )),
                )}
              </svg>
            )
          }
        />
        <StageBar camera={camera} stream={stream} stages={STAGES} />
      </section>

      <aside className="side analysis" aria-label="Head pose">
        {stream.result?.head_pose?.error && (
          <div className="notice" role="alert">
            <p className="strong">The model failed on the last frames.</p>
            <p className="small mono break">{stream.result.head_pose.error}</p>
            <p className="small">The full message is in the terminal running the app.</p>
          </div>
        )}
        {typeof info.gpu_error === "string" && (
          <p className="small muted">The GPU could not run this model, so it runs on the CPU ({info.gpu_error}).</p>
        )}
        {faces.length > 1 && (
          <div className="face-picker" role="tablist" aria-label="Face">
            {faces.map((f, i) => (
              <button key={f.track_id} type="button" role="tab" aria-selected={f.track_id === selected?.track_id}
                className="face-pick" onClick={() => setPicked(f.track_id)}>
                Face {i + 1}
              </button>
            ))}
          </div>
        )}
        {!pose ? (
          <div className="analysis-empty">
            <Empty>No face in view. Turn your head slowly to see the angles change.</Empty>
          </div>
        ) : (
          <div className="analysis-card">
            <div className="model-block pose-block">
              <HeadModel pose={pose} />
              <div>
                <p className="verdict">{DIRECTION_TEXT[pose.direction]}</p>
                <p className="muted small num">
                  yaw {pose.yaw.toFixed(0)}° · pitch {pose.pitch.toFixed(0)}° · roll {pose.roll.toFixed(0)}°
                </p>
                {!mock && pose.source === "detector" && (
                  <p className="small warn">Face partly hidden: using the detector's 5 points for now.</p>
                )}
                {!mock && pose.source === "starting" && (
                  <p className="small muted">Waiting for the first landmarks from the model…</p>
                )}
              </div>
            </div>
            <div className="model-block">
              <AngleGauge label="Yaw (left / right)" value={pose.yaw} range={60} mirror />
              <AngleGauge label="Pitch (up / down)" value={pose.pitch} range={45} />
              <AngleGauge label="Roll (tilt)" value={pose.roll} range={45} mirror />
            </div>
          </div>
        )}
        <div className="analysis-card">
          <div className="model-block">
            <div className="model-head"><h3>{mock ? "Estimated from 5 detector points" : "Thành's head pose model"}</h3></div>
            <p className="muted small">
              {mock
                ? "Rough angles from YuNet's eye, nose and mouth points."
                : `HRNet-W48 predicts 68 face landmarks (green dots); yaw, pitch and roll come from fitting them to a generic 3D face. Runs on ${String(info.device ?? "cpu").toUpperCase()} in ${String(info.precision ?? "fp32")}, on its own thread, so the camera keeps its frame rate and the points follow the face between results. If the landmarks disagree with the detector (a hand over the face), the detector's points are used.`}
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}

/** Mặt vẽ bằng SVG, xoay 3D theo góc đo được (soi gương như video). */
function HeadModel({ pose }: { pose: HeadPoseResult }) {
  return (
    <div className="head-model" aria-hidden="true">
      <div
        className="head-face"
        style={{ transform: `rotateY(${-pose.yaw}deg) rotateX(${pose.pitch}deg) rotateZ(${-pose.roll}deg)` }}
      >
        <svg width="74" height="90" viewBox="0 0 74 90" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <ellipse cx="37" cy="45" rx="32" ry="41" fill="var(--accent-soft)" />
          <circle cx="25" cy="38" r="3.4" fill="currentColor" stroke="none" />
          <circle cx="49" cy="38" r="3.4" fill="currentColor" stroke="none" />
          <path d="M37 42v12l-4 3" />
          <path d="M27 66c6 4 14 4 20 0" />
        </svg>
      </div>
    </div>
  );
}
