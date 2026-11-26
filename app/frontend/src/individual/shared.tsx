import { useApp } from "../App";
import type { useCamera } from "../hooks/useCamera";
import type { useFaceStream } from "../hooks/useFaceStream";
import { StreamDot } from "../pages/Live";
import type { IndividualInfo, IndividualMode } from "../types";

export function useFeatureInfo(mode: IndividualMode) {
  const { health } = useApp();
  return { loading: !health, info: health?.models.individual?.[mode] };
}

// tên module Python -> gói pip tương ứng
const PACKAGES: Record<string, string> = { timm: "timm==1.0.24", cv2: "opencv-python==4.13.0.92", yaml: "pyyaml" };

/** Hướng dẫn sửa theo loại lỗi khi model của một tab không load được. */
function fixHint(info: IndividualInfo | undefined, mode: IndividualMode) {
  const error = info?.error ?? "";
  if (!info) return <>Add <code>models.individual.{mode}</code> to <code>app/config.yaml</code>.</>;
  const missing = /No module named '([^']+)'/.exec(error)?.[1];
  if (missing === "mediapipe") {
    return <>Install the hand tracker: <code>pip install -r requirements/hand-gestures.txt</code>, then restart the app.</>;
  }
  if (missing && !missing.startsWith("src.")) {
    // thiếu thư viện Python (không phải thiếu code trong repo)
    const pkg = PACKAGES[missing.split(".")[0]] ?? missing.split(".")[0];
    return <>A Python package is missing. Run <code>pip install {pkg}</code> (or <code>pip install -r requirements.txt</code>) in the app's virtual environment, then restart the app.</>;
  }
  if (missing && info.type === "external") {
    return <>The predictor named in <code>app/config.yaml</code> is not in the repo yet.</>;
  }
  if (error.includes("hand model not found")) {
    return (
      <>
        Download <a className="link" href="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task">hand_landmarker.task</a>{" "}
        into <code>artifacts/hand_gestures/assets/</code>, then reload the models.
      </>
    );
  }
  if (error.includes("FileNotFoundError")) return <>Put the checkpoint at the path in <code>app/config.yaml</code>, then reload the models.</>;
  return null;
}

export function FeatureUnavailable({ mode, info, contract }: { mode: IndividualMode; info?: IndividualInfo; contract?: string }) {
  const { go } = useApp();
  const predictor = info?.type === "external" ? String((info as Record<string, unknown>).predictor ?? "") : "";
  return (
    <div className="notice feature-notice">
      <p className="strong">This feature is not running yet.</p>
      {info?.error && <p className="small mono break">{info.error}</p>}
      <p className="small">{fixHint(info, mode)}</p>
      {contract && (
        <div className="contract">
          <p className="small strong">What the app calls</p>
          {predictor && <code className="block">{predictor}</code>}
          <code className="block">{contract}</code>
        </div>
      )}
      <div className="notice-actions">
        <button type="button" className="button button-small" onClick={() => go("settings")}>
          Open Settings
        </button>
      </div>
    </div>
  );
}

export function StageBar({
  camera,
  stream,
  stages,
}: {
  camera: ReturnType<typeof useCamera>;
  stream: ReturnType<typeof useFaceStream>;
  stages: { key: string; label: string }[];
}) {
  const latency = stream.result?.latency_ms ?? {};
  return (
    <div className="stage-bar">
      <StreamDot status={stream.status} />
      <span className="num">{stream.fps.toFixed(1)} fps</span>
      <div className="pipeline-strip" aria-label="Time per step for the last frame">
        {stages.map((s) => (
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
  );
}

/** Thước ngang từ -range đến +range độ, trái/phải theo góc nhìn soi gương của người dùng. */
export function AngleGauge({
  label,
  value,
  range = 45,
  target,
  mirror = false,
}: {
  label: string;
  value: number | null;
  range?: number;
  target?: number;
  mirror?: boolean;
}) {
  const clamp = (v: number) => Math.max(-range, Math.min(range, v));
  const position = (v: number) => 50 + ((mirror ? -1 : 1) * clamp(v) / range) * 50;
  const zone = target !== undefined ? 50 - (target / range) * 50 : 0;
  return (
    <div className="gauge">
      <div className="meter-head">
        <span>{label}</span>
        <span className="num">{value === null ? "–" : `${value > 0 ? "+" : ""}${value.toFixed(0)}°`}</span>
      </div>
      <div className="gauge-track">
        {target !== undefined && (
          <>
            <span className="gauge-zone" style={{ left: 0, width: `${zone}%` }} />
            <span className="gauge-zone" style={{ right: 0, width: `${zone}%` }} />
          </>
        )}
        <span className="gauge-center" />
        {value !== null && <span className="gauge-needle" style={{ left: `${position(value)}%` }} />}
      </div>
    </div>
  );
}
