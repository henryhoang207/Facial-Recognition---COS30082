import { useState } from "react";
import { useApp } from "../App";
import { CameraView } from "../components/CameraView";
import { Empty, Meter } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import { StreamDot } from "../pages/Live";
import type { Face } from "../types";

interface MaskInfo {
  type: string;
  path: string | null;
  arch: string;
  checkpoint_threshold: number;
  img_size: number;
  crop_scale: number;
  detector_score: number | null;
}

const ARCH_NAMES: Record<string, string> = {
  mobilenetv3s: "MobileNetV3-Small",
  mobilenetv3l: "MobileNetV3-Large",
  mobilenetv4m: "MobileNetV4-Conv-Medium",
  dummy: "Dummy model (mock mode)",
};

// Kết quả trên FMLD test (12 267 mặt), lấy từ reports/figures/mask_results.csv
const TEST_RESULTS: Record<string, { accuracy: number; f1: number; maskRecall: number; noMaskRecall: number; cpuMs: number }> = {
  mobilenetv3s: { accuracy: 0.9786, f1: 0.9779, maskRecall: 0.9808, noMaskRecall: 0.9772, cpuMs: 11.4 },
  mobilenetv3l: { accuracy: 0.9653, f1: 0.9644, maskRecall: 0.9908, noMaskRecall: 0.9478, cpuMs: 22.1 },
};

const STAGES = [
  { key: "detect", label: "Detect" },
  { key: "mask", label: "Mask model" },
];

const isMasked = (f: Face) => f.mask?.label === "mask";
const percent = (value: number) => `${Math.round(value * 100)}%`;

export function MaskDetection() {
  const { health, go } = useApp();
  if (!health) return <Empty>Loading…</Empty>;
  const info = health.models.mask as unknown as MaskInfo | null | undefined;
  if (!info) {
    return (
      <div className="notice">
        <p className="strong">The mask model is not loaded.</p>
        <p className="small">
          Put the checkpoint at the path set under <code>models.mask</code> in <code>app/config.yaml</code>{" "}
          (default <code>model/Mask/mask_mobilenetv3s.pt</code>), then reload the models.
        </p>
        <div className="notice-actions">
          <button type="button" className="button button-small" onClick={() => go("settings")}>
            Open Settings
          </button>
        </div>
      </div>
    );
  }
  return <MaskLive info={info} />;
}

function MaskLive({ info }: { info: MaskInfo }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "mask", camera.state === "on");
  const [pickedTrack, setPickedTrack] = useState<number | null>(null);
  const faces = stream.result?.faces ?? [];
  const latency = stream.result?.latency_ms ?? {};
  const selected = faces.find((f) => f.track_id === pickedTrack) ?? faces[0];
  const masked = faces.filter(isMasked).length;

  return (
    <div className="live-layout">
      <section className="stage" aria-label="Camera">
        <CameraView
          videoRef={camera.videoRef}
          state={camera.state}
          faces={faces}
          frameSize={stream.result?.frame_size}
          selectedTrack={faces.length > 1 ? selected?.track_id : undefined}
          onSelect={setPickedTrack}
          tone={(f) => (!f.mask ? "neutral" : isMasked(f) ? "live" : "spoof")}
          renderLabel={(f) =>
            f.mask && (
              <span className={isMasked(f) ? "tag tag-live" : "tag tag-spoof"}>
                {isMasked(f) ? "Mask" : "No mask"}
                <span className="num tag-num">{percent(isMasked(f) ? f.mask.prob : 1 - f.mask.prob)}</span>
              </span>
            )
          }
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

      <aside className="side analysis" aria-label="Mask results">
        <div className="mask-counts" aria-live="polite">
          <div className="mask-count">
            <span className="num mask-count-value">{faces.length}</span>
            <span className="muted small">In view</span>
          </div>
          <div className="mask-count is-mask">
            <span className="num mask-count-value">{masked}</span>
            <span className="muted small">Mask</span>
          </div>
          <div className="mask-count is-no-mask">
            <span className="num mask-count-value">{faces.length - masked}</span>
            <span className="muted small">No mask</span>
          </div>
        </div>

        {faces.length > 1 && (
          <div className="face-picker" role="tablist" aria-label="Face">
            {faces.map((f, i) => (
              <button
                key={f.track_id}
                type="button"
                role="tab"
                aria-selected={f.track_id === selected?.track_id}
                className={`face-pick tone-${isMasked(f) ? "live" : "spoof"}`}
                onClick={() => setPickedTrack(f.track_id)}
              >
                Face {i + 1}
              </button>
            ))}
          </div>
        )}

        {!selected?.mask ? (
          <div className="analysis-empty">
            <Empty>No face in view. Step in front of the camera, then try with and without a mask.</Empty>
          </div>
        ) : (
          <MaskCard key={selected.track_id} face={selected} />
        )}

        <ModelCard info={info} />
      </aside>
    </div>
  );
}

function MaskCard({ face }: { face: Face }) {
  const mask = face.mask!;
  const on = mask.label === "mask";
  return (
    <div className={`analysis-card mask-card ${on ? "is-mask" : "is-no-mask"}`}>
      <div className="model-block">
        <div className="mask-verdict">
          <MaskIcon on={on} />
          <div>
            <p className={`verdict ${on ? "ok" : "warn"}`}>{on ? "Wearing a mask" : "No mask"}</p>
            <p className="muted small">
              {on ? "Mouth and nose are covered." : "The lower face is uncovered."} Threshold {mask.threshold.toFixed(2)}.
            </p>
          </div>
        </div>
        <Meter label="Mask probability, this frame" value={mask.raw} threshold={mask.threshold} tone={mask.raw >= mask.threshold ? "accent" : "warn"} detail={percent(mask.raw)} />
        <Meter label="Smoothed over recent frames" value={mask.prob} threshold={mask.threshold} tone={on ? "accent" : "warn"} detail={percent(mask.prob)} />
      </div>
    </div>
  );
}

function ModelCard({ info }: { info: MaskInfo }) {
  const results = TEST_RESULTS[info.arch];
  return (
    <div className="analysis-card">
      <div className="model-block">
        <div className="model-head">
          <h3>{ARCH_NAMES[info.arch] ?? info.arch}</h3>
          {info.path && <span className="model-file" title={info.path}>{info.path.split("/").pop()}</span>}
        </div>
        <p className="muted small">
          ImageNet weights fine-tuned on FMLD (MAFA + WIDER FACE). Face crop {info.img_size} px at scale {info.crop_scale}
          {info.detector_score !== null && <>; detector score lowered to {info.detector_score} here so masked faces are found</>}.
        </p>
        {results && (
          <dl className="result-grid">
            <div><dt>Test accuracy</dt><dd className="num">{(results.accuracy * 100).toFixed(1)}%</dd></div>
            <div><dt>Macro F1</dt><dd className="num">{results.f1.toFixed(3)}</dd></div>
            <div><dt>Mask recall</dt><dd className="num">{(results.maskRecall * 100).toFixed(1)}%</dd></div>
            <div><dt>No-mask recall</dt><dd className="num">{(results.noMaskRecall * 100).toFixed(1)}%</dd></div>
          </dl>
        )}
        {results && <p className="muted small">FMLD test set, 12,267 faces; {results.cpuMs} ms per face on CPU.</p>}
      </div>
    </div>
  );
}

function MaskIcon({ on }: { on: boolean }) {
  return (
    <svg className="mask-icon" width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <ellipse cx="20" cy="19" rx="11" ry="13.5" />
      <path d="M15.5 16h.01M24.5 16h.01" strokeWidth="2.6" />
      {on ? (
        <>
          <path className="mask-shape" d="M9.5 20.5c3.5-1.6 7-2.4 10.5-2.4s7 .8 10.5 2.4c-.4 5.6-4.6 9.5-10.5 9.5s-10.1-3.9-10.5-9.5z" />
          <path d="M13.5 23.5h13M14.5 26.5h11" strokeWidth="1.2" />
        </>
      ) : (
        <path d="M16 25c1.2 1 2.5 1.5 4 1.5s2.8-.5 4-1.5" />
      )}
    </svg>
  );
}
