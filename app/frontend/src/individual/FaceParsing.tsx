import { useState } from "react";
import { CameraView } from "../components/CameraView";
import { Empty } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import type { IndividualInfo } from "../types";
import { FeatureUnavailable, StageBar, useFeatureInfo } from "./shared";

const STAGES = [
  { key: "detect", label: "Detect" },
  { key: "face_parsing", label: "Parsing (background)" },
];

const CONTRACT = "predict_frame(frame_bgr, box) -> (mask HxW uint8 class ids, region (x1, y1, x2, y2))";

export function FaceParsing() {
  const { loading, info } = useFeatureInfo("face_parsing");
  if (loading) return <Empty>Loading…</Empty>;
  if (!info?.available) return <FeatureUnavailable mode="face_parsing" info={info} contract={CONTRACT} />;
  return <FaceParsingLive info={info} />;
}

function FaceParsingLive({ info }: { info: IndividualInfo }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "face_parsing", camera.state === "on");
  const [opacity, setOpacity] = useState(0.85);
  const [picked, setPicked] = useState<number | null>(null);
  const faces = stream.result?.faces ?? [];
  const selected = faces.find((f) => f.track_id === picked) ?? faces[0];
  const parsing = selected?.face_parsing ?? null;
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
          renderOverlay={(scale, [width]) =>
            faces.map((f) => {
              if (!f.face_parsing) return null;
              const [x1, y1, x2, y2] = f.face_parsing.region;
              return (
                <img
                  key={f.track_id}
                  className="parsing-overlay"
                  src={f.face_parsing.overlay}
                  alt=""
                  style={{
                    left: (width - x2) * scale,
                    top: y1 * scale,
                    width: (x2 - x1) * scale,
                    height: (y2 - y1) * scale,
                    opacity,
                  }}
                />
              );
            })
          }
        />
        <StageBar camera={camera} stream={stream} stages={STAGES} />
      </section>

      <aside className="side analysis" aria-label="Face parts">
        {stream.result?.face_parsing?.error && (
          <div className="notice" role="alert">
            <p className="strong">The model failed on the last frames.</p>
            <p className="small mono break">{stream.result.face_parsing.error}</p>
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
        <div className="analysis-card">
          <div className="model-block">
            <div className="model-head">
              <h3>Face parts</h3>
              <span className="muted small">share of the face</span>
            </div>
            {!parsing ? (
              <p className="muted small">{faces.length ? "Analysing the face…" : "No face in view."}</p>
            ) : (
              <ul className="parts">
                {[...parsing.parts].sort((a, b) => b.share - a.share).map((part) => (
                  <li key={part.name}>
                    <span className="swatch" style={{ background: part.color }} />
                    <span className="part-name">{part.name}</span>
                    <span className="part-bar"><span style={{ width: `${Math.min(100, part.share * 100)}%`, background: part.color }} /></span>
                    <span className="num small">{(part.share * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="model-block">
            <label className="field">
              <span className="small">Overlay strength</span>
              <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
          </div>
        </div>
        <div className="analysis-card">
          <div className="model-block">
            <div className="model-head"><h3>{mock ? "Rough regions from 5 detector points" : "Mạnh Đức's face parsing model"}</h3></div>
            <p className="muted small">
              {mock
                ? "Mock mode: ellipses around YuNet's eye, nose and mouth points, so the page can be tried without the model."
                : `BiSeNet-style network on ResNet-18, 11 face classes (LaPa layout), 512 px crop around the face. Runs on ${String(info.device ?? "cpu").toUpperCase()} in ${String(info.precision ?? "fp32")}, on its own thread: the camera keeps its frame rate and the colours follow the face between model results.`}
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}
