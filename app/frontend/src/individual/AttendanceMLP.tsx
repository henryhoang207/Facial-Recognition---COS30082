import { useState } from "react";
import { useApp } from "../App";
import { CameraView } from "../components/CameraView";
import { Empty, Meter } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import { StreamDot } from "../pages/Live";
import type { Face } from "../types";

function verdict(face: Face) {
  if (face.liveness?.label === "spoof") {
    return { label: "Spoof blocked", tone: "spoof" } as const;
  }
  const result = face.attendance_mlp;
  if (result?.state === "accepted" && result.candidate_name) {
    return { label: result.candidate_name, tone: "live" } as const;
  }
  return { label: result?.state === "rejected" ? "Unknown" : "Unverified", tone: "neutral" } as const;
}

export function AttendanceMLP() {
  const { health } = useApp();
  if (!health) return <Empty>Loading models…</Empty>;
  if (!health.models.attendance) {
    return (
      <div className="notice">
        <p className="strong">Attendance MLP is unavailable.</p>
        <p className="small">Load a matching Model A and the checkpoint configured under <code>models.attendance</code>.</p>
      </div>
    );
  }
  return <AttendanceCamera gallerySize={health.gallery_size} />;
}

function AttendanceCamera({ gallerySize }: { gallerySize: number }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "attendance_mlp", camera.state === "on");
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const faces = stream.result?.faces ?? [];
  const selected = faces.find((face) => face.track_id === selectedTrack) ?? faces[0];
  const result = selected?.attendance_mlp;
  const count = result?.gallery_size ?? stream.ready?.gallery_size ?? gallerySize;
  const selectedVerdict = selected && verdict(selected);
  return (
    <div className="live-layout">
      <section className="stage" aria-label="Attendance camera">
        <CameraView
          videoRef={camera.videoRef}
          state={camera.state}
          faces={faces}
          frameSize={stream.result?.frame_size}
          selectedTrack={faces.length > 1 ? selected?.track_id : undefined}
          onSelect={faces.length > 1 ? setSelectedTrack : undefined}
          tone={(face) => verdict(face).tone}
          renderLabel={(face) => {
            const { label, tone } = verdict(face);
            return <span className={tone === "neutral" ? "tag" : `tag tag-${tone}`}>{label}</span>;
          }}
        />
        <div className="stage-bar">
          <StreamDot status={stream.status} />
          <span className="num">{stream.fps.toFixed(1)} fps</span>
          {camera.devices.length > 1 && (
            <select className="select" aria-label="Camera" value={camera.deviceId} onChange={(e) => camera.setDeviceId(e.target.value)}>
              {camera.devices.map((device, i) => (
                <option key={device.deviceId} value={device.deviceId}>{device.label || `Camera ${i + 1}`}</option>
              ))}
            </select>
          )}
        </div>
      </section>
      <aside className="side analysis" aria-label="Attendance MLP results">
        {faces.length > 1 && (
          <div className="face-picker" role="tablist" aria-label="Face">
            {faces.map((face, index) => (
              <button key={face.track_id} type="button" role="tab" className="face-pick"
                aria-selected={selected?.track_id === face.track_id}
                onClick={() => setSelectedTrack(face.track_id)}>Face {index + 1}</button>
            ))}
          </div>
        )}
        <div className="analysis-card">
          <header className="analysis-head">
            <div>
              <p className="analysis-name">{selectedVerdict?.label ?? "No face in view"}</p>
              <p className="muted small">{count} registered</p>
            </div>
            {selected?.logged && <span className="badge badge-live">Checked in</span>}
          </header>
          <section className="model-block">
            {selected?.liveness && <Meter label="Liveness" value={selected.liveness.raw} threshold={selected.liveness.threshold} />}
            {count === 0 ? (
              <p className="small">Add someone in <a className="link" href="#/register">Register</a> to start.</p>
            ) : selectedVerdict?.tone === "spoof" ? (
              <p className="small">Liveness check failed.</p>
            ) : result?.state === "needs_gallery" ? (
              <p className="small">This checkpoint requires {result.required_gallery_size} registered people.</p>
            ) : result?.score != null ? (
              <Meter label="MLP confidence" value={result.score} threshold={result.threshold} tone={result.state === "accepted" ? "accent" : "warn"} />
            ) : result ? (
              <p className="small">Live frames collected: {result.frames}/{result.required_frames}</p>
            ) : (
              <p className="small">Look at the camera to check in.</p>
            )}
          </section>
          <section className="model-block">
            <a className="link" href="#/attendance">View attendance</a>
          </section>
        </div>
      </aside>
    </div>
  );
}
