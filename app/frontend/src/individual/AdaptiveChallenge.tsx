import { useState } from "react";
import { CameraView } from "../components/CameraView";
import { Empty, Meter } from "../components/ui";
import { useCamera } from "../hooks/useCamera";
import { useFaceStream } from "../hooks/useFaceStream";
import type { ChallengeResult, Face, IndividualInfo } from "../types";
import { AngleGauge, FeatureUnavailable, StageBar, useFeatureInfo } from "./shared";

const STAGES = [
  { key: "detect", label: "Detect" },
  { key: "adaptive_liveness", label: "Liveness" },
  { key: "embed", label: "Recognise" },
];

const SIDE = { left: "left", right: "right" } as const;

function tone(c?: ChallengeResult | null): "live" | "spoof" | "neutral" {
  if (c?.state === "passed") return "live";
  if (c?.state === "failed") return "spoof";
  return "neutral";
}

function tagText(c: ChallengeResult) {
  if (c.state === "checking") return "Checking…";
  if (c.state === "challenge") return c.step === "left" ? "← Turn left" : "Turn right →";
  if (c.state === "failed") return "Not verified";
  return c.identity?.name ?? "Verified";
}

export function AdaptiveChallenge() {
  const { loading, info } = useFeatureInfo("challenge");
  if (loading) return <Empty>Loading…</Empty>;
  if (!info?.available) return <FeatureUnavailable mode="challenge" info={info} />;
  return <ChallengeLive info={info} />;
}

function ChallengeLive({ info }: { info: IndividualInfo }) {
  const camera = useCamera();
  const stream = useFaceStream(camera.capture, "challenge", camera.state === "on");
  const [picked, setPicked] = useState<number | null>(null);
  const faces = stream.result?.faces ?? [];
  const selected = faces.find((f) => f.track_id === picked) ?? faces[0];
  const current = selected?.challenge ?? null;

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
          tone={(f) => tone(f.challenge)}
          renderLabel={(f) =>
            f.challenge && (
              <span className={tone(f.challenge) === "neutral" ? "tag" : `tag tag-${tone(f.challenge)}`}>{tagText(f.challenge)}</span>
            )
          }
        >
          {current?.state === "challenge" && current.step && <TurnPrompt challenge={current} />}
        </CameraView>
        <StageBar camera={camera} stream={stream} stages={STAGES} />
      </section>

      <aside className="side analysis" aria-label="Liveness check">
        {faces.length > 1 && (
          <div className="face-picker" role="tablist" aria-label="Face">
            {faces.map((f, i) => (
              <button key={f.track_id} type="button" role="tab" aria-selected={f.track_id === selected?.track_id}
                className={`face-pick tone-${tone(f.challenge)}`} onClick={() => setPicked(f.track_id)}>
                Face {i + 1}
              </button>
            ))}
          </div>
        )}
        {!current ? (
          <div className="analysis-empty">
            <Empty>No face in view. Look at the camera to start the check.</Empty>
          </div>
        ) : (
          <StatusCard key={selected!.track_id} face={selected!} challenge={current} />
        )}
        <ModelNote info={info} />
      </aside>
    </div>
  );
}

function TurnPrompt({ challenge }: { challenge: ChallengeResult }) {
  const left = challenge.step === "left";
  const total = 10;
  return (
    <div className={`turn-prompt ${left ? "is-left" : "is-right"}`} key={`${challenge.step_index}`} role="status">
      <svg className="turn-arrow" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {left ? <path d="M15 5l-7 7 7 7" /> : <path d="M9 5l7 7-7 7" />}
      </svg>
      <div>
        <p className="turn-title">Turn your head to the {SIDE[challenge.step!]}</p>
        <p className="turn-sub num">
          Step {challenge.step_index + 1} of {challenge.steps.length} · {Math.ceil(challenge.seconds_left ?? 0)} s left
        </p>
      </div>
      <span className="turn-timer" style={{ width: `${((challenge.seconds_left ?? 0) / total) * 100}%` }} />
    </div>
  );
}

function StatusCard({ face, challenge: c }: { face: Face; challenge: ChallengeResult }) {
  const headline = {
    checking: ["Checking liveness", "Hold still while the model looks at a few frames."],
    challenge: ["Prove it is a live face", "The model is not sure. Turn your head as asked: a printed photo or a still screen cannot."],
    passed:
      c.reason === "live"
        ? ["Live face", "Clearly live, so no challenge was needed."]
        : ["Verified by head turn", "The doubtful face followed both turns in time."],
    failed: ["Not verified", `The turns were not completed in time. Retrying in ${Math.ceil(c.retry_in ?? 0)} s.`],
  }[c.state];
  const verdictTone = c.state === "passed" ? "ok" : c.state === "failed" ? "warn" : "";

  return (
    <div className={`analysis-card challenge-card is-${c.state}`}>
      <div className="model-block">
        <p className={`verdict ${verdictTone}`}>{headline[0]}</p>
        <p className="muted small">{headline[1]}</p>
        {c.state === "checking" && (
          <Meter label="Frames checked" value={c.frames / c.required_frames} detail={`${c.frames}/${c.required_frames}`} tone="quiet" />
        )}
        <Meter label="Live probability" value={c.live_prob} threshold={c.threshold} tone={c.live_prob >= c.threshold ? "accent" : "warn"} detail={`${(c.live_prob * 100).toFixed(1)}%`} />
      </div>

      {c.steps.length > 0 && (
        <div className="model-block">
          <ol className="challenge-steps">
            {c.steps.map((step, i) => {
              const done = c.state === "passed" || i < c.step_index;
              const active = c.state === "challenge" && i === c.step_index;
              return (
                <li key={step} className={done ? "done" : active ? "active" : ""}>
                  <span className="step-mark" aria-hidden="true">{done ? "✓" : i + 1}</span>
                  Turn {step}
                  {active && <span className="hold num">{"●".repeat(c.hold)}{"○".repeat(Math.max(0, c.hold_frames - c.hold))}</span>}
                </li>
              );
            })}
          </ol>
          <AngleGauge label="Head turn" value={c.yaw} target={c.turn_deg} mirror />
          {c.yaw === null && <p className="small warn">Face landmarks are not available from the detector.</p>}
        </div>
      )}

      <div className="model-block">
        <div className="identity-row">
          <span className="muted small">Identity</span>
          {c.state !== "passed" ? (
            <span className="muted small">After the check</span>
          ) : face.challenge?.identity?.known ? (
            <span className="strong">
              {c.identity!.name} <span className="muted num small">{c.identity!.score.toFixed(2)}</span>
            </span>
          ) : (
            <span className="strong">Not registered</span>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelNote({ info }: { info: IndividualInfo }) {
  const acer = typeof info.val_acer === "number" ? info.val_acer : null;
  const auc = typeof info.val_auc === "number" ? info.val_auc : null;
  return (
    <div className="analysis-card">
      <div className="model-block">
        <div className="model-head">
          <h3>MobileNetV3-Small, distilled</h3>
          {info.path && <span className="model-file" title={info.path}>{info.path.split("/").pop()}</span>}
        </div>
        <dl className="result-grid">
          <div><dt>Threshold</dt><dd className="num">{Number(info.threshold).toFixed(3)}</dd></div>
          {acer !== null && <div><dt>Val ACER</dt><dd className="num">{(acer * 100).toFixed(2)}%</dd></div>}
          {auc !== null && <div><dt>Val AUC</dt><dd className="num">{auc.toFixed(4)}</dd></div>}
          <div><dt>Turn needed</dt><dd className="num">±{Number(info.turn_deg)}°</dd></div>
        </dl>
        <p className="muted small">
          Faces under the threshold get a left / right challenge with {Number(info.challenge_seconds)} s to finish. The turn
          is measured from the detector's eye and nose points. Nothing is written to attendance here.
        </p>
      </div>
    </div>
  );
}
