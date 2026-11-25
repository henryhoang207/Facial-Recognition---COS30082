import { useEffect, useState } from "react";
import { api, dateOf } from "../api";
import { useApp } from "../App";
import { Empty, useToast } from "../components/ui";
import type { Person, Settings } from "../types";

const PIPELINE_FIELDS: { key: string; label: string; hint: string; step?: number }[] = [
  { key: "smooth_window", label: "Smoothing window", hint: "Frames averaged per face" },
  { key: "stable_frames", label: "Frames before check-in", hint: "Live and same person in a row" },
  { key: "model_every_n", label: "Run models every", hint: "Frames; raise it on a slow laptop" },
  { key: "max_faces", label: "Faces per frame", hint: "Largest faces first" },
  { key: "spoof_cooldown_s", label: "Spoof log cooldown", hint: "Seconds between logs per face", step: 0.5 },
];

export function SettingsPage() {
  const { health, refresh } = useApp();
  const toast = useToast();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const [reloading, setReloading] = useState(false);

  useEffect(() => {
    api.settings().then((s) => {
      setSettings(s);
      setDraft(s);
    });
    api.persons().then(setPeople);
  }, []);

  if (!draft || !settings) return <Empty>Loading settings…</Empty>;

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);
  const setThreshold = (key: "recognition" | "liveness" | "mask", value: number | null) =>
    setDraft({ ...draft, thresholds: { ...draft.thresholds, [key]: value } });

  const save = async () => {
    const saved = await api.updateSettings({ thresholds: draft.thresholds, pipeline: draft.pipeline, features: draft.features });
    setSettings(saved);
    setDraft(saved);
    refresh();
    toast({ tone: "info", title: "Settings saved", body: "The live view uses them from the next frame." });
  };

  const reload = async () => {
    setReloading(true);
    try {
      await api.reloadModels();
      refresh();
      toast({ tone: "info", title: "Models reloaded", body: "Checkpoints were read again from app/config.yaml." });
    } catch (err) {
      toast({ tone: "spoof", title: "Reload failed", body: err instanceof Error ? err.message : String(err) });
    } finally {
      setReloading(false);
    }
  };

  const remove = async (person: Person) => {
    await api.deletePerson(person.id);
    setPeople((list) => list.filter((p) => p.id !== person.id));
    setConfirmId(null);
    refresh();
    toast({ tone: "info", title: `${person.name} removed`, body: "Their face data and check-ins were deleted." });
  };

  return (
    <div className="stack settings">
      <div className="toolbar">
        <h1 className="page-title">Settings</h1>
        <div className="toolbar-actions">
          <button type="button" className="button button-ghost" disabled={!dirty} onClick={() => setDraft(settings)}>Discard</button>
          <button type="button" className="button button-primary" disabled={!dirty} onClick={save}>Save changes</button>
        </div>
      </div>

      <section className="section">
        <h2 className="section-title">Thresholds</h2>
        <ThresholdField
          label="Recognition"
          hint="Cosine score needed to name a person"
          value={draft.thresholds.recognition}
          fallback={settings.effective.recognition}
          checkpointValue={Number(health?.models.embedder?.checkpoint_threshold ?? settings.effective.recognition)}
          onChange={(v) => setThreshold("recognition", v)}
        />
        <ThresholdField
          label="Liveness"
          hint="Probability of a live face needed to pass"
          value={draft.thresholds.liveness}
          fallback={settings.effective.liveness}
          checkpointValue={Number(health?.models.liveness?.checkpoint_threshold ?? settings.effective.liveness)}
          onChange={(v) => setThreshold("liveness", v)}
        />
        {health?.models.mask && settings.effective.mask !== null && (
          <ThresholdField
            label="Mask (Individual)"
            hint="Probability of a mask needed to show “Mask”"
            value={draft.thresholds.mask}
            fallback={settings.effective.mask}
            checkpointValue={Number(health.models.mask.checkpoint_threshold ?? settings.effective.mask)}
            onChange={(v) => setThreshold("mask", v)}
          />
        )}
      </section>

      <section className="section">
        <h2 className="section-title">Live pipeline</h2>
        <div className="field-grid">
          {PIPELINE_FIELDS.map((f) => (
            <label key={f.key} className="field">
              <span>{f.label}</span>
              <input
                className="input num"
                type="number"
                min={f.step ? 0 : 1}
                step={f.step ?? 1}
                value={draft.pipeline[f.key] ?? ""}
                onChange={(e) => setDraft({ ...draft, pipeline: { ...draft.pipeline, [f.key]: Number(e.target.value) } })}
              />
              <small className="muted">{f.hint}</small>
            </label>
          ))}
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Extra features</h2>
        {Object.keys(draft.features).length === 0 ? (
          <Empty>No feature switches yet. Add them under <code>features</code> in app/config.yaml.</Empty>
        ) : (
          <div className="switch-list">
            {Object.entries(draft.features).map(([key, on]) => (
              <label key={key} className="switch-row">
                <span className="num">{key}</span>
                <input type="checkbox" role="switch" className="switch" checked={on} onChange={(e) => setDraft({ ...draft, features: { ...draft.features, [key]: e.target.checked } })} />
              </label>
            ))}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section-head">
          <h2 className="section-title">Models</h2>
          <button type="button" className="button" onClick={reload} disabled={reloading}>{reloading ? "Reloading…" : "Reload models"}</button>
        </div>
        <table className="table">
          <thead><tr><th>Role</th><th>Type</th><th>Checkpoint</th></tr></thead>
          <tbody>
            {(["detector", "liveness", "embedder", "emotion", "mask", "attendance"] as const).map((role) => {
              const m = health?.models[role] as Record<string, unknown> | null | undefined;
              return (
                <tr key={role}>
                  <td className="strong">{role}</td>
                  <td>{m ? String(m.type ?? "–") : <span className="muted">not loaded</span>}</td>
                  <td className="num muted break">{m ? String(m.path ?? "built-in") : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted small">To swap a model, edit its type and path in app/config.yaml, then reload.</p>
      </section>

      <section className="section">
        <h2 className="section-title">Registered people ({people.length})</h2>
        {people.length === 0 ? (
          <Empty>Nobody is registered yet. Add people from the Register page.</Empty>
        ) : (
          <table className="table">
            <thead><tr><th>Name</th><th>Student ID</th><th>Registered</th><th>Face data</th><th /></tr></thead>
            <tbody>
              {people.map((p) => (
                <tr key={p.id}>
                  <td className="strong">{p.name}</td>
                  <td className="num muted">{p.student_id ?? "–"}</td>
                  <td className="num">{dateOf(p.created_at)}</td>
                  <td>{p.embedding_ok ? <span className="badge badge-live">Ready</span> : <span className="badge badge-warn" title="Saved for a different recognition model">Register again</span>}</td>
                  <td className="right">
                    {confirmId === p.id ? (
                      <span className="confirm">
                        <button type="button" className="button button-danger button-small" onClick={() => remove(p)}>Delete {p.name}</button>
                        <button type="button" className="button button-ghost button-small" onClick={() => setConfirmId(null)}>Keep</button>
                      </span>
                    ) : (
                      <button type="button" className="button button-ghost button-small" onClick={() => setConfirmId(p.id)}>Delete</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

function ThresholdField({
  label,
  hint,
  value,
  fallback,
  checkpointValue,
  onChange,
}: {
  label: string;
  hint: string;
  value: number | null;
  fallback: number;
  checkpointValue: number;
  onChange: (value: number | null) => void;
}) {
  const shown = value ?? checkpointValue ?? fallback;
  return (
    <div className="threshold">
      <div className="threshold-head">
        <span className="strong">{label}</span>
        <span className="num">{shown.toFixed(3)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.005}
        value={shown}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={`${label} threshold`}
      />
      <div className="threshold-foot">
        <small className="muted">{hint}</small>
        {value === null ? (
          <small className="muted">From the checkpoint</small>
        ) : (
          <button type="button" className="link-button" onClick={() => onChange(null)}>
            Use checkpoint value ({checkpointValue.toFixed(3)})
          </button>
        )}
      </div>
    </div>
  );
}
