import { type FormEvent, useCallback, useEffect, useState } from "react";
import { api, dateOf, timeOf } from "../api";
import { useApp } from "../App";
import { CountUp, EmotionIcon, Empty, Tabs, useToast } from "../components/ui";
import type { AttendanceReport, Session, SpoofRow } from "../types";

type View = "present" | "absent" | "blocked";

export function Attendance() {
  const { session: active, refresh } = useApp();
  const toast = useToast();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [report, setReport] = useState<AttendanceReport | null>(null);
  const [spoofs, setSpoofs] = useState<SpoofRow[]>([]);
  const [view, setView] = useState<View>("present");
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const loadSessions = useCallback(() => {
    api.sessions().then((list) => {
      setSessions(list);
      setSelected((current) => current ?? list.find((s) => s.active)?.id ?? list[0]?.id ?? null);
    });
  }, []);
  useEffect(loadSessions, [loadSessions]);

  const load = useCallback(() => {
    if (selected === null) return;
    api.attendance(selected).then(setReport).catch(() => undefined);
    api.spoofEvents(selected).then(setSpoofs).catch(() => undefined);
  }, [selected]);

  useEffect(() => {
    load();
    const current = sessions.find((s) => s.id === selected);
    if (!current?.active) return;
    // buổi đang mở: cập nhật mỗi 5 giây
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, [load, selected, sessions]);

  const startSession = async (event: FormEvent) => {
    event.preventDefault();
    const created = await api.startSession(newName.trim() || `Session ${new Date().toLocaleDateString()}`);
    toast({ tone: "info", title: `${created.name} started`, body: "New check-ins go to this session." });
    setNewName("");
    setCreating(false);
    setSelected(created.id);
    loadSessions();
    refresh();
  };

  const endSession = async (id: number) => {
    await api.endSession(id);
    toast({ tone: "info", title: "Session ended", body: "A new session starts with the next check-in." });
    loadSessions();
    refresh();
  };

  const current = sessions.find((s) => s.id === selected);
  const stats = report?.stats;
  const topEmotion = stats
    ? Object.entries(stats.emotions).sort((a, b) => b[1] - a[1])[0]?.[0]
    : undefined;

  return (
    <div className="stack">
      <div className="toolbar">
        <h1 className="page-title">Attendance</h1>
        <div className="toolbar-actions">
          <select
            className="select"
            aria-label="Session"
            value={selected ?? ""}
            onChange={(e) => setSelected(Number(e.target.value))}
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}, {dateOf(s.started_at)}{s.active ? " (open)" : ""}
              </option>
            ))}
          </select>
          {current?.active && (
            <button type="button" className="button" onClick={() => endSession(current.id)}>End session</button>
          )}
          {creating ? (
            <form className="inline-form" onSubmit={startSession}>
              <input className="input" placeholder="Session name" value={newName} onChange={(e) => setNewName(e.target.value)} autoFocus aria-label="New session name" />
              <button type="submit" className="button button-primary">Start</button>
              <button type="button" className="button button-ghost" onClick={() => setCreating(false)}>Cancel</button>
            </form>
          ) : (
            <button type="button" className="button" onClick={() => setCreating(true)}>New session</button>
          )}
          {selected !== null && (
            <a className="button button-primary" href={api.exportUrl(selected)} download>Export CSV</a>
          )}
        </div>
      </div>

      {stats && (
        <div className="stats">
          <div className="stat">
            <span className="stat-value num"><CountUp value={stats.present} /><span className="muted"> / {stats.registered}</span></span>
            <span className="stat-label">present</span>
          </div>
          <div className="stat">
            <span className="stat-value num"><CountUp value={report.absent.length} /></span>
            <span className="stat-label">absent</span>
          </div>
          <div className="stat stat-warn">
            <span className="stat-value num"><CountUp value={stats.spoof_blocked} /></span>
            <span className="stat-label">spoof attempts blocked</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              {topEmotion ? <><EmotionIcon label={topEmotion} size={26} /> {topEmotion}</> : "–"}
            </span>
            <span className="stat-label">most common mood</span>
          </div>
        </div>
      )}

      <Tabs<View>
        label="Attendance view"
        value={view}
        onChange={setView}
        items={[
          { id: "present", label: `Present (${report?.present.length ?? 0})` },
          { id: "absent", label: `Absent (${report?.absent.length ?? 0})` },
          { id: "blocked", label: `Blocked (${spoofs.length})` },
        ]}
      />

      <div className="table-wrap" key={`${view}-${selected}`}>
        {view === "present" &&
          (report?.present.length ? (
            <table className="table">
              <thead>
                <tr><th>Name</th><th>Student ID</th><th>Checked in</th><th className="right">Live</th><th className="right">Match</th><th>Mood</th></tr>
              </thead>
              <tbody>
                {report.present.map((row) => (
                  <tr key={row.person_id}>
                    <td className="strong">{row.name}</td>
                    <td className="num muted">{row.student_id ?? "–"}</td>
                    <td className="num">{timeOf(row.first_seen)}</td>
                    <td className="num right">{row.live_prob.toFixed(2)}</td>
                    <td className="num right">{row.match_score.toFixed(2)}</td>
                    <td>{row.emotion ? <span className="emotion-chip"><EmotionIcon label={row.emotion} size={15} />{row.emotion}</span> : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>No check-ins yet. People are checked in from the Live page.</Empty>
          ))}

        {view === "absent" &&
          (report?.absent.length ? (
            <table className="table">
              <thead><tr><th>Name</th><th>Student ID</th></tr></thead>
              <tbody>
                {report.absent.map((row) => (
                  <tr key={row.person_id}><td className="strong">{row.name}</td><td className="num muted">{row.student_id ?? "–"}</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>Everyone registered is present.</Empty>
          ))}

        {view === "blocked" &&
          (spoofs.length ? (
            <table className="table">
              <thead><tr><th>Time</th><th>Snapshot</th><th className="right">Live</th><th>Result</th></tr></thead>
              <tbody>
                {spoofs.map((row) => (
                  <tr key={row.id}>
                    <td className="num">{timeOf(row.ts)}</td>
                    <td>
                      {row.snapshot_path ? (
                        <img className="snapshot" src={api.snapshotUrl(row.snapshot_path)} alt="Blocked face" loading="lazy" />
                      ) : (
                        <span className="snapshot snapshot-empty" />
                      )}
                    </td>
                    <td className="num right warn">{row.live_prob.toFixed(2)}</td>
                    <td className="muted">Recognition skipped, not logged</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>No spoof attempts in this session.</Empty>
          ))}
      </div>
      {active && current && !current.active && (
        <p className="muted small">You are viewing a closed session. The open session is {active.name}.</p>
      )}
    </div>
  );
}
