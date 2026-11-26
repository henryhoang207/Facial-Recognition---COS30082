import { useState } from "react";
import { useApp } from "../App";
import { Tabs } from "../components/ui";
import { MEMBERS, type Member } from "../members";
import type { Health } from "../types";

const STORAGE_KEY = "faceguard.individual.member";

function isRunning(member: Member, health: Health | null) {
  const model = member.feature?.model;
  if (!model || !health) return false;
  if (model === "mask" || model === "attendance") return Boolean(health.models[model]);
  return Boolean(health.models.individual?.[model]?.available);
}

function savedMember() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

const initials = (name: string) =>
  name
    .split(" ")
    .slice(-2)
    .map((part) => part[0])
    .join("");

export function Individual() {
  const { health } = useApp();
  // nhớ tab đã mở lần trước, mặc định là tab Nam Khánh
  const [memberId, setMemberId] = useState(() => savedMember() ?? "namkhanh");
  const choose = (id: string) => {
    setMemberId(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      // trình duyệt chặn lưu trữ: chỉ không nhớ được tab
    }
  };
  const member = MEMBERS.find((m) => m.id === memberId) ?? MEMBERS[0];
  const Feature = member.feature?.component;

  return (
    <div className="stack individual">
      <div>
        <h1 className="page-title">Individual features</h1>
        <p className="muted lead-gap">Each member's own extension of FaceGuard, on the same camera and face detector.</p>
      </div>

      <div className="member-tabs">
        <Tabs
          label="Team member"
          value={member.id}
          onChange={choose}
          items={MEMBERS.map((m) => ({
            id: m.id,
            label: (
              <span className="member-tab">
                {m.short}
                {m.feature && (
                  <span
                    className={`member-dot ${isRunning(m, health) ? "is-on" : ""}`}
                    title={isRunning(m, health) ? "Model loaded" : "Model not loaded"}
                  />
                )}
              </span>
            ),
          }))}
        />
      </div>

      <section className="member-panel" key={member.id} aria-label={member.name}>
        <header className="member-head">
          <span className="avatar" aria-hidden="true">{initials(member.name)}</span>
          <div className="member-meta">
            <p className="member-name">{member.name}</p>
            <p className="muted small">Core part: {member.core}</p>
          </div>
          {member.feature && (
            <div className="member-feature">
              <p className="feature-title">{member.feature.title}</p>
              <p className="muted small">{member.feature.summary}</p>
            </div>
          )}
        </header>

        {Feature ? (
          <Feature />
        ) : (
          <div className="feature-empty">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
              <rect x="3.5" y="3.5" width="17" height="17" rx="3" />
              <path d="M12 8.5v7M8.5 12h7" />
            </svg>
            <p className="strong">No individual feature here yet</p>
            <p className="muted small">
              A feature appears in this tab once it is added in <code>app/frontend/src/members.ts</code>.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
