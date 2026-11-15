import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api } from "./api";
import { Attendance } from "./pages/Attendance";
import { Live } from "./pages/Live";
import { Register } from "./pages/Register";
import { SettingsPage } from "./pages/Settings";
import type { Health, Session } from "./types";

const PAGES = [
  { id: "live", label: "Live" },
  { id: "register", label: "Register" },
  { id: "attendance", label: "Attendance" },
  { id: "settings", label: "Settings" },
] as const;

export type PageId = (typeof PAGES)[number]["id"];

interface AppState {
  health: Health | null;
  session: Session | null;
  offline: boolean;
  refresh: () => void;
  go: (page: PageId) => void;
}

const AppContext = createContext<AppState>({
  health: null,
  session: null,
  offline: false,
  refresh: () => undefined,
  go: () => undefined,
});
export const useApp = () => useContext(AppContext);

const pageFromHash = (): PageId => {
  const id = location.hash.replace("#/", "");
  return (PAGES.find((p) => p.id === id)?.id ?? "live") as PageId;
};

export default function App() {
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(() => {
    Promise.all([api.health(), api.sessions()])
      .then(([h, sessions]) => {
        setHealth(h);
        setSession(sessions.find((s) => s.active) ?? null);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    refresh();
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [refresh]);

  useEffect(() => {
    if (!offline) return;
    const timer = window.setInterval(refresh, 3000);
    return () => window.clearInterval(timer);
  }, [offline, refresh]);

  const go = useCallback((id: PageId) => {
    location.hash = `#/${id}`;
  }, []);

  return (
    <AppContext.Provider value={{ health, session, offline, refresh, go }}>
      <header className="topbar">
        <a className="brand" href="#/live" aria-label="FaceGuard, live view">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />
            <circle cx="12" cy="11" r="2.6" />
            <path d="M7.8 17c1.1-1.6 2.5-2.4 4.2-2.4s3.1.8 4.2 2.4" />
          </svg>
          FaceGuard
        </a>
        <nav className="nav" aria-label="Main">
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className="nav-link" aria-current={page === p.id ? "page" : undefined}>
              {p.label}
            </a>
          ))}
        </nav>
        <div className="topbar-status">
          {health?.mock_mode && <span className="badge badge-warn">Mock models</span>}
          {session && <span className="session-name" title="Active session">{session.name}</span>}
        </div>
      </header>
      {offline && (
        <div className="banner" role="alert">
          Can't reach the server. Start it with <code>python -m app.run</code> from the repo root; this page reconnects on its own.
        </div>
      )}
      <main className="page" key={page}>
        {page === "live" && <Live />}
        {page === "register" && <Register />}
        {page === "attendance" && <Attendance />}
        {page === "settings" && <SettingsPage />}
      </main>
    </AppContext.Provider>
  );
}
