import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

/* Thanh xác suất có vạch threshold ------------------------------------------ */

export function Meter({
  label,
  value,
  threshold,
  tone = "accent",
  detail,
}: {
  label: string;
  value: number;
  threshold?: number;
  tone?: "accent" | "warn" | "quiet";
  detail?: string;
}) {
  const percent = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="num">{detail ?? value.toFixed(2)}</span>
      </div>
      <div className="meter-track">
        <div className={`meter-fill tone-${tone}`} style={{ width: `${percent}%` }} />
        {threshold !== undefined && (
          <div
            className="meter-threshold"
            style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
            title={`Threshold ${threshold.toFixed(2)}`}
          />
        )}
      </div>
    </div>
  );
}

/* Tabs với gạch dưới trượt ------------------------------------------------ */

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: { id: T; label: ReactNode }[];
  value: T;
  onChange: (id: T) => void;
  label: string;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });

  useLayoutEffect(() => {
    const active = listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (active) setIndicator({ left: active.offsetLeft, width: active.offsetWidth });
  }, [value, items.length]);

  return (
    <div className="tabs" role="tablist" aria-label={label} ref={listRef}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={item.id === value}
          className="tab"
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
      <span className="tab-indicator" style={{ transform: `translateX(${indicator.left}px)`, width: indicator.width }} />
    </div>
  );
}

/* Toasts ----------------------------------------------------------------------- */

interface Toast {
  id: number;
  title: string;
  body?: string;
  tone: "live" | "spoof" | "info";
}

const ToastContext = createContext<(toast: Omit<Toast, "id">) => void>(() => undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((toast: Omit<Toast, "id">) => {
    const id = Date.now() + Math.random();
    setToasts((list) => [...list.slice(-3), { ...toast, id }]);
    window.setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), 4200);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast tone-${toast.tone}`}>
            <strong>{toast.title}</strong>
            {toast.body && <span>{toast.body}</span>}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);

/* Icon cảm xúc (nét vẽ, không dùng emoji) -------------------------------------- */

const MOUTHS: Record<string, string> = {
  happy: "M8 14.5c1.1 1.4 2.4 2 4 2s2.9-.6 4-2",
  sad: "M8.5 16.5c1-1.2 2.2-1.8 3.5-1.8s2.5.6 3.5 1.8",
  neutral: "M8.5 15.5h7",
  angry: "M8.5 16.5c1-1 2.2-1.4 3.5-1.4s2.5.4 3.5 1.4",
  surprise: "M12 17.2a1.7 2 0 1 0 0-4 1.7 2 0 1 0 0 4z",
  disgust: "M8.5 15.8c1.2-.8 2.3.8 3.5 0s2.3-.8 3.5 0",
  fear: "M8.5 16.2c.9-.9 1.7.9 2.6 0s1.7-.9 2.6 0 1.4.4 1.8 0",
};
const BROWS: Record<string, string> = {
  angry: "M7.5 8.5l3 1.2M16.5 8.5l-3 1.2",
  sad: "M7.5 9.6l3-1M16.5 9.6l-3-1",
  surprise: "M7.6 7.8c1-.6 2-.6 3 0M13.4 7.8c1-.6 2-.6 3 0",
  fear: "M7.6 8.6c1-.8 2-.8 3 0M13.4 8.6c1-.8 2-.8 3 0",
};

export function EmotionIcon({ label, size = 18 }: { label: string; size?: number }) {
  const key = label.toLowerCase();
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="9.2" />
      <path d="M9.2 11h.01M14.8 11h.01" strokeWidth="2.2" />
      {BROWS[key] && <path d={BROWS[key]} />}
      <path d={MOUTHS[key] ?? MOUTHS.neutral} />
    </svg>
  );
}

/* Số chạy lên khi đổi giá trị -------------------------------------------------- */

export function CountUp({ value }: { value: number }) {
  const [shown, setShown] = useState(value);
  const previous = useRef(value);
  useEffect(() => {
    const from = previous.current;
    previous.current = value;
    if (from === value || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(value);
      return;
    }
    const started = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / 500);
      setShown(Math.round(from + (value - from) * (1 - (1 - t) ** 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  return <>{shown}</>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}
