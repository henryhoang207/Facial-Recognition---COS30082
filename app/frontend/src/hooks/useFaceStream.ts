import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api";
import type { Health, Mode, StreamEvent, StreamResult } from "../types";

export type StreamStatus = "connecting" | "live" | "reconnecting" | "paused";

const MIN_INTERVAL_MS = 50; // tối đa ~20 frame/giây gửi lên server

/**
 * Gửi frame từ camera lên /ws/stream và nhận kết quả.
 * Chỉ có 1 frame đang chờ tại một thời điểm, nên không bị dồn trễ khi server chậm.
 */
export function useFaceStream(
  capture: (quality?: number) => Promise<Blob | null>,
  mode: Mode,
  active: boolean,
  onEvent?: (event: StreamEvent) => void,
) {
  const [result, setResult] = useState<StreamResult | null>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [fps, setFps] = useState(0);
  const [ready, setReady] = useState<Partial<Health> | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    setResult(null);
    if (!active) {
      setStatus("paused");
      return;
    }
    let socket: WebSocket | null = null;
    let closed = false;
    let inFlight = false;
    let frameId = 0;
    let lastSent = 0;
    let timer: number | undefined;
    let retry: number | undefined;
    const stamps: number[] = [];

    const send = async () => {
      if (closed || inFlight || socket?.readyState !== WebSocket.OPEN) return;
      const wait = MIN_INTERVAL_MS - (performance.now() - lastSent);
      if (wait > 0) {
        timer = window.setTimeout(send, wait);
        return;
      }
      const blob = await capture(0.72);
      if (!blob || closed || socket?.readyState !== WebSocket.OPEN) {
        timer = window.setTimeout(send, 150);
        return;
      }
      const header = new ArrayBuffer(4);
      new DataView(header).setUint32(0, ++frameId);
      inFlight = true;
      lastSent = performance.now();
      socket.send(await new Blob([header, blob]).arrayBuffer());
    };

    const connect = () => {
      setStatus(frameId ? "reconnecting" : "connecting");
      socket = new WebSocket(wsUrl(mode));
      socket.onmessage = (message) => {
        if (closed) return;
        const data = JSON.parse(message.data);
        if (data.type === "ready") {
          setReady(data);
          setStatus("live");
          inFlight = false;
          send();
        } else if (data.type === "result") {
          inFlight = false;
          setResult(data);
          const now = performance.now();
          stamps.push(now);
          while (stamps.length && now - stamps[0] > 2000) stamps.shift();
          setFps(stamps.length / 2);
          send();
        } else if (data.type === "error") {
          inFlight = false;
          send();
        } else {
          onEventRef.current?.(data);
        }
      };
      socket.onclose = () => {
        if (closed) return;
        inFlight = false;
        setResult(null);
        setStatus("reconnecting");
        retry = window.setTimeout(connect, 1000);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearTimeout(retry);
      socket?.close();
    };
  }, [capture, mode, active]);

  return { result, status, fps, ready };
}
