import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react";
import type { CameraState } from "../hooks/useCamera";
import type { Face } from "../types";

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  state: CameraState;
  faces?: Face[];
  frameSize?: [number, number];
  renderLabel?: (face: Face) => ReactNode;
  tone?: (face: Face) => "live" | "spoof" | "neutral";
  flashKey?: number;
  selectedTrack?: number;
  onSelect?: (trackId: number) => void;
  /** Vẽ thêm lên khung hình (toạ độ frame nhân `scale`, trục x đã lật như video). */
  renderOverlay?: (scale: number, frameSize: [number, number]) => ReactNode;
  children?: ReactNode;
}

const CAMERA_MESSAGES: Record<CameraState, string | null> = {
  starting: "Starting camera…",
  on: null,
  off: "Camera is off",
  denied: "Camera access is blocked. Allow it in the browser's address bar, then reload.",
  missing: "No camera found. Plug one in, or open the app on http://localhost.",
};

/** Video soi gương + các khung bám theo mặt, toạ độ quy đổi từ frame gửi lên server. */
export function CameraView({
  videoRef,
  state,
  faces = [],
  frameSize,
  renderLabel,
  tone = () => "neutral",
  flashKey = 0,
  selectedTrack,
  onSelect,
  renderOverlay,
  children,
}: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [aspect, setAspect] = useState(4 / 3);

  useEffect(() => {
    const element = boxRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => video.videoWidth && setAspect(video.videoWidth / video.videoHeight);
    video.addEventListener("loadedmetadata", update);
    update();
    return () => video.removeEventListener("loadedmetadata", update);
  }, [videoRef]);

  const scale = frameSize ? width / frameSize[0] : 0;

  return (
    <div className="camera" ref={boxRef} style={{ aspectRatio: String(aspect) }}>
      <video ref={videoRef} className="camera-video" muted playsInline />
      {CAMERA_MESSAGES[state] && <div className="camera-message">{CAMERA_MESSAGES[state]}</div>}
      {flashKey > 0 && <div key={flashKey} className="camera-flash" aria-hidden="true" />}
      {scale > 0 && renderOverlay && <div className="camera-overlay">{renderOverlay(scale, frameSize!)}</div>}
      {scale > 0 &&
        faces.map((face) => {
          const [x1, y1, x2, y2] = face.box;
          // video đang lật ngang nên toạ độ x cũng lật
          const left = (frameSize![0] - x2) * scale;
          return (
            <div
              key={face.track_id}
              className={`face-box tone-${tone(face)}${face.track_id === selectedTrack ? " is-selected" : ""}${onSelect ? " is-clickable" : ""}`}
              onClick={onSelect ? () => onSelect(face.track_id) : undefined}
              style={{
                transform: `translate(${left}px, ${y1 * scale}px)`,
                width: (x2 - x1) * scale,
                height: (y2 - y1) * scale,
              }}
            >
              <span className="corner tl" />
              <span className="corner tr" />
              <span className="corner bl" />
              <span className="corner br" />
              {renderLabel && (
                // mặt sát mép trên: đưa nhãn vào trong khung để không bị cắt
                <div className={y1 * scale < 36 ? "face-label is-inside" : "face-label"}>{renderLabel(face)}</div>
              )}
            </div>
          );
        })}
      {children}
    </div>
  );
}
