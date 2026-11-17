"""WebSocket /ws/stream: browser sends JPEG frames, server answers with one JSON result per frame.

Client -> server (binary): 4-byte big-endian frame_id + JPEG bytes.
Server -> client (text JSON): {"type": "result", "frame_id", ...} for every frame,
plus {"type": "attendance_logged" | "spoof_blocked", ...} events when they happen.
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from ..pipeline.pipeline import MODES, FramePipeline

router = APIRouter()


def decode_frame(data: bytes) -> tuple[int, np.ndarray | None]:
    if len(data) < 5:
        return 0, None
    (frame_id,) = struct.unpack(">I", data[:4])
    frame = cv2.imdecode(np.frombuffer(data[4:], dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame_id, frame


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket, mode: str = "pipeline"):
    await websocket.accept()
    if mode not in MODES:
        await websocket.send_json({"type": "error", "message": f"mode must be one of {MODES}"})
        await websocket.close(code=1008)
        return
    ctx = websocket.app.state.ctx
    try:
        pipeline = FramePipeline(ctx, mode)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1008)
        return
    await websocket.send_json({"type": "ready", "mode": mode, **ctx.status()})
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None:
                continue  # client chỉ gửi frame dạng binary
            frame_id, frame = decode_frame(data)
            if frame is None:
                await websocket.send_json(
                    {"type": "error", "frame_id": frame_id, "message": "bad frame"}
                )
                continue
            # pipeline chạy trong thread pool để không chặn event loop
            result, events = await run_in_threadpool(pipeline.process, frame)
            await websocket.send_json({**result, "frame_id": frame_id})
            for event in events:
                await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.close()
