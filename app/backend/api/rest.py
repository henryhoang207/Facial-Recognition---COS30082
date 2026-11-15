"""REST API: persons, sessions/attendance, settings, system status."""

from __future__ import annotations

import csv
import io
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..services.attendance import snapshot_path
from ..services.registration import RegistrationError

router = APIRouter(prefix="/api")


def _ctx(request: Request):
    return request.app.state.ctx


# system --------------------------------------------------------------------


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    return {"ok": True, **_ctx(request).status()}


@router.post("/models/reload")
async def reload_models(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    try:
        report = await run_in_threadpool(ctx.reload_models)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "gallery": report, **ctx.status()}


# settings --------------------------------------------------------------------


class SettingsPatch(BaseModel):
    thresholds: dict[str, float | None] | None = None
    pipeline: dict[str, float | int] | None = None
    features: dict[str, bool] | None = None


@router.get("/settings")
def get_settings(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    return {**ctx.runtime_settings(), "effective": ctx.status()["thresholds"]}


@router.put("/settings")
def put_settings(patch: SettingsPatch, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    try:
        ctx.update_settings(patch.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**ctx.runtime_settings(), "effective": ctx.status()["thresholds"]}


# persons ----------------------------------------------------------------------


@router.get("/persons")
def list_persons(request: Request) -> list[dict[str, Any]]:
    ctx = _ctx(request)
    return ctx.repo.list_persons(ctx.models.model_version)


@router.post("/persons/register", status_code=201)
async def register_person(
    request: Request,
    name: str = Form(...),
    student_id: str | None = Form(None),
    force: bool = Form(False),
    frames: list[UploadFile] = File(...),
) -> dict[str, Any]:
    ctx = _ctx(request)
    images = []
    for upload in frames:
        data = np.frombuffer(await upload.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    try:
        result = await run_in_threadpool(
            ctx.registration.register, name, student_id, images, force
        )
    except RegistrationError as exc:
        status = 409 if "duplicate_of" in exc.details else 422
        raise HTTPException(status, {"message": str(exc), **exc.details}) from exc
    return {
        "person_id": result.person_id,
        "name": result.name,
        "used_frames": result.used_frames,
        "checks": [c.__dict__ for c in result.checks],
        "duplicate_of": result.duplicate_of,
    }


@router.delete("/persons/{person_id}", status_code=204)
def delete_person(person_id: int, request: Request) -> None:
    ctx = _ctx(request)
    if not ctx.repo.delete_person(person_id):
        raise HTTPException(404, "person not found")
    ctx.gallery.remove(person_id)
    ctx.registration.delete_faces(person_id)
    ctx.attendance.forget_cache()


# sessions + attendance --------------------------------------------------------


class SessionCreate(BaseModel):
    name: str


@router.get("/sessions")
def list_sessions(request: Request) -> list[dict[str, Any]]:
    ctx = _ctx(request)
    active = ctx.attendance.session_id()
    return [
        {**s.model_dump(), "active": s.id == active} for s in ctx.repo.list_sessions()
    ]


@router.post("/sessions", status_code=201)
def start_session(body: SessionCreate, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    session = ctx.repo.start_session(body.name.strip() or "Session")
    ctx.attendance.forget_cache()
    return {**session.model_dump(), "active": True}


@router.patch("/sessions/{session_id}/end")
def end_session(session_id: int, request: Request) -> dict[str, Any]:
    session = _ctx(request).repo.end_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session.model_dump()


def _session_or_404(ctx, session_id: int):
    session = ctx.repo.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@router.get("/sessions/{session_id}/attendance")
def attendance(session_id: int, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    session = _session_or_404(ctx, session_id)
    present = ctx.repo.attendance_rows(session_id)
    present_ids = {row["person_id"] for row in present}
    absent = [
        {"person_id": p["id"], "name": p["name"], "student_id": p["student_id"]}
        for p in ctx.repo.list_persons(ctx.models.model_version)
        if p["id"] not in present_ids
    ]
    return {
        "session": session.model_dump(),
        "stats": ctx.repo.session_stats(session_id),
        "present": present,
        "absent": absent,
    }


@router.get("/sessions/{session_id}/spoof-events")
def spoof_events(session_id: int, request: Request) -> list[dict[str, Any]]:
    ctx = _ctx(request)
    _session_or_404(ctx, session_id)
    return [e.model_dump() for e in ctx.repo.spoof_events(session_id)]


@router.get("/snapshots/{path:path}")
def snapshot(path: str, request: Request):
    try:
        target = snapshot_path(_ctx(request), path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.exists():
        raise HTTPException(404, "snapshot not found")
    return FileResponse(target)


@router.get("/sessions/{session_id}/export.csv")
def export_csv(session_id: int, request: Request):
    ctx = _ctx(request)
    session = _session_or_404(ctx, session_id)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["name", "student_id", "first_seen", "live_prob", "match_score", "emotion"])
    for row in ctx.repo.attendance_rows(session_id):
        writer.writerow(
            [row["name"], row["student_id"] or "", row["first_seen"].isoformat(sep=" "),
             f"{row['live_prob']:.4f}", f"{row['match_score']:.4f}", row["emotion"] or ""]
        )
    filename = f"attendance_session{session.id}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
