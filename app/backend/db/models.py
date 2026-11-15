"""SQLite tables (SQLModel). See app/README.md for the schema diagram."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, LargeBinary, UniqueConstraint
from sqlmodel import Field, SQLModel


def now() -> datetime:
    # giờ địa phương của máy chạy app (không múi giờ), đủ cho điểm danh trên một laptop
    return datetime.now().replace(microsecond=0)


def _time(**kwargs):
    return Field(sa_type=DateTime(timezone=False), **kwargs)


class Person(SQLModel, table=True):
    __tablename__ = "persons"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    student_id: str | None = None
    created_at: datetime = _time(default_factory=now)


class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"

    id: int | None = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="persons.id", index=True, ondelete="CASCADE")
    vector: bytes = Field(sa_column=Column(LargeBinary, nullable=False))  # float32
    dim: int
    model_version: str = Field(index=True)
    preprocessing_version: str
    n_frames: int
    created_at: datetime = _time(default_factory=now)


class AttendanceSession(SQLModel, table=True):
    __tablename__ = "sessions"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    started_at: datetime = _time(default_factory=now)
    ended_at: datetime | None = _time(default=None)


class Attendance(SQLModel, table=True):
    __tablename__ = "attendance"
    # DB tự chặn ghi trùng: mỗi người một dòng mỗi session
    __table_args__ = (UniqueConstraint("session_id", "person_id"),)

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="sessions.id", index=True, ondelete="CASCADE")
    person_id: int = Field(foreign_key="persons.id", index=True, ondelete="CASCADE")
    first_seen: datetime = _time(default_factory=now)
    live_prob: float
    match_score: float
    emotion: str | None = None


class SpoofEvent(SQLModel, table=True):
    __tablename__ = "spoof_events"

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="sessions.id", index=True, ondelete="CASCADE")
    ts: datetime = _time(default_factory=now)
    live_prob: float
    snapshot_path: str | None = None
