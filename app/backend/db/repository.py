"""All SQL lives here; services and the pipeline never write queries themselves."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, delete, func, select

from .models import (
    Attendance,
    AttendanceSession,
    Embedding,
    Person,
    SpoofEvent,
    now,
)


class Repository:
    def __init__(self, engine: Engine):
        self.engine = engine

    # persons + embeddings -------------------------------------------------

    def create_person(
        self,
        name: str,
        student_id: str | None,
        vector: np.ndarray,
        model_version: str,
        preprocessing_version: str,
        n_frames: int,
    ) -> Person:
        with Session(self.engine) as db:
            person = Person(name=name, student_id=student_id)
            db.add(person)
            db.flush()
            db.add(
                Embedding(
                    person_id=person.id,
                    vector=vector.astype(np.float32).tobytes(),
                    dim=int(vector.shape[0]),
                    model_version=model_version,
                    preprocessing_version=preprocessing_version,
                    n_frames=n_frames,
                )
            )
            db.commit()
            db.refresh(person)
            return person

    def replace_embedding(
        self,
        person_id: int,
        vector: np.ndarray,
        model_version: str,
        preprocessing_version: str,
        n_frames: int,
    ) -> None:
        with Session(self.engine) as db:
            db.exec(delete(Embedding).where(Embedding.person_id == person_id))
            db.add(
                Embedding(
                    person_id=person_id,
                    vector=vector.astype(np.float32).tobytes(),
                    dim=int(vector.shape[0]),
                    model_version=model_version,
                    preprocessing_version=preprocessing_version,
                    n_frames=n_frames,
                )
            )
            db.commit()

    def load_embeddings(self, model_version: str) -> list[tuple[int, str, np.ndarray]]:
        query = (
            select(Embedding, Person)
            .join(Person, Person.id == Embedding.person_id)
            .where(Embedding.model_version == model_version)
        )
        with Session(self.engine) as db:
            return [
                (person.id, person.name, np.frombuffer(emb.vector, dtype=np.float32))
                for emb, person in db.exec(query)
            ]

    def list_persons(self, model_version: str) -> list[dict]:
        with Session(self.engine) as db:
            persons = db.exec(select(Person).order_by(Person.name)).all()
            versions: dict[int, set[str]] = {}
            for person_id, version in db.exec(
                select(Embedding.person_id, Embedding.model_version)
            ):
                versions.setdefault(person_id, set()).add(version)
        return [
            {
                "id": p.id,
                "name": p.name,
                "student_id": p.student_id,
                "created_at": p.created_at,
                "embedding_ok": model_version in versions.get(p.id, set()),
            }
            for p in persons
        ]

    def persons_needing_reembed(self, model_version: str) -> list[int]:
        with Session(self.engine) as db:
            ok = select(Embedding.person_id).where(
                Embedding.model_version == model_version
            )
            return list(db.exec(select(Person.id).where(col(Person.id).not_in(ok))))

    def get_person(self, person_id: int) -> Person | None:
        with Session(self.engine) as db:
            return db.get(Person, person_id)

    def delete_person(self, person_id: int) -> bool:
        with Session(self.engine) as db:
            person = db.get(Person, person_id)
            if person is None:
                return False
            db.exec(delete(Embedding).where(Embedding.person_id == person_id))
            db.exec(delete(Attendance).where(Attendance.person_id == person_id))
            db.delete(person)
            db.commit()
            return True

    # sessions --------------------------------------------------------------

    def active_session(self) -> AttendanceSession | None:
        query = (
            select(AttendanceSession)
            .where(AttendanceSession.ended_at == None)  # noqa: E711
            .order_by(col(AttendanceSession.id).desc())
        )
        with Session(self.engine) as db:
            return db.exec(query).first()

    def start_session(self, name: str) -> AttendanceSession:
        with Session(self.engine) as db:
            for open_session in db.exec(
                select(AttendanceSession).where(AttendanceSession.ended_at == None)  # noqa: E711
            ):
                open_session.ended_at = now()
                db.add(open_session)
            created = AttendanceSession(name=name)
            db.add(created)
            db.commit()
            db.refresh(created)
            return created

    def end_session(self, session_id: int) -> AttendanceSession | None:
        with Session(self.engine) as db:
            session = db.get(AttendanceSession, session_id)
            if session and session.ended_at is None:
                session.ended_at = now()
                db.add(session)
                db.commit()
                db.refresh(session)
            return session

    def list_sessions(self) -> list[AttendanceSession]:
        with Session(self.engine) as db:
            return list(
                db.exec(
                    select(AttendanceSession).order_by(col(AttendanceSession.id).desc())
                )
            )

    def get_session(self, session_id: int) -> AttendanceSession | None:
        with Session(self.engine) as db:
            return db.get(AttendanceSession, session_id)

    # attendance + spoof ----------------------------------------------------

    def log_attendance(
        self,
        session_id: int,
        person_id: int,
        live_prob: float,
        match_score: float,
        emotion: str | None,
    ) -> bool:
        """True nếu là lần đầu người này có mặt trong session."""
        statement = (
            insert(Attendance)
            .values(
                session_id=session_id,
                person_id=person_id,
                first_seen=now(),
                live_prob=live_prob,
                match_score=match_score,
                emotion=emotion,
            )
            .on_conflict_do_nothing(index_elements=["session_id", "person_id"])
        )
        with Session(self.engine) as db:
            result = db.exec(statement)
            db.commit()
            return result.rowcount == 1

    def attendance_rows(self, session_id: int) -> list[dict]:
        query = (
            select(Attendance, Person)
            .join(Person, Person.id == Attendance.person_id)
            .where(Attendance.session_id == session_id)
            .order_by(Attendance.first_seen)
        )
        with Session(self.engine) as db:
            return [
                {
                    "person_id": person.id,
                    "name": person.name,
                    "student_id": person.student_id,
                    "first_seen": row.first_seen,
                    "live_prob": row.live_prob,
                    "match_score": row.match_score,
                    "emotion": row.emotion,
                }
                for row, person in db.exec(query)
            ]

    def add_spoof_event(
        self, session_id: int, live_prob: float, snapshot_path: str | None
    ) -> SpoofEvent:
        with Session(self.engine) as db:
            spoof = SpoofEvent(
                session_id=session_id, live_prob=live_prob, snapshot_path=snapshot_path
            )
            db.add(spoof)
            db.commit()
            db.refresh(spoof)
            return spoof

    def spoof_events(self, session_id: int) -> list[SpoofEvent]:
        query = (
            select(SpoofEvent)
            .where(SpoofEvent.session_id == session_id)
            .order_by(col(SpoofEvent.ts).desc())
        )
        with Session(self.engine) as db:
            return list(db.exec(query))

    def session_stats(self, session_id: int) -> dict:
        with Session(self.engine) as db:
            registered = db.exec(select(func.count()).select_from(Person)).one()
            present = db.exec(
                select(func.count())
                .select_from(Attendance)
                .where(Attendance.session_id == session_id)
            ).one()
            spoofs = db.exec(
                select(func.count())
                .select_from(SpoofEvent)
                .where(SpoofEvent.session_id == session_id)
            ).one()
            emotions = dict(
                db.exec(
                    select(Attendance.emotion, func.count())
                    .where(Attendance.session_id == session_id)
                    .group_by(Attendance.emotion)
                ).all()
            )
        return {
            "registered": registered,
            "present": present,
            "spoof_blocked": spoofs,
            "emotions": {k: v for k, v in emotions.items() if k},
        }

    @staticmethod
    def timestamp() -> datetime:
        return now()
