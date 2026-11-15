"""Fill the database with mock data so every screen has something to show.

    python -m app.backend.seed --config app/config.mock.yaml --reset

Mock people get random embeddings, so they never match a real face; register
yourself through the app to test recognition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
from sqlmodel import Session, SQLModel

from .config import load_config
from .db.database import make_engine
from .db.models import Attendance, AttendanceSession, SpoofEvent
from .db.repository import Repository
from .models.registry import build_embedder

MOCK_PEOPLE = [
    ("Hoàng Hữu Hoan", "104000001", "Neutral"),
    ("Trần Mạnh Đức", "104000002", "Surprise"),
    ("Lê Đức Thành", "104000003", "Neutral"),
    ("Nguyễn Nam Khánh", "104000004", "Happy"),
    ("Giáp Hoàng Gia Khánh", "104000005", "Happy"),
    ("Lê Đức Anh", "104000006", None),
]


def seed(config_path: str | None, reset: bool) -> None:
    config = load_config(config_path)
    db_path = config.path(config.storage.db)
    engine = make_engine(db_path)
    if reset:
        SQLModel.metadata.drop_all(engine)
        SQLModel.metadata.create_all(engine)
    repo = Repository(engine)
    embedder, _ = build_embedder(config, config.models.embedder)
    rng = np.random.default_rng(7)

    person_ids = []
    for name, student_id, _ in MOCK_PEOPLE:
        vector = rng.standard_normal(embedder.embedding_dim).astype(np.float32)
        vector /= np.linalg.norm(vector)
        person = repo.create_person(
            name,
            student_id,
            vector,
            str(embedder.model_version),
            str(embedder.preprocessing_version),
            0,
        )
        person_ids.append(person.id)

    today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    with Session(engine) as db:
        past = AttendanceSession(
            name="COS30082 tuần trước",
            started_at=today - timedelta(days=7),
            ended_at=today - timedelta(days=7, hours=-2),
        )
        current = AttendanceSession(name="COS30082 hôm nay", started_at=today)
        db.add_all([past, current])
        db.flush()
        for offset, (person_id, (_, _, emotion)) in enumerate(
            zip(person_ids, MOCK_PEOPLE, strict=True)
        ):
            db.add(
                Attendance(
                    session_id=past.id,
                    person_id=person_id,
                    first_seen=past.started_at + timedelta(minutes=offset),
                    live_prob=0.97,
                    match_score=0.78,
                    emotion="Neutral",
                )
            )
            if emotion is None:
                continue  # người cuối vắng mặt hôm nay
            db.add(
                Attendance(
                    session_id=current.id,
                    person_id=person_id,
                    first_seen=today + timedelta(minutes=2 * offset, seconds=12),
                    live_prob=round(0.93 + 0.01 * offset, 2),
                    match_score=round(0.74 + 0.02 * offset, 2),
                    emotion=emotion,
                )
            )
        for minutes, prob in [(3, 0.09), (5, 0.12), (6, 0.06)]:
            db.add(
                SpoofEvent(
                    session_id=current.id,
                    ts=today + timedelta(minutes=minutes),
                    live_prob=prob,
                    snapshot_path=None,
                )
            )
        db.commit()
    print(f"seeded {len(person_ids)} people and 2 sessions into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--reset", action="store_true", help="drop all tables first")
    args = parser.parse_args()
    seed(args.config, args.reset)


if __name__ == "__main__":
    main()
