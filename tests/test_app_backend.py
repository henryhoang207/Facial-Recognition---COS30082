"""Backend tests with dummy models: python -m pytest tests/test_app_backend.py"""

import struct
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from app.backend.main import create_app
from app.backend.pipeline.tracker import Tracker, iou


def _config(tmp_path: Path) -> Path:
    config = {
        "models": {
            "detector": {"type": "center"},
            "liveness": {"type": "dummy"},
            "embedder": {"type": "dummy", "dim": 64},
            "emotion": {"type": "dummy"},
        },
        "pipeline": {"stable_frames": 3, "smooth_window": 4, "spoof_cooldown_s": 60},
        "registration": {"min_face_pixels": 10, "min_valid_frames": 3},
        "storage": {
            "db": str(tmp_path / "test.db"),
            "faces_dir": str(tmp_path / "faces"),
            "spoof_dir": str(tmp_path / "instance" / "spoof"),
            "settings_override": str(tmp_path / "settings.json"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    frame[60:180, 100:220] = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    return frame


def _jpeg(frame: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.backend.services.attendance as attendance_module

    # ảnh spoof lưu tương đối so với app/instance -> trỏ về thư mục tạm
    monkeypatch.setattr(
        attendance_module.AttendanceService,
        "record_spoof",
        _patched_record_spoof(tmp_path),
    )
    with TestClient(create_app(_config(tmp_path))) as test_client:
        yield test_client


def _patched_record_spoof(tmp_path):
    def record(self, frame, box, live_prob):
        session_id = self.session_id()
        event = self.ctx.repo.add_spoof_event(session_id, live_prob, None)
        return {"session_id": session_id, "event_id": event.id, "live_prob": live_prob}

    return record


def _register(client, name, seed, force=False):
    files = [
        ("frames", (f"{i}.jpg", _jpeg(_face(seed)), "image/jpeg")) for i in range(4)
    ]
    return client.post(
        "/api/persons/register",
        data={"name": name, "student_id": "1", "force": str(force).lower()},
        files=files,
    )


def test_tracker_keeps_id_and_drops_lost_tracks():
    tracker = Tracker(iou_threshold=0.3, ttl=1, window=4)
    first, _ = tracker.update([(10, 10, 60, 60)])
    second, _ = tracker.update([(12, 12, 62, 62)])
    assert first[0].track_id == second[0].track_id
    tracker.update([])
    _, removed = tracker.update([])
    assert removed == [first[0].track_id]
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_health_and_settings(client):
    health = client.get("/api/health").json()
    assert health["embedding_dim"] == 64 and health["mock_mode"]
    updated = client.put("/api/settings", json={"thresholds": {"recognition": 0.7}})
    assert updated.json()["effective"]["recognition"] == 0.7


def test_register_then_duplicate_is_rejected(client):
    created = _register(client, "Alice", seed=1)
    assert created.status_code == 201, created.text
    assert created.json()["used_frames"] == 4
    duplicate = _register(client, "Alice again", seed=1)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["duplicate_of"]["name"] == "Alice"
    assert [p["name"] for p in client.get("/api/persons").json()] == ["Alice"]


def test_stream_recognises_and_logs_once(client):
    _register(client, "Alice", seed=1)
    frame = _jpeg(_face(1))
    messages = []
    with client.websocket_connect("/ws/stream?mode=pipeline") as ws:
        assert ws.receive_json()["type"] == "ready"
        for frame_id in range(6):
            ws.send_bytes(struct.pack(">I", frame_id) + frame)
            result = ws.receive_json()
            assert result["frame_id"] == frame_id
            messages.append(result)
            if frame_id == 2:  # stable_frames = 3 -> sự kiện ở frame thứ 3
                event = ws.receive_json()
                assert event["type"] == "attendance_logged"
                assert event["name"] == "Alice"
    face = messages[-1]["faces"][0]
    assert face["identity"]["name"] == "Alice" and face["logged"]
    assert face["liveness"]["label"] == "live" and face["emotion"]["label"]
    session_id = client.get("/api/health").json()["active_session_id"]
    report = client.get(f"/api/sessions/{session_id}/attendance").json()
    assert [row["name"] for row in report["present"]] == ["Alice"]
    csv_text = client.get(f"/api/sessions/{session_id}/export.csv").text
    assert "Alice" in csv_text


def test_live_gesture_stream_without_a_face(client, monkeypatch):
    lifecycle = []
    detected = {"status": "detected", "label": "A", "confidence": 0.9}
    no_hand = {"status": "no_hand", "label": None, "confidence": None}

    @contextmanager
    def open_stream():
        lifecycle.append("opened")
        results = iter((detected, no_hand, no_hand, no_hand))

        def predict(frame):
            assert frame.shape == (240, 320, 3)
            return next(results)

        try:
            yield predict
        finally:
            lifecycle.append("closed")

    models = client.app.state.ctx.models
    monkeypatch.setattr(models, "gesture", SimpleNamespace(open_stream=open_stream))
    monkeypatch.setattr(models.detector, "detect", lambda frame: [])
    frame = _jpeg(np.zeros((240, 320, 3), dtype=np.uint8))
    with client.websocket_connect("/ws/stream?mode=pipeline") as ws:
        assert ws.receive_json()["type"] == "ready"
        for frame_id, expected in enumerate((detected, no_hand, no_hand, no_hand)):
            ws.send_bytes(struct.pack(">I", frame_id) + frame)
            result = ws.receive_json()
            assert result["type"] == "result" and result["frame_id"] == frame_id
            assert result["gesture"] == expected
            assert result["faces"] == []
        assert lifecycle == ["opened"]
    assert lifecycle == ["opened", "closed"]
    session_id = client.get("/api/health").json()["active_session_id"]
    report = client.get(f"/api/sessions/{session_id}/attendance").json()
    assert report["present"] == [] and report["stats"]["spoof_blocked"] == 0


@pytest.mark.parametrize("mode", ["pipeline", "liveness"])
def test_stream_without_configured_gesture_model(client, mode):
    frame = _jpeg(_face(1))
    with client.websocket_connect(f"/ws/stream?mode={mode}") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(struct.pack(">I", 1) + frame)
        result = ws.receive_json()
    assert result["faces"]
    if mode == "pipeline":
        assert result["gesture"] == {
            "status": "unavailable",
            "label": None,
            "confidence": None,
        }
    else:
        assert "gesture" not in result


def test_spoof_never_reaches_recognition(client):
    _register(client, "Alice", seed=1)
    client.app.state.ctx.models.liveness.force_spoof = True
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        events = []
        for frame_id in range(4):
            ws.send_bytes(struct.pack(">I", frame_id) + _jpeg(_face(1)))
            face = ws.receive_json()["faces"][0]
            assert face["liveness"]["label"] == "spoof"
            assert face["identity"] is None and not face["logged"]
            if frame_id == 2:
                events.append(ws.receive_json())
    assert events[0]["type"] == "spoof_blocked"
    session_id = client.get("/api/health").json()["active_session_id"]
    report = client.get(f"/api/sessions/{session_id}/attendance").json()
    assert report["present"] == [] and report["stats"]["spoof_blocked"] == 1


def test_new_session_and_delete_person(client):
    person = _register(client, "Bob", seed=2).json()
    session = client.post("/api/sessions", json={"name": "Lab 2"}).json()
    assert session["active"]
    assert client.delete(f"/api/persons/{person['person_id']}").status_code == 204
    assert client.get("/api/persons").json() == []
    assert len(client.app.state.ctx.gallery) == 0


def test_switching_embedder_reembeds_from_saved_faces(tmp_path):
    from app.backend.config import load_config
    from app.backend.context import AppContext

    config_path = _config(tmp_path)
    with TestClient(create_app(config_path)) as client:
        assert _register(client, "Alice", seed=1).status_code == 201

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["models"]["embedder"]["dim"] = 32  # embedder khác -> model_version khác
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    ctx = AppContext(load_config(config_path))
    assert ctx.models.embedding_dim == 32
    assert len(ctx.gallery) == 1
    assert ctx.repo.list_persons(ctx.models.model_version)[0]["embedding_ok"]
