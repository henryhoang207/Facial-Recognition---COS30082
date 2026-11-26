"""Individual feature modes of the web app: python -m pytest tests/test_app_individual.py"""

import struct
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from torch import nn
from torchvision.models import mobilenet_v3_small

from app.backend.individual import (
    BackgroundRunner,
    challenge,
    face_parsing,
    from_box,
    gesture,
    head_pose,
    to_box,
)
from app.backend.individual.geometry import head_angles
from app.backend.main import create_app
from app.backend.pipeline.pipeline import _Clock
from app.backend.pipeline.tracker import Track

# 5 điểm YuNet của một mặt nhìn thẳng: mắt phải, mắt trái, mũi, khoé miệng phải, khoé miệng trái
FRONT = np.array([[130, 100], [190, 100], [160, 133], [138, 160], [182, 160]], np.float32)


def _turned(degrees: float) -> np.ndarray:
    points = FRONT.copy()
    points[2, 0] += np.tan(np.radians(degrees)) * 0.55 * 60  # đầu mũi lệch theo góc quay
    return points


def test_head_angles_sign_and_direction():
    assert abs(head_angles(FRONT)["yaw"]) < 1
    assert head_angles(_turned(30))["yaw"] == pytest.approx(30, abs=0.5)
    assert head_pose.direction(head_angles(_turned(-30))["yaw"], 0) == "right"
    assert head_pose.direction(0, 20) == "up"


def _challenge(probs):
    model = SimpleNamespace(threshold=0.9, predict_frame=lambda frame, box: probs.pop(0))
    pipeline = SimpleNamespace(_identity=lambda frame, track, clock: {"name": "Alice"})
    return challenge.Feature(None, model, pipeline)


def _run(feature, points, frames):
    track = Track(1, (100, 60, 220, 200), 4)
    out = None
    for _ in range(frames):
        out = feature.face(None, track, points, _Clock())
    return out


def test_confident_live_face_skips_the_challenge():
    out = _run(_challenge([0.99] * 10), FRONT, challenge.CHECK_FRAMES)
    assert out["state"] == "passed" and out["reason"] == "live"
    assert out["identity"] == {"name": "Alice"}


def test_doubtful_face_must_turn_both_ways(monkeypatch):
    monkeypatch.setattr(challenge.random, "sample", lambda steps, k: ["left", "right"])
    feature = _challenge([0.5] * 40)
    out = _run(feature, FRONT, challenge.CHECK_FRAMES)
    assert out["state"] == "challenge" and out["step"] == "left" and out["identity"] is None
    out = _run(feature, _turned(-30), challenge.HOLD_FRAMES)  # quay sai hướng
    assert out["step_index"] == 0
    out = _run(feature, _turned(30), challenge.HOLD_FRAMES)
    assert out["step"] == "right"
    out = _run(feature, _turned(-30), challenge.HOLD_FRAMES)
    assert out["state"] == "passed" and out["reason"] == "challenge"


def test_challenge_times_out(monkeypatch):
    feature = _challenge([0.5] * 10)
    _run(feature, FRONT, challenge.CHECK_FRAMES)
    clock = challenge.time.monotonic() + challenge.CHALLENGE_SECONDS + 1
    monkeypatch.setattr(challenge.time, "monotonic", lambda: clock)
    out = _run(feature, FRONT, 1)
    assert out["state"] == "failed" and out["reason"] == "timeout"


def test_adaptive_liveness_checkpoint(tmp_path):
    net = mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    path = tmp_path / "student.pt"
    torch.save(
        {
            "model": {f"net.{k}": v for k, v in net.state_dict().items()},
            "config": {"model": "mobilenetv3_small", "img_size": 224, "crop_scale": 2.0},
            "threshold": 0.97,
        },
        path,
    )
    model = challenge.AdaptiveLiveness(path)
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert 0 <= model.predict_frame(frame, (100, 60, 220, 180)) <= 1
    assert model.threshold == pytest.approx(0.97)


def test_face_parsing_overlay_from_points():
    frame = np.zeros((240, 320, 3), np.uint8)
    mask, region = face_parsing.geometry_mask(frame, (100, 60, 220, 200), FRONT)
    assert mask.shape == (region[3] - region[1], region[2] - region[0])
    overlay, parts = face_parsing.colour(mask, face_parsing.CLASSES)
    assert overlay.startswith("data:image/png;base64,")
    names = {part["name"] for part in parts}
    assert {"Skin", "Eyes", "Nose", "Lips", "Hair"} <= names
    assert sum(part["share"] for part in parts) == pytest.approx(1.0, abs=1e-3)


def _gesture_checkpoint(path: Path) -> None:
    features = import_module("src.hand-gestures.features")
    models = import_module("src.hand-gestures.models")
    columns = list(features.sequence_columns(False))
    classes = ["J", "Z", "HELLO", "OTHER"]
    net = models.GestureGRU(len(columns), len(classes))
    torch.save(
        {
            "format_version": 1,
            "checkpoint_kind": "inference",
            "task": {f"model.{k}": v for k, v in net.state_dict().items()},
            "metadata": {
                "model": "mlp_gru",
                "classes": classes,
                "columns": columns,
                "preprocess": {
                    "sequence_length": 32,
                    "include_body": False,
                    "min_hand_coverage": 0.8,
                    "max_hand_gap_sec": 0.1,
                },
            },
        },
        path,
    )


def test_gesture_stream_collects_then_classifies(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    checkpoint, hand = tmp_path / "gesture.pt", tmp_path / "hand.task"
    _gesture_checkpoint(checkpoint)
    hand.write_bytes(b"stub")
    model = gesture.DynamicGestureModel(checkpoint, hand)

    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.6, (21, 3)).astype(np.float32)
    base[9] = base[0] + [0.1, 0.1, 0]  # bàn tay có kích thước
    visible = {"on": False}

    @contextmanager
    def fake_landmarker(_):
        def detect(frame, ms):
            if not visible["on"]:
                return np.full((21, 3), np.nan, np.float32), "", 0.0, np.full((4, 2), np.nan)
            return base + ms / 1e5, "Right", 0.9, np.full((4, 2), np.nan)

        yield detect

    monkeypatch.setattr(gesture._module("landmarks"), "video_landmarker", fake_landmarker)
    now = {"t": 100.0}
    monkeypatch.setattr(gesture.time, "monotonic", lambda: now["t"])
    feature = gesture.Feature(None, model, None)
    frame = np.zeros((240, 320, 3), np.uint8)
    assert feature.frame(frame, _Clock())["status"] == "no_hand"
    visible["on"] = True
    statuses = []
    for _ in range(25):
        now["t"] += 0.06
        statuses.append(feature.frame(frame, _Clock()))
    assert statuses[0]["status"] == "collecting" and statuses[0]["points"] is not None
    final = statuses[-1]
    assert final["status"] in ("detected", "other") and final["label"] in model.classes
    assert len(final["top"]) == 3
    feature.close()


def _config(tmp_path: Path, individual: dict) -> Path:
    config = {
        "models": {
            "detector": {"type": "center"},
            "liveness": {"type": "dummy"},
            "embedder": {"type": "dummy", "dim": 64},
            "emotion": {"type": "dummy"},
            "individual": individual,
        },
        "storage": {
            "db": str(tmp_path / "test.db"),
            "faces_dir": str(tmp_path / "faces"),
            "spoof_dir": str(tmp_path / "spoof"),
            "settings_override": str(tmp_path / "settings.json"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_stream_modes_report_availability(tmp_path):
    individual = {
        "head_pose": {"type": "geometry"},
        "face_parsing": {"type": "external", "predictor": "src.missing.predictor:Missing"},
    }
    frame = np.full((240, 320, 3), 40, np.uint8)
    jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
    with TestClient(create_app(_config(tmp_path, individual))) as client:
        info = client.get("/api/health").json()["models"]["individual"]
        assert info["head_pose"]["available"] is True
        assert (
            info["face_parsing"]["available"] is False
            and "ModuleNotFoundError" in info["face_parsing"]["error"]
        )
        with client.websocket_connect("/ws/stream?mode=head_pose") as ws:
            ws.receive_json()
            ws.send_bytes(struct.pack(">I", 1) + jpeg)
            result = ws.receive_json()
            assert result["mode"] == "head_pose" and "head_pose" in result["faces"][0]
        with client.websocket_connect("/ws/stream?mode=face_parsing") as ws:
            message = ws.receive_json()
            assert message["type"] == "error" and "ModuleNotFoundError" in message["message"]


def test_face_parsing_predictor_loads_checkpoint(tmp_path):
    from src.face_parsing.predictor import FaceParsingNet, FaceParsingPredictor

    classes = face_parsing.CLASSES
    path = tmp_path / "parsing.pt"
    torch.save({"weights": FaceParsingNet(len(classes)).state_dict(), "classes": classes, "side": 128}, path)
    predictor = FaceParsingPredictor(path)
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    mask, region = predictor.predict_frame(frame, (120, 80, 200, 170))
    assert mask.shape == (region[3] - region[1], region[2] - region[0])
    assert mask.dtype == np.uint8 and mask.max() < len(classes)


def test_checkpoint_fallback_path(tmp_path):
    from app.backend.config import ModelSpec, load_config
    from app.backend.models.registry import _require

    config = load_config(_config(tmp_path, {}))
    old = tmp_path / "old.pt"
    old.write_bytes(b"x")
    spec = ModelSpec(type="liveness", path=str(tmp_path / "new.pt"), fallback_paths=[str(old)])
    assert _require(config, spec) == old and spec.path == str(old)
    with pytest.raises(FileNotFoundError):
        _require(config, ModelSpec(type="liveness", path=str(tmp_path / "missing.pt")))


def test_pose_from_landmarks_signs():
    import math

    from src.head_pose.predictor import FACE_3D, PNP_INDEX, pose_from_landmarks

    def project(yaw):
        a = math.radians(yaw)
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        p = (rot @ FACE_3D.T).T
        z = 3000 - p[:, 2]
        points = np.zeros((68, 2))
        points[PNP_INDEX] = np.c_[640 * p[:, 0] / z + 320, 640 * -p[:, 1] / z + 240]
        return points

    pose = pose_from_landmarks(project(25), (640, 480))
    assert pose["yaw"] == pytest.approx(25, abs=0.5)
    assert abs(pose["pitch"]) < 0.5 and abs(pose["roll"]) < 0.5


def test_head_pose_falls_back_when_landmarks_disagree():
    far = np.zeros((68, 2)).tolist()  # landmark hỏng: tất cả dồn về góc ảnh
    model = SimpleNamespace(
        predict_frame=lambda frame, box: {"yaw": 5, "pitch": 0, "roll": 0, "landmarks": far}
    )
    feature = head_pose.Feature(None, model, None)
    box = (100, 60, 220, 200)
    out = feature._estimate(None, box, _turned(30))
    assert out["source"] == "detector" and out["yaw"] == pytest.approx(30, abs=1)
    feature.close()


def test_background_runner_keeps_stream_moving():
    import threading

    gate = threading.Event()
    runner = BackgroundRunner(lambda value: (gate.wait(2), value)[1])
    runner.offer(1, [1, 2], "first")
    runner.offer(2, [1, 2], "skipped while busy")
    assert runner.results == {}
    gate.set()
    runner._busy.result(timeout=2)
    runner.offer(2, [1, 2], "second")  # thu kết quả 1, rồi tới lượt mặt 2
    runner._busy.result(timeout=2)
    runner.poll()
    assert runner.results == {1: "first", 2: "second"}
    runner.close()


def test_results_follow_the_face_box():
    corners = to_box((100, 100, 200, 200), [[90, 80], [210, 220]])
    moved = from_box((150, 120, 350, 320), corners)  # mặt dịch và to gấp đôi
    assert np.allclose(moved, [[130, 80], [370, 360]])


def test_broken_gpu_falls_back_to_cpu(tmp_path, monkeypatch):
    from src.face_parsing.predictor import FaceParsingNet, FaceParsingPredictor

    path = tmp_path / "parsing.pt"
    classes = face_parsing.CLASSES
    torch.save({"weights": FaceParsingNet(len(classes)).state_dict(), "classes": classes, "side": 64}, path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # máy báo có GPU nhưng không chạy được
    predictor = FaceParsingPredictor(path, device="auto")
    assert predictor.device.type == "cpu" and predictor.gpu_error
