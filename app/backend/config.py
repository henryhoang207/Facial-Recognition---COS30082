"""App configuration: app/config.yaml + runtime overrides in app/instance/settings.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "app" / "config.yaml"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    path: str | None = None


class ModelsConfig(BaseModel):
    detector: ModelSpec
    liveness: ModelSpec
    embedder: ModelSpec
    emotion: ModelSpec
    mask: ModelSpec | None = None  # tính năng cá nhân (Nam Khánh); không có thì tab Mask báo chưa cấu hình
    attendance: ModelSpec | None = None
    gesture: ModelSpec | None = None


class Thresholds(BaseModel):
    recognition: float | None = Field(default=None, ge=-1, le=1)
    liveness: float | None = Field(default=None, ge=0, le=1)
    mask: float | None = Field(default=None, ge=0, le=1)


class PipelineConfig(BaseModel):
    max_faces: int = Field(default=5, ge=1)
    smooth_window: int = Field(default=8, ge=1)
    stable_frames: int = Field(default=5, ge=1)
    track_iou: float = Field(default=0.3, gt=0, lt=1)
    track_ttl_frames: int = Field(default=10, ge=1)
    spoof_cooldown_s: float = Field(default=5.0, ge=0)
    model_every_n: int = Field(default=1, ge=1)


class RegistrationConfig(BaseModel):
    frames: int = Field(default=10, ge=1)
    min_valid_frames: int = Field(default=6, ge=1)
    min_face_pixels: int = Field(default=80, ge=1)
    duplicate_check: bool = True


class StorageConfig(BaseModel):
    db: str = "app/instance/faceguard.db"
    faces_dir: str = "app/instance/faces"
    spoof_dir: str = "app/instance/spoof"
    settings_override: str = "app/instance/settings.json"


class AppConfig(BaseModel):
    device: str = "cpu"
    models: ModelsConfig
    thresholds: Thresholds = Thresholds()
    pipeline: PipelineConfig = PipelineConfig()
    registration: RegistrationConfig = RegistrationConfig()
    storage: StorageConfig = StorageConfig()
    features: dict[str, bool] = {}

    def path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate


# Những nhóm được phép đổi lúc chạy qua /api/settings
RUNTIME_KEYS = ("thresholds", "pipeline", "features")


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(raw)
    override = config.path(config.storage.settings_override)
    if override.exists():
        config = apply_overrides(config, json.loads(override.read_text("utf-8")))
    return config


def apply_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    data = config.model_dump()
    for key in RUNTIME_KEYS:
        if key in overrides and isinstance(overrides[key], dict):
            data[key] = {**data[key], **overrides[key]}
    return AppConfig.model_validate(data)


def save_overrides(config: AppConfig) -> None:
    target = config.path(config.storage.settings_override)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: getattr(config, key) for key in RUNTIME_KEYS}
    payload = {
        key: value.model_dump() if isinstance(value, BaseModel) else value
        for key, value in payload.items()
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
