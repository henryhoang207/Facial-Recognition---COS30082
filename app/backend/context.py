"""Shared state for the whole app: config, models, gallery, database, services."""

from __future__ import annotations

import logging
import threading

from .config import AppConfig, apply_overrides, load_config, save_overrides
from .db.database import make_engine
from .db.repository import Repository
from .gallery import GalleryCache
from .models.registry import ModelBundle, build_models
from .services.attendance import AttendanceService
from .services.registration import RegistrationService

log = logging.getLogger("faceguard")


class AppContext:
    def __init__(
        self, config: AppConfig, models: ModelBundle | None = None, config_path=None
    ):
        self.config = config
        self._config_source = config_path
        self.repo = Repository(make_engine(config.path(config.storage.db)))
        self.models = models or build_models(config)
        self.gallery = GalleryCache(self.models.embedding_dim)
        self.attendance = AttendanceService(self)
        self.registration = RegistrationService(self)
        self._reload_lock = threading.Lock()
        self.sync_gallery()
        self.attendance.session_id()

    @classmethod
    def from_file(cls, path=None) -> AppContext:
        return cls(load_config(path), config_path=path)

    # thresholds ----------------------------------------------------------

    def recognition_threshold(self) -> float:
        value = self.config.thresholds.recognition
        return self.models.checkpoint_recognition_threshold if value is None else value

    def liveness_threshold(self) -> float:
        value = self.config.thresholds.liveness
        return float(self.models.liveness.threshold) if value is None else value

    def mask_threshold(self) -> float | None:
        if self.models.mask is None:
            return None
        value = self.config.thresholds.mask
        return float(self.models.mask.threshold) if value is None else value

    # lifecycle -----------------------------------------------------------

    def sync_gallery(self) -> dict:
        """Tính lại embedding cũ (nếu đổi embedder) rồi nạp gallery vào RAM."""
        report = self.registration.reembed_all()
        if report["reembedded"]:
            log.info("re-embedded persons %s", report["reembedded"])
        if report["missing_faces"]:
            log.warning(
                "persons %s have no saved faces for the current embedder; re-register them",
                report["missing_faces"],
            )
        self.gallery = GalleryCache(self.models.embedding_dim)
        self.gallery.load(self.repo.load_embeddings(self.models.model_version))
        return report

    def reload_models(self) -> dict:
        with self._reload_lock:
            fresh = load_config(self._config_source)
            fresh = apply_overrides(fresh, self.runtime_settings())
            models = build_models(fresh)
            with self.models.lock:
                self.config, self.models = fresh, models
            return self.sync_gallery()

    def runtime_settings(self) -> dict:
        return {
            "thresholds": self.config.thresholds.model_dump(),
            "pipeline": self.config.pipeline.model_dump(),
            "features": dict(self.config.features),
        }

    def update_settings(self, patch: dict) -> dict:
        self.config = apply_overrides(self.config, patch)
        save_overrides(self.config)
        return self.runtime_settings()

    def status(self) -> dict:
        return {
            "models": self.models.info,
            "thresholds": {
                "recognition": self.recognition_threshold(),
                "liveness": self.liveness_threshold(),
                "mask": self.mask_threshold(),
            },
            "gallery_size": len(self.gallery),
            "embedding_dim": self.models.embedding_dim,
            "active_session_id": self.attendance.session_id(),
            "mock_mode": all(
                spec.type in ("dummy", "center")
                for spec in (
                    self.config.models.liveness,
                    self.config.models.embedder,
                    self.config.models.emotion,
                )
            ),
        }
