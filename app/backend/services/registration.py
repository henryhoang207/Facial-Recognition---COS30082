"""Register a person from ~10 webcam frames, and re-embed everyone when the embedder changes."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.common.face_crop import crop_face

if TYPE_CHECKING:
    from ..context import AppContext

# Lưu crop rộng (scale 2.0) để embedder sau này dùng size/scale khác vẫn cắt lại được
SAVED_SIZE, SAVED_SCALE = 256, 2.0
SAVED_BOX = (SAVED_SIZE // 4, SAVED_SIZE // 4, SAVED_SIZE * 3 // 4, SAVED_SIZE * 3 // 4)


class RegistrationError(ValueError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class FrameCheck:
    index: int
    ok: bool
    reason: str | None = None
    live_prob: float | None = None


@dataclass
class RegistrationResult:
    person_id: int
    name: str
    used_frames: int
    checks: list[FrameCheck] = field(default_factory=list)
    duplicate_of: dict | None = None


class RegistrationService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def _check(self, index: int, frame: np.ndarray) -> tuple[FrameCheck, tuple | None]:
        models, cfg = self.ctx.models, self.ctx.config.registration
        boxes = models.detector.detect(frame)
        if len(boxes) != 1:
            return FrameCheck(index, False, "no_face" if not boxes else "more_than_one_face"), None
        box = boxes[0]
        if min(box[2] - box[0], box[3] - box[1]) < cfg.min_face_pixels:
            return FrameCheck(index, False, "face_too_small"), None
        _, live_prob = models.liveness.predict_frame(frame, box)
        if live_prob < self.ctx.liveness_threshold():
            return FrameCheck(index, False, "not_live", float(live_prob)), None
        size, scale = models.embed_crop
        return FrameCheck(index, True, None, float(live_prob)), (
            crop_face(frame, box, size, scale),
            crop_face(frame, box, SAVED_SIZE, SAVED_SCALE),
        )

    def register(
        self,
        name: str,
        student_id: str | None,
        frames: list[np.ndarray],
        force: bool = False,
    ) -> RegistrationResult:
        name = name.strip()
        if not name:
            raise RegistrationError("name is required")
        cfg, models = self.ctx.config.registration, self.ctx.models
        checks, crops, wide = [], [], []
        with models.lock:
            for index, frame in enumerate(frames):
                check, pair = self._check(index, frame)
                checks.append(check)
                if pair is not None:
                    crops.append(pair[0])
                    wide.append(pair[1])
            if len(crops) < cfg.min_valid_frames:
                raise RegistrationError(
                    f"only {len(crops)} valid frames, need {cfg.min_valid_frames}",
                    {"checks": [c.__dict__ for c in checks]},
                )
            vectors = np.stack([models.embedder.embed(c) for c in crops])
        # trung bình các embedding rồi chuẩn hoá lại về độ dài 1
        mean = vectors.mean(axis=0)
        mean = (mean / np.linalg.norm(mean)).astype(np.float32)

        duplicate = None
        if cfg.duplicate_check:
            best = self.ctx.gallery.match(mean, 1)
            if best and best[0].score >= self.ctx.recognition_threshold():
                duplicate = {
                    "person_id": best[0].person_id,
                    "name": best[0].name,
                    "score": round(best[0].score, 4),
                }
                if not force:
                    raise RegistrationError(
                        "face already registered", {"duplicate_of": duplicate}
                    )

        person = self.ctx.repo.create_person(
            name,
            student_id or None,
            mean,
            models.model_version,
            str(models.embedder.preprocessing_version),
            len(crops),
        )
        self._save_crops(int(person.id), wide)
        self.ctx.gallery.add(int(person.id), person.name, mean)
        return RegistrationResult(int(person.id), person.name, len(crops), checks, duplicate)

    def _save_crops(self, person_id: int, crops: list[np.ndarray]) -> None:
        folder = self.ctx.config.path(self.ctx.config.storage.faces_dir) / str(person_id)
        folder.mkdir(parents=True, exist_ok=True)
        for index, crop in enumerate(crops):
            cv2.imwrite(str(folder / f"{index:02d}.png"), crop)

    def delete_faces(self, person_id: int) -> None:
        folder = self.ctx.config.path(self.ctx.config.storage.faces_dir) / str(person_id)
        shutil.rmtree(folder, ignore_errors=True)

    def reembed_all(self) -> dict[str, list[int]]:
        """Tính lại embedding từ ảnh crop đã lưu cho người có model_version cũ."""
        models = self.ctx.models
        size, scale = models.embed_crop
        done, missing = [], []
        faces_dir = self.ctx.config.path(self.ctx.config.storage.faces_dir)
        for person_id in self.ctx.repo.persons_needing_reembed(models.model_version):
            images = sorted((faces_dir / str(person_id)).glob("*.png"))
            crops = [cv2.imread(str(p)) for p in images]
            crops = [c for c in crops if c is not None]
            if not crops:
                missing.append(person_id)
                continue
            crops = [crop_face(c, SAVED_BOX, size, scale) for c in crops]
            with models.lock:
                vectors = np.stack([models.embedder.embed(c) for c in crops])
            mean = vectors.mean(axis=0)
            mean = (mean / np.linalg.norm(mean)).astype(np.float32)
            self.ctx.repo.replace_embedding(
                person_id,
                mean,
                models.model_version,
                str(models.embedder.preprocessing_version),
                len(crops),
            )
            done.append(person_id)
        return {"reembedded": done, "missing_faces": missing}
