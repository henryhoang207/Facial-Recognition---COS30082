"""Inventory team photos for gallery registration, never for training splits."""

from collections import Counter

import cv2

from .catalog import IMAGE_EXTENSIONS, FingerprintCache, write_csv, write_json
from .config import PreparationConfig


def select_gallery_face(boxes, width: int, height: int, min_area_fraction: float):
    if not boxes:
        return "no_face", None
    if len(boxes) > 1:
        return "multiple_faces", None
    x1, y1, x2, y2 = boxes[0]
    if (x2 - x1) * (y2 - y1) < width * height * min_area_fraction:
        return "small_face_review", None
    return "ready", boxes[0]


class TeamPreparer:
    def __init__(self, config: PreparationConfig):
        self.config = config

    def prepare(self) -> dict:
        config, source = self.config, self.config.team
        photos = sorted(
            path
            for path in source.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        fingerprints = FingerprintCache(config.output_dir).inspect(
            [path.resolve() for path in photos], config.workers
        )
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if detector.empty():
            raise ValueError("OpenCV's frontal-face cascade is unavailable")
        rows = []
        for path in photos:
            relative = path.relative_to(source.root)
            if len(relative.parts) < 2:
                raise ValueError(
                    "team photos require person_id/image or person_id/session/image"
                )
            person_id = relative.parts[0]
            session = "/".join(relative.parts[1:-1])
            entry = fingerprints[str(path.resolve())]
            row = {
                "path": relative.as_posix(),
                "person_id": person_id,
                "role": source.role,
                "session": session,
                "status": "invalid_image",
                "width": entry["width"],
                "height": entry["height"],
                "sha256": entry["sha256"],
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            }
            frame = cv2.imread(str(path)) if not entry["error"] else None
            if frame is not None:
                height, width = frame.shape[:2]
                row.update(width=width, height=height)
                scale = min(1.0, source.detector_max_side / max(height, width))
                gray = cv2.cvtColor(
                    cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY
                )
                faces = detector.detectMultiScale(
                    cv2.equalizeHist(gray),
                    scaleFactor=source.detector_scale_factor,
                    minNeighbors=source.detector_min_neighbors,
                    minSize=(source.detector_min_size, source.detector_min_size),
                )
                boxes = [
                    (
                        max(0, round(x / scale)),
                        max(0, round(y / scale)),
                        min(width, round((x + box_width) / scale)),
                        min(height, round((y + box_height) / scale)),
                    )
                    for x, y, box_width, box_height in faces
                ]
                row["status"], box = select_gallery_face(
                    boxes, width, height, source.min_face_area_fraction
                )
                if box is not None:
                    row.update(dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)))
            rows.append(row)
        mapping = {
            person: index
            for index, person in enumerate(sorted({row["person_id"] for row in rows}))
        }
        for row in rows:
            row["person_index"] = mapping[row["person_id"]]
        write_csv(
            config.output_dir / "team_gallery.csv",
            rows,
            [
                "path",
                "person_id",
                "person_index",
                "role",
                "session",
                "status",
                "width",
                "height",
                "x1",
                "y1",
                "x2",
                "y2",
                "sha256",
            ],
        )
        write_json(config.output_dir / "team_person_to_idx.json", mapping)
        summary = {
            "images": len(rows),
            "people": len(mapping),
            "per_person": dict(Counter(row["person_id"] for row in rows)),
            "status": dict(Counter(row["status"] for row in rows)),
            "role": source.role,
            "included_in_training": False,
            "policy": (
                "only ready rows have an unambiguous detected face; review Haar "
                "boxes before app registration; no cross-session evaluation claim"
            ),
        }
        write_json(config.output_dir / "team_summary.json", summary)
        return summary
