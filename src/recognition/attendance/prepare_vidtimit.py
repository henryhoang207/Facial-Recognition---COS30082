"""Build MLP training manifests from separate VidTIMIT recording sessions."""

import re
from collections import defaultdict
from pathlib import Path
from threading import local
from zipfile import ZipFile

import numpy as np

from src.datasets.catalog import holdout, stable_rng
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.config import DetectorConfig
from src.recognition.preprocessing.detector import YuNetDetector, ensure_model
from src.recognition.preprocessing.prepare import digest
from src.training.config import read_yaml, resolve_device

from .config import VidTIMITConfig, load_config
from .features import attempt_features, gallery_template, unit_rows
from .prepare import AttemptPreparer


class VidTIMITPreparer(AttemptPreparer):
    cfg: VidTIMITConfig

    def probe_windows(self, clips, session, vectors):
        count = self.cfg.frame_count
        for (clip_session, clip), paths in clips.items():
            if clip_session != session:
                continue
            offset = 0
            while offset + count <= len(paths):
                window = paths[offset : offset + count]
                if all(path in vectors for path in window):
                    yield clip, window
                    offset += count
                else:
                    offset += 1

    def extract(self, archives: list[Path]):
        people = {}
        for archive in archives:
            identity = archive.stem
            clips = defaultdict(list)
            with ZipFile(archive) as source:
                for name in source.namelist():
                    match = re.fullmatch(
                        rf"{re.escape(identity)}/video/([a-z0-9]+)/([0-9]+)", name
                    )
                    if match:
                        clip, number = match.groups()
                        clips[clip].append((int(number), name))
                speech = sorted(set(clips) - {"head", "head2", "head3"})
                if len(speech) != 10 or not {"head", "head2", "head3"} <= clips.keys():
                    raise ValueError(f"unexpected VidTIMIT session layout: {archive}")
                sessions = (
                    speech[:6],
                    ("head2", *speech[6:8]),
                    ("head3", *speech[8:]),
                )
                person = {}
                for session, names in enumerate(sessions, 1):
                    for clip in names:
                        frames = sorted(clips[clip])[:: self.cfg.vidtimit.frame_stride]
                        if session == 1:
                            midpoint = len(frames) // 2
                            frames = frames[max(0, midpoint - 1) : midpoint + 2]
                        paths = []
                        for number, member in frames:
                            relative = Path(
                                "frames", identity, clip, f"{number:06d}.jpg"
                            )
                            path = self.cfg.root / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            content = source.read(member)
                            if not content.startswith(b"\xff\xd8"):
                                raise ValueError(f"expected a JPEG frame: {member}")
                            if not path.is_file() or path.read_bytes() != content:
                                path.write_bytes(content)
                            paths.append(relative.as_posix())
                        person[session, clip] = paths
                people[identity] = person
        return people

    def sample_sessions(self, people, vectors):
        cfg = self.cfg
        train_dev, test = holdout(
            list(people),
            cfg.vidtimit.test_fraction,
            stable_rng(cfg.training.seed, "vidtimit-test"),
        )
        train, dev = holdout(
            train_dev,
            cfg.dev_fraction,
            stable_rng(cfg.training.seed, "vidtimit-dev"),
        )
        pools = {"train": train, "dev": dev, "test": test}
        enrollment, attempts = {}, []
        skipped_sessions = []
        for split, names in pools.items():
            if len(names) <= cfg.gallery_size:
                raise ValueError(f"{split} needs more identities than the gallery size")
            for identity in names:
                rng = stable_rng(cfg.training.seed, f"vidtimit:{identity}")
                clips = people[identity]
                choices = [
                    valid[len(valid) // 2]
                    for (session, _), paths in clips.items()
                    if session == 1
                    and (valid := [path for path in paths if path in vectors])
                ]
                if len(choices) < cfg.enrollment_count:
                    raise ValueError(f"{identity} has too few usable enrollment clips")
                enrollment[identity] = sorted(rng.sample(choices, cfg.enrollment_count))
                alternatives = [name for name in names if name != identity]
                for session in (2, 3):
                    windows = list(self.probe_windows(clips, session, vectors))
                    rng.shuffle(windows)
                    if len(windows) < 2:
                        skipped_sessions.append(
                            {"identity": identity, "session": session}
                        )
                        continue
                    budget = min(cfg.windows_per_identity // 2, len(windows))
                    for index, (clip, window) in enumerate(
                        windows[: budget - budget % 2]
                    ):
                        enrolled = index % 2 == 0
                        gallery = rng.sample(
                            alternatives, cfg.gallery_size - int(enrolled)
                        )
                        if enrolled:
                            gallery.append(identity)
                        attempts.append(
                            {
                                "attempt_id": f"{split}:{identity}:{session}:{index}",
                                "split": split,
                                "identity": identity,
                                "session": session,
                                "clip": clip,
                                "frame_paths": window,
                                "gallery": sorted(gallery),
                                "is_enrolled": int(enrolled),
                                "gallery_sampling": (
                                    "hard"
                                    if split == "train" and index % 4 < 2
                                    else "random"
                                ),
                            }
                        )
        return pools, enrollment, attempts, skipped_sessions

    def mine_hard_attempts(self, people, names, enrollment, attempts, vectors, minimum):
        gallery_size = self.cfg.gallery_size
        templates = {
            name: gallery_template(
                np.stack([vectors[path] for path in enrollment[name]])
            )
            for name in names
        }
        matrix = np.stack([templates[name] for name in names])
        used = {path for row in attempts for path in row["frame_paths"]}
        for identity in names:
            for session in (2, 3):
                for clip, window in self.probe_windows(
                    people[identity], session, vectors
                ):
                    if any(path in used for path in window):
                        continue
                    frames = np.stack([vectors[path] for path in window])
                    similarities = unit_rows(frames).mean(axis=0) @ matrix.T
                    distractors = [
                        names[index]
                        for index in np.argsort(-similarities, kind="stable")
                        if names[index] != identity
                    ][:gallery_size]
                    unknown_gallery = sorted(distractors)
                    known_gallery = sorted([identity, *distractors[:-1]])
                    _, unknown = attempt_features(
                        frames, np.stack([templates[name] for name in unknown_gallery])
                    )
                    winner, _ = attempt_features(
                        frames, np.stack([templates[name] for name in known_gallery])
                    )
                    hard_unknown = unknown[0] >= minimum
                    if not hard_unknown and known_gallery[winner] == identity:
                        continue
                    start_frame = Path(window[0]).stem
                    for enrolled, gallery in enumerate(
                        (unknown_gallery, known_gallery)
                    ):
                        if not enrolled and not hard_unknown:
                            continue
                        yield {
                            "attempt_id": (
                                f"train:{identity}:{session}:{clip}:{start_frame}"
                                f":mined:{enrolled}"
                            ),
                            "split": "train",
                            "identity": identity,
                            "session": session,
                            "clip": clip,
                            "frame_paths": window,
                            "gallery": gallery,
                            "is_enrolled": enrolled,
                            "gallery_sampling": "mined_hard",
                        }

    def prepare(self) -> Path:
        cfg = self.cfg
        if cfg.dataset_dir.exists():
            raise FileExistsError(f"dataset already exists: {cfg.dataset_dir}")
        if cfg.windows_per_identity < 4 or cfg.windows_per_identity % 2:
            raise ValueError("VidTIMIT needs an even window budget of at least four")
        archives = sorted((cfg.root / "archives").glob("*.zip"))
        if len(archives) != 43:
            raise ValueError(
                f"expected 43 VidTIMIT subject archives, found {len(archives)}"
            )
        device = resolve_device(cfg.embedding_device)
        task, saved = load_task(cfg.encoder_checkpoint, device)
        preprocess = PreprocessConfig.model_validate(
            saved["metadata"]["preprocess_config"]
        )
        detector_path = cfg.vidtimit.detector_config
        detector_raw = read_yaml(detector_path)["detector"]
        detector_raw["model"] = (detector_path.parent / detector_raw["model"]).resolve()
        detector_config = DetectorConfig.model_validate(detector_raw)
        ensure_model(detector_config)
        worker = local()

        def face_box(frame):
            if not hasattr(worker, "detector"):
                worker.detector = YuNetDetector(detector_config)
            boxes = worker.detector.detect(frame)
            return boxes[0] if len(boxes) == 1 else None

        people = self.extract(archives)
        paths = [
            path
            for clips in people.values()
            for frames in clips.values()
            for path in frames
        ]
        vectors = self.embed(task.model.encoder, preprocess, paths, device, face_box)
        pools, enrollment, attempts, skipped_sessions = self.sample_sessions(
            people, vectors
        )
        mining = {}
        if (minimum := cfg.vidtimit.hard_mining_min_cosine) is not None:
            extra = list(
                self.mine_hard_attempts(
                    people, pools["train"], enrollment, attempts, vectors, minimum
                )
            )
            attempts.extend(extra)
            mining = {
                "hard_mining_min_cosine": minimum,
                "mined_train_attempts": len(extra),
            }
        return self.export(
            pools,
            enrollment,
            attempts,
            vectors,
            saved["epoch"],
            {
                "source": "vidtimit",
                "source_url": "https://zenodo.org/records/158963",
                "detector_sha256": detector_config.sha256,
                "frame_stride": cfg.vidtimit.frame_stride,
                "enrollment_session": 1,
                "probe_sessions": [2, 3],
                "frames_sampled": len(paths),
                "frames_skipped_detection": len(paths) - len(vectors),
                "probe_sessions_skipped": skipped_sessions,
                **mining,
                "archive_sha256": {path.name: digest(path) for path in archives},
            },
        )


if __name__ == "__main__":
    config = load_config(__doc__)
    if not isinstance(config, VidTIMITConfig):
        raise SystemExit("A VidTIMIT YAML configuration is required.")
    print(VidTIMITPreparer(config).prepare())
