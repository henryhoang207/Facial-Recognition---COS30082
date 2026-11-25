"""Shared gallery scoring and feature extraction for one attendance attempt."""

import numpy as np

FEATURES = (
    "cosine_mean",
    "cosine_min",
    "cosine_std",
    "margin_mean",
    "winner_fraction",
)
TOP2_FEATURES = (*FEATURES, "top2_gap")
VARIABLE_GALLERY_FEATURES = (*TOP2_FEATURES, "log2_gallery_size")


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("expected a nonempty matrix of finite embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("embedding norms must be positive and finite")
    return values / norms


def gallery_template(enrollment: np.ndarray) -> np.ndarray:
    return unit_rows(unit_rows(enrollment).mean(axis=0, keepdims=True))[0]


def attempt_features(
    frames: np.ndarray, templates: np.ndarray, *, include_top2_gap: bool = False,
    include_gallery_size: bool = False,
) -> tuple[int, np.ndarray]:
    frames, templates = unit_rows(frames), unit_rows(templates)
    if frames.shape[1] != templates.shape[1]:
        raise ValueError("gallery templates must match the frame dimension")
    return features_from_scores(
        np.clip(frames @ templates.T, -1, 1), include_top2_gap=include_top2_gap,
        include_gallery_size=include_gallery_size,
    )


def features_from_scores(
    scores: np.ndarray, *, include_top2_gap: bool = False,
    include_gallery_size: bool = False,
) -> tuple[int, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    if (
        scores.ndim != 2
        or not len(scores)
        or scores.shape[1] < (1 if include_gallery_size else 2)
        or not np.isfinite(scores).all()
        or (np.abs(scores) > 1).any()
    ):
        raise ValueError(
            "expected finite cosine scores for a supported gallery"
        )
    means = scores.mean(axis=0)
    candidate = int(means.argmax())
    chosen = scores[:, candidate]
    margin = gap = 0.0
    if len(means) > 1:
        competitors = np.delete(scores, candidate, axis=1).max(axis=1)
        margin = (chosen - competitors).mean()
        gap = means[candidate] - np.partition(means, -2)[-2]
    features = [
        chosen.mean(),
        chosen.min(),
        chosen.std(),
        margin,
        (scores.argmax(axis=1) == candidate).mean(),
    ]
    if include_top2_gap or include_gallery_size:
        features.append(gap)
    if include_gallery_size:
        features.append(np.log2(len(means)))
    return candidate, np.asarray(features, dtype=np.float32)
