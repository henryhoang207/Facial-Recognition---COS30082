"""Embedding evaluation, conservative calibration, and reproducible plot artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embeddings must be finite one-dimensional vectors.")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("A zero embedding cannot define cosine similarity.")
    return vector / norm


def _labels(values) -> np.ndarray:
    """Read booleans without treating the string 'False' as truthy."""
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = []
    for value in values:
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError("is_same must contain booleans or 0/1 values.")
        result.append(mapping[key])
    return np.asarray(result, dtype=bool)


def _worker_init(worker_id: int) -> None:
    """Keep evaluation workers small and compatible with Windows spawn."""
    import random

    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def extract_embeddings(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    data_root: Path,
    device,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Evaluate each path once, return unit FP32 vectors, and restore module modes.

    The model must already be on ``device``. FaceDataset supplies the shared,
    deterministic evaluation crop; no identity classifier is called here.
    """
    from .data import FaceDataset

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if not {"rel_path", "identity_id"}.issubset(frame.columns):
        raise ValueError("frame must contain rel_path and identity_id.")
    unique = frame.drop_duplicates("rel_path").copy().reset_index(drop=True)
    if "label" not in unique:
        unique["label"] = pd.factorize(unique["identity_id"], sort=True)[0]
    if unique.empty:
        return {}
    dataset = FaceDataset(unique, Path(data_root), train=False, seed=42)
    device = torch.device(device)
    options = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    if num_workers:
        options.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_worker_init,
        )
    loader = DataLoader(dataset, **options)
    modes = [(module, module.training) for module in model.modules()]
    embeddings = {}
    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=device.type == "cuda")
                features = model(images).float()
                if features.ndim != 2 or features.shape[0] != images.shape[0]:
                    raise ValueError("The encoder must return a [batch, dimension] tensor.")
                norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
                if not torch.isfinite(features).all() or torch.any(norms <= 1e-12):
                    raise ValueError("The encoder produced a non-finite or zero embedding.")
                features = (features / norms).cpu().numpy().astype(np.float32, copy=False)
                embeddings.update(zip(map(str, batch["rel_path"]), features, strict=True))
    finally:
        # Restore mixed modes too, such as frozen BatchNorm inside a train-mode model.
        for module, training in modes:
            module.training = training
    return embeddings


def pair_scores(
    embeddings: Mapping[str, np.ndarray], pairs: pd.DataFrame
) -> np.ndarray:
    """Return cosine scores in pair-manifest order; larger means more similar."""
    required = {"path_a", "path_b", "is_same"}
    if not required.issubset(pairs.columns) or pairs.empty:
        raise ValueError("A nonempty pair manifest needs path_a, path_b, and is_same.")
    labels = _labels(pairs["is_same"])
    if {"identity_a", "identity_b"}.issubset(pairs.columns):
        same_identity = pairs["identity_a"].astype(str).to_numpy() == pairs[
            "identity_b"
        ].astype(str).to_numpy()
        if not np.array_equal(labels, same_identity):
            raise ValueError("Pair labels disagree with identity_a/identity_b.")
    paths = set(pairs["path_a"].astype(str)) | set(pairs["path_b"].astype(str))
    missing = paths.difference(embeddings)
    if missing:
        raise KeyError(f"Missing embeddings for {len(missing)} paths; first: {min(missing)}")
    unit = {path: _unit_vector(embeddings[path]) for path in paths}
    scores = [
        float(np.dot(unit[str(a)], unit[str(b)]))
        for a, b in zip(pairs["path_a"], pairs["path_b"], strict=True)
    ]
    return np.clip(np.asarray(scores, dtype=np.float64), -1.0, 1.0)


def _operating_curve(
    scores: np.ndarray, labels: np.ndarray, positive_success: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Enumerate realizable score>=threshold points, preserving tied scores."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must be matching finite one-dimensional arrays.")
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        raise ValueError("Evaluation requires both genuine and impostor observations.")
    successful = labels if positive_success is None else labels & positive_success
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(scores) - 1]
    thresholds = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    return {
        "thresholds": thresholds,
        "fmr": np.r_[0.0, np.cumsum((~labels)[order])[ends] / negatives],
        "tar": np.r_[0.0, np.cumsum(successful[order])[ends] / positives],
    }


def _calibration_index(curve: dict[str, np.ndarray], target: float) -> int:
    if not 0 <= target <= 1:
        raise ValueError("The target error rate must be between zero and one.")
    allowed = np.flatnonzero(curve["fmr"] <= target)
    # Maximize true accepts, then minimize false accepts, then prefer higher tau.
    return int(
        max(
            allowed,
            key=lambda i: (
                curve["tar"][i],
                -curve["fmr"][i],
                curve["thresholds"][i],
            ),
        )
    )


def roc_points(scores: np.ndarray, is_same) -> dict[str, np.ndarray]:
    """Return attainable ROC arrays; no interpolation is used for calibration."""
    return _operating_curve(scores, _labels(is_same))


def _eer(curve: dict[str, np.ndarray]) -> float:
    fmr, fnmr = curve["fmr"], 1.0 - curve["tar"]
    delta = fmr - fnmr
    right = int(np.flatnonzero(delta >= 0)[0])
    if right == 0 or delta[right] == 0:
        return float(fmr[right])
    left = right - 1
    fraction = -delta[left] / (delta[right] - delta[left])
    return float(fmr[left] + fraction * (fmr[right] - fmr[left]))


def verification_metrics(
    embeddings: Mapping[str, np.ndarray],
    pairs: pd.DataFrame,
    target_fmr: float = 0.001,
    threshold: float | None = None,
) -> dict:
    """Compute verification metrics, optionally at an unchanged dev threshold.

    ``tar_at_fmr`` always describes the supplied pair set's attainable ROC.
    ``tar`` and ``fmr`` describe the returned operating ``threshold``. Supplying a
    threshold never recalibrates it, including when test FMR exceeds the target.
    EER is linearly interpolated only as a descriptive summary of the ROC.
    """
    scores = pair_scores(embeddings, pairs)
    labels = _labels(pairs["is_same"])
    curve = _operating_curve(scores, labels)
    index = _calibration_index(curve, target_fmr)
    tau = float(curve["thresholds"][index] if threshold is None else threshold)
    if not np.isfinite(tau):
        raise ValueError("The operating threshold must be finite.")
    accepted = scores >= tau
    return {
        "auc": float(np.trapezoid(curve["tar"], curve["fmr"])),
        "eer": _eer(curve),
        "tar_at_fmr": float(curve["tar"][index]),
        "target_fmr": float(target_fmr),
        "fmr": float(accepted[~labels].mean()),
        "tar": float(accepted[labels].mean()),
        "threshold": tau,
        "threshold_source": "calibrated_on_supplied_pairs" if threshold is None else "fixed",
        "genuine_pairs": int(labels.sum()),
        "impostor_pairs": int((~labels).sum()),
    }


def evaluate_probe(
    model: torch.nn.Module,
    train_frame: pd.DataFrame,
    probe_triplets: pd.DataFrame,
    data_root: Path,
    device,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict:
    """Measure the fixed train-only curriculum probe without changing model state."""
    columns = ["anchor", "positive", "negative"]
    if not set(columns).issubset(probe_triplets.columns) or probe_triplets.empty:
        raise ValueError("Probe triplets need nonempty anchor/positive/negative columns.")
    paths = set(probe_triplets[columns].astype(str).to_numpy().ravel())
    known = train_frame.drop_duplicates("rel_path").set_index("rel_path", drop=False)
    if not paths.issubset(known.index):
        raise ValueError("All curriculum probe images must belong to the training manifest.")
    for anchor, positive, negative in probe_triplets[columns].itertuples(index=False, name=None):
        if anchor == positive or known.loc[anchor, "identity_id"] != known.loc[positive, "identity_id"]:
            raise ValueError("Probe anchor/positive must be different images of the same identity.")
        if known.loc[anchor, "identity_id"] == known.loc[negative, "identity_id"]:
            raise ValueError("Probe negatives must have a different identity.")
    selected = known.loc[sorted(paths)].reset_index(drop=True)
    embeddings = extract_embeddings(model, selected, data_root, device, batch_size, num_workers)
    anchors = np.stack([embeddings[str(p)] for p in probe_triplets["anchor"]])
    positives = np.stack([embeddings[str(p)] for p in probe_triplets["positive"]])
    negatives = np.stack([embeddings[str(p)] for p in probe_triplets["negative"]])
    dap = np.square(anchors - positives).sum(axis=1, dtype=np.float32)
    dan = np.square(anchors - negatives).sum(axis=1, dtype=np.float32)
    unique = np.stack([embeddings[p] for p in sorted(paths)])
    return {
        "ranking_rate": float((dap < dan).mean()),
        "mean_gap": float((dan - dap).mean()),
        "feature_spread": float(unique.std(axis=0, ddof=0).mean()),
        "triplets": int(len(probe_triplets)),
        "unique_images": int(len(paths)),
    }


def create_open_set_manifest(
    frame: pd.DataFrame, seed: int = 42, enrolled: int = 30, enrollment: int = 3
) -> dict:
    """Choose a fixed gallery and separate genuine/unknown probes from held-out IDs."""
    if not {"rel_path", "identity_id"}.issubset(frame.columns):
        raise ValueError("The identification manifest needs rel_path and identity_id.")
    if enrolled < 1 or enrollment < 1:
        raise ValueError("enrolled and enrollment must be positive.")
    rows = frame.drop_duplicates("rel_path").copy()
    rows["identity_id"] = rows["identity_id"].astype(str)
    rows["rel_path"] = rows["rel_path"].astype(str)
    identities = sorted(rows["identity_id"].unique())
    if len(identities) <= enrolled:
        raise ValueError("An open-set protocol needs at least one unknown identity.")
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(identities, size=enrolled, replace=False).tolist())
    gallery, probes = [], []
    for identity, group in rows.groupby("identity_id", sort=True):
        group = group.sort_values("rel_path")
        # Existing manifests deduplicate exact files; preserve any supplied group too.
        group_column = next(
            (column for column in ("duplicate_group", "sha256") if column in group), None
        )
        candidates = group.drop_duplicates(group_column) if group_column else group
        if identity in selected:
            if len(candidates) <= enrollment:
                raise ValueError(f"Identity {identity} needs enrollment images and a genuine probe.")
            chosen = candidates.iloc[rng.choice(len(candidates), enrollment, replace=False)]
            paths = sorted(chosen["rel_path"].tolist())
            gallery.append({"identity_id": identity, "paths": paths})
            remaining = group[~group["rel_path"].isin(paths)]
            if group_column:
                remaining = remaining[~remaining[group_column].isin(chosen[group_column])]
        else:
            remaining = group
        probes.extend(
            {
                "rel_path": path,
                "identity_id": identity,
                "is_enrolled": identity in selected,
            }
            for path in remaining["rel_path"]
        )
    return {"seed": int(seed), "gallery": gallery, "probes": probes}


def evaluate_identification(
    embeddings: Mapping[str, np.ndarray],
    frame: pd.DataFrame | None = None,
    *,
    protocol: dict | None = None,
    seed: int = 42,
    enrolled: int = 30,
    enrollment: int = 3,
    target_fpir: float = 0.01,
    threshold: float | None = None,
) -> dict:
    """Evaluate rank-1 identification and unknown-search FPIR, not pair-level FMR."""
    if protocol is None:
        if frame is None:
            raise ValueError("Supply a frame or a saved identification protocol.")
        protocol = create_open_set_manifest(frame, seed, enrolled, enrollment)
    identities = [str(item["identity_id"]) for item in protocol["gallery"]]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("The gallery must contain distinct identities.")
    gallery_paths = {str(p) for item in protocol["gallery"] for p in item["paths"]}
    probes = protocol["probes"]
    if gallery_paths.intersection(str(item["rel_path"]) for item in probes):
        raise ValueError("Enrollment images must never also be probes.")
    templates = np.stack(
        [
            _unit_vector(np.mean([_unit_vector(embeddings[p]) for p in item["paths"]], axis=0))
            for item in protocol["gallery"]
        ]
    )
    if not probes:
        raise ValueError("The protocol needs genuine and unknown probes.")
    known = np.asarray([str(item["identity_id"]) in identities for item in probes])
    flags = np.asarray([bool(item["is_enrolled"]) for item in probes])
    if not np.array_equal(flags, known):
        raise ValueError("Probe enrollment flags disagree with gallery identities.")
    best_scores, correct = [], []
    # Chunk score matrices so evaluation does not grow with all probes times gallery.
    for start in range(0, len(probes), 2048):
        chunk = probes[start : start + 2048]
        vectors = np.stack([_unit_vector(embeddings[str(item["rel_path"])]) for item in chunk])
        scores = np.clip(vectors @ templates.T, -1.0, 1.0)
        winners = scores.argmax(axis=1)
        best_scores.extend(scores[np.arange(len(chunk)), winners].tolist())
        correct.extend(
            identities[int(index)] == str(item["identity_id"])
            for index, item in zip(winners, chunk, strict=True)
        )
    best_scores, correct = np.asarray(best_scores), np.asarray(correct, dtype=bool)
    curve = _operating_curve(best_scores, known, positive_success=correct)
    index = _calibration_index(curve, target_fpir)
    tau = float(curve["thresholds"][index] if threshold is None else threshold)
    if not np.isfinite(tau):
        raise ValueError("The operating threshold must be finite.")
    accepted = best_scores >= tau
    tpir = float((accepted & correct)[known].mean())
    return {
        "rank1": float(correct[known].mean()),
        "tpir": tpir,
        "dir": tpir,
        "fpir": float(accepted[~known].mean()),
        "threshold": tau,
        "threshold_source": "calibrated_on_supplied_probes" if threshold is None else "fixed",
        "target_fpir": float(target_fpir),
        "tpir_at_fpir": float(curve["tar"][index]),
        "enrolled_identities": int(len(identities)),
        "unknown_identities": int(len({str(p["identity_id"]) for p in probes if not p["is_enrolled"]})),
        "enrollment_images": int(len(gallery_paths)),
        "genuine_probes": int(known.sum()),
        "unknown_probes": int((~known).sum()),
        "rank_tie_policy": "first gallery identity in saved protocol order",
        "protocol": protocol,
    }


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _save_figure(figure, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return str(path)


def save_training_graph(history: pd.DataFrame | str | Path, output_path: str | Path) -> str:
    """Save complete loss and dev curves without requiring an interactive backend."""
    from matplotlib.figure import Figure

    history = _read_frame(history)
    if history.empty:
        raise ValueError("Training history cannot be empty.")
    figure = Figure(figsize=(10, 9))
    axes = figure.subplots(3, 1, sharex=True)
    x = history["epoch"] if "epoch" in history else np.arange(1, len(history) + 1)
    choices = [
        (("train_loss", "loss"), "Training loss"),
        (("dev_tar_at_fmr", "dev_tar", "tar_at_fmr"), "Dev TAR at target FMR"),
        (("dev_auc", "auc"), "Dev ROC AUC"),
    ]
    for axis, (columns, title) in zip(axes, choices, strict=True):
        column = next((name for name in columns if name in history), None)
        if column is not None:
            axis.plot(x, history[column], marker=".", linewidth=1.4)
        else:
            axis.text(0.5, 0.5, "Metric not recorded", transform=axis.transAxes, ha="center")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        stage_column = "strategy" if "strategy" in history else "stage"
        if stage_column in history:
            switched = history[stage_column].astype(str).ne(history[stage_column].astype(str).shift())
            for position in np.flatnonzero(switched.to_numpy())[1:]:
                axis.axvline(np.asarray(x)[position], color="gray", linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("Epoch")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def save_verification_graphs(
    embeddings: Mapping[str, np.ndarray],
    pairs: pd.DataFrame,
    output_dir: str | Path,
    threshold: float | None = None,
    target_fmr: float = 0.001,
) -> dict:
    """Save ROC, score distributions, and JSON metrics for the supplied split only."""
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, labels = pair_scores(embeddings, pairs), _labels(pairs["is_same"])
    curve = _operating_curve(scores, labels)
    metrics = verification_metrics(embeddings, pairs, target_fmr, threshold)
    figure = Figure(figsize=(7, 5))
    axis = figure.subplots()
    minimum = min(1.0 / metrics["impostor_pairs"], max(target_fmr / 10, 1e-8))
    axis.semilogx(np.maximum(curve["fmr"], minimum), curve["tar"], label=f"AUC={metrics['auc']:.4f}")
    axis.scatter(max(metrics["fmr"], minimum), metrics["tar"], label="Operating threshold", zorder=3)
    axis.set(xlabel="False match rate (FMR)", ylabel="True accept rate (TAR)", xlim=(minimum, 1), ylim=(0, 1.01))
    axis.grid(alpha=0.25)
    axis.legend()
    roc_path = _save_figure(figure, output_dir / "roc.png")
    figure = Figure(figsize=(7, 5))
    axis = figure.subplots()
    bins = np.linspace(-1, 1, 61)
    axis.hist(scores[~labels], bins=bins, density=True, alpha=0.55, label="Impostor")
    axis.hist(scores[labels], bins=bins, density=True, alpha=0.55, label="Genuine")
    axis.axvline(metrics["threshold"], linestyle="--", color="black", label="Operating threshold")
    axis.set(xlabel="Cosine similarity", ylabel="Density")
    axis.legend()
    distribution_path = _save_figure(figure, output_dir / "score_distribution.png")
    metrics_path = output_dir / "verification_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"metrics": metrics, "roc": roc_path, "score_distribution": distribution_path, "metrics_path": str(metrics_path)}


def plot_sweep(sweep: pd.DataFrame | str | Path, output_path: str | Path) -> str:
    """Plot dev-only margin screening; never rank configurations using test scores."""
    from matplotlib.figure import Figure

    sweep = _read_frame(sweep)
    if "stage" in sweep and (sweep["stage"] == "screening").any():
        sweep = sweep[sweep["stage"] == "screening"]
    if sweep.empty or "margin" not in sweep:
        raise ValueError("A nonempty sweep needs a margin column.")
    metric = next(
        (name for name in ("dev_tar_at_fmr", "best_dev_tar_at_fmr", "dev_tar") if name in sweep), None
    )
    if metric is None:
        raise ValueError("The sweep needs a dev TAR column.")
    family = next((name for name in ("family", "objective", "model", "method") if name in sweep), None)
    groups = list(sweep.groupby(family, sort=True)) if family else [("Configurations", sweep)]
    figure = Figure(figsize=(6 * len(groups), 5))
    axes = np.atleast_1d(figure.subplots(1, len(groups)))
    for axis, (name, group) in zip(axes, groups, strict=True):
        summary = group.groupby("margin")[metric].agg(["mean", "std"]).sort_index()
        axis.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), marker="o", capsize=4)
        axis.set(title=str(name), xlabel="Margin (objective-specific units)", ylabel="Dev TAR at target FMR")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def subject_bootstrap_ci(embeddings, pairs: pd.DataFrame, threshold: float,
                         n_bootstrap: int = 2000, confidence: float = 0.95,
                         seed: int = 42) -> dict:
    """Bootstrap identity clusters at a fixed operating threshold.

    Genuine comparisons receive the resampled subject weight; impostors receive
    the product of the weights at both identity endpoints. The interval conditions
    on this trained model, observed pairs, and the unchanged dev threshold.
    """
    if n_bootstrap < 2 or not 0 < confidence < 1 or not np.isfinite(threshold):
        raise ValueError("Invalid bootstrap count, confidence, or fixed threshold.")
    if not {"identity_a", "identity_b"}.issubset(pairs):
        raise ValueError("Subject bootstrap requires both pair identity columns.")
    scores = pair_scores(embeddings, pairs)
    labels = _labels(pairs.is_same)
    identities = sorted(set(pairs.identity_a) | set(pairs.identity_b))
    lookup = {identity: index for index, identity in enumerate(identities)}
    frame = pairs[["identity_a", "identity_b"]].copy()
    frame["accepted"] = scores >= threshold
    grouped = frame.groupby(["identity_a", "identity_b"]).accepted.agg(["sum", "size"]).reset_index()
    first = grouped.identity_a.map(lookup).to_numpy()
    second = grouped.identity_b.map(lookup).to_numpy()
    genuine = first == second
    accepted, counts = grouped["sum"].to_numpy(), grouped["size"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        weights = rng.multinomial(len(identities), np.full(len(identities), 1/len(identities)))
        pair_weight = np.where(genuine, weights[first], weights[first] * weights[second])
        denominator = [np.dot(pair_weight[mask], counts[mask]) for mask in (genuine, ~genuine)]
        if min(denominator) == 0:
            continue
        samples.append([np.dot(pair_weight[mask], accepted[mask])/total
                        for mask, total in zip((genuine, ~genuine), denominator)])
    if len(samples) < 2:
        raise ValueError("Insufficient valid subject-bootstrap replicates.")
    alpha = (1-confidence)/2
    bounds = np.quantile(np.asarray(samples), [alpha, 1-alpha], axis=0)
    operating = scores >= threshold
    return {"confidence": confidence, "requested_replicates": n_bootstrap,
            "valid_replicates": len(samples), "seed": seed, "fixed_threshold": float(threshold),
            "identity_count": len(identities),
            "tar": {"estimate": float(operating[labels].mean()), "lower": float(bounds[0, 0]), "upper": float(bounds[1, 0])},
            "fmr": {"estimate": float(operating[~labels].mean()), "lower": float(bounds[0, 1]), "upper": float(bounds[1, 1])},
            "method": "identity resampling with multiplicative impostor endpoint weights; threshold not recalibrated"}
