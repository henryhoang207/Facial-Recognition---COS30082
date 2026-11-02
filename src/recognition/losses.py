"""FP32 ArcFace logits, online Triplet mining, and a resumable curriculum gate."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_batch(embeddings: Tensor, labels: Tensor) -> None:
    if embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be a two-dimensional floating-point tensor")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must contain one identity index per embedding")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("labels must contain integer identity indices")


class ArcFaceHead(nn.Module):
    """Full-class ArcFace logits with a non-easy monotonic target fallback.

    The angular margin is measured in radians. The head has no bias and is
    discarded for verification on unseen identities. Its Xavier initialization
    uses an isolated CPU RNG without changing the caller's CPU or CUDA RNG.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        margin: float = 0.5,
        scale: float = 64.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_classes < 2 or embedding_dim < 1:
            raise ValueError("At least two classes and a positive embedding_dim are required")
        if not math.isfinite(margin) or not 0.0 <= margin < math.pi / 2:
            raise ValueError("margin must be finite and in [0, pi/2) radians")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)
        self.scale = float(scale)
        self.easy_margin = False
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.sinmm = math.sin(math.pi - self.margin) * self.margin
        with torch.random.fork_rng(devices=[]):
            local_generator = torch.Generator(device="cpu").manual_seed(int(seed))
            torch.set_rng_state(local_generator.get_state())
            weight = torch.empty(
                self.num_classes, self.embedding_dim, device="cpu", dtype=torch.float32
            )
            nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Return FP32 scaled logits; only the labeled class receives a margin."""
        _check_batch(embeddings, labels)
        if embeddings.shape[1] != self.embedding_dim or embeddings.shape[0] == 0:
            raise ValueError("Unexpected embedding shape for the ArcFace head")
        labels = labels.to(device=embeddings.device, dtype=torch.long)
        if torch.any((labels < 0) | (labels >= self.num_classes)).item():
            raise ValueError("ArcFace labels must index the training identity head")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            cosine = F.linear(
                F.normalize(embeddings.float(), dim=1, eps=1e-12),
                F.normalize(self.weight.float(), dim=1, eps=1e-12),
            ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            target = cosine.gather(1, labels[:, None])
            sine = (1.0 - target.square()).clamp_min(1e-7).sqrt()
            angular = target * self.cos_m - sine * self.sin_m
            target_with_margin = torch.where(
                target > self.threshold, angular, target - self.sinmm
            )
            logits = cosine.scatter(1, labels[:, None], target_with_margin)
            return logits * self.scale


class TripletMetricLoss(nn.Module):
    """Squared-L2 hinge loss with one sampled negative per directed A/P pair.

    Embeddings are expected to be normalized by the encoder. Every directed
    same-identity pair of distinct images remains in the denominator, including
    pairs with no eligible semi-hard negative. Mining is detached from autograd;
    distances for selected triplets are recomputed with gradients in FP32.
    Distance diagnostics also report squared distances, never Euclidean norms.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be finite and positive")
        self.margin = float(margin)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        strategy: str = "random",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, float | int]]:
        """Return the loss and Python-valued mining diagnostics."""
        _check_batch(embeddings, labels)
        if strategy not in {"random", "semi_hard"}:
            raise ValueError("strategy must be 'random' or 'semi_hard'")
        labels = labels.to(device=embeddings.device)
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            features = embeddings.float()
            with torch.no_grad():
                squared_norm = features.detach().square().sum(dim=1)
                distances = (
                    squared_norm[:, None]
                    + squared_norm[None, :]
                    - 2.0 * (features.detach() @ features.detach().T)
                ).clamp_min(0.0)
                same_identity = labels[:, None] == labels[None, :]
                positive_mask = same_identity & ~torch.eye(
                    len(labels), dtype=torch.bool, device=features.device
                )
                anchors, positives = positive_mask.nonzero(as_tuple=True)
                positive_pairs = anchors.numel()
                candidates = ~same_identity[anchors]
                positive_distances = distances[anchors, positives]
                if strategy == "semi_hard":
                    anchor_distances = distances[anchors]
                    candidates &= (
                        (anchor_distances > positive_distances[:, None])
                        & (anchor_distances < positive_distances[:, None] + self.margin)
                    )
                eligible_rows = candidates.any(dim=1).nonzero(as_tuple=True)[0]
                selected = eligible_rows.numel()
                if selected:
                    sampling_weights = candidates[eligible_rows].float()
                    sampling_device = (
                        sampling_weights.device if generator is None else generator.device
                    )
                    negatives = torch.multinomial(
                        sampling_weights.to(sampling_device), 1, generator=generator
                    ).flatten().to(features.device)
                    chosen_anchors = anchors[eligible_rows]
                    chosen_positives = positives[eligible_rows]

            if selected:
                positive_squared = (
                    features[chosen_anchors] - features[chosen_positives]
                ).square().sum(dim=1)
                negative_squared = (
                    features[chosen_anchors] - features[negatives]
                ).square().sum(dim=1)
                hinge = F.relu(positive_squared - negative_squared + self.margin)
                loss = hinge.sum() / positive_pairs
                active = int((hinge.detach() > 0).sum().item())
                mean_negative = float(negative_squared.detach().mean().item())
            else:
                loss = features.sum() * 0.0
                active = 0
                mean_negative = 0.0

            stats: dict[str, float | int] = {
                "positive_pairs": positive_pairs,
                "selected_triplets": selected,
                "active_triplets": active,
                "empty_batch": int(selected == 0),
                "zero_loss_batch": int(active == 0),
                "mean_positive_distance": (
                    float(positive_distances.mean().item()) if positive_pairs else 0.0
                ),
                "mean_negative_distance": mean_negative,
                "candidate_coverage": selected / positive_pairs if positive_pairs else 0.0,
            }
        return loss, stats


class CurriculumGate:
    """Switch once from random to semi-hard mining using training-only probes.

    Epochs are completed, one-based epoch counts. Checks from earlier random
    epochs count toward the consecutive-check requirement; the switch cannot
    occur before ``min_random_epochs``. Thresholds are engineering heuristics.
    """

    def __init__(
        self,
        min_random_epochs: int = 5,
        consecutive_checks: int = 3,
        ranking_threshold: float = 0.75,
        gap_threshold: float = 0.05,
        spread_threshold: float = 0.001,
    ) -> None:
        if min_random_epochs < 1 or consecutive_checks < 1:
            raise ValueError("Epoch and consecutive-check limits must be positive")
        if not 0.0 <= ranking_threshold <= 1.0:
            raise ValueError("ranking_threshold must be in [0, 1]")
        if not all(math.isfinite(value) for value in (gap_threshold, spread_threshold)):
            raise ValueError("Curriculum thresholds must be finite")
        self.min_random_epochs = int(min_random_epochs)
        self.consecutive_checks = int(consecutive_checks)
        self.ranking_threshold = float(ranking_threshold)
        self.gap_threshold = float(gap_threshold)
        self.spread_threshold = float(spread_threshold)
        self.strategy = "random"
        self.consecutive_passes = 0
        self.last_epoch = 0
        self.switch_epoch: int | None = None

    def update(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        """Return True only when this check triggers the permanent switch."""
        if epoch <= self.last_epoch:
            raise ValueError("Curriculum checks must use increasing completed epoch counts")
        self.last_epoch = int(epoch)
        if self.strategy == "semi_hard":
            return False

        def metric(*names: str) -> float:
            for name in names:
                if name in metrics:
                    return float(metrics[name])
            raise KeyError(names[0])

        ranking = metric("q", "ranking_rate", "ranking_accuracy")
        gap = metric("g", "mean_gap")
        spread = metric("v", "feature_spread", "embedding_spread")
        passed = (
            all(math.isfinite(value) for value in (ranking, gap, spread))
            and ranking >= self.ranking_threshold
            and gap >= self.gap_threshold
            and spread > self.spread_threshold
        )
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if epoch >= self.min_random_epochs and self.consecutive_passes >= self.consecutive_checks:
            self.strategy = "semi_hard"
            self.switch_epoch = int(epoch)
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        """Return JSON- and checkpoint-compatible configuration and progress."""
        return dict(vars(self))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore both thresholds and progress without re-triggering a switch."""
        restored = type(self)(
            min_random_epochs=state["min_random_epochs"],
            consecutive_checks=state["consecutive_checks"],
            ranking_threshold=state["ranking_threshold"],
            gap_threshold=state["gap_threshold"],
            spread_threshold=state["spread_threshold"],
        )
        if state["strategy"] not in {"random", "semi_hard"}:
            raise ValueError("Unknown saved curriculum strategy")
        restored.strategy = state["strategy"]
        restored.consecutive_passes = int(state["consecutive_passes"])
        restored.last_epoch = int(state["last_epoch"])
        restored.switch_epoch = state["switch_epoch"]
        self.__dict__.update(vars(restored))
