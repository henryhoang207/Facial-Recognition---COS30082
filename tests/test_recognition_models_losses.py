"""Numerical and gradient checks for the shared recognition model and losses."""

import math

import pytest
import torch
from torch.nn import functional as F

from src.recognition.identification.losses import (
    ArcFaceHead,
    CurriculumGate,
    TripletMetricLoss,
)
from src.recognition.identification.models import IResNet34


def test_triplet_uses_squared_distances_and_all_directed_positives():
    embeddings = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    loss, stats = TripletMetricLoss(margin=0.2)(embeddings, labels)
    # The two hinges are max(.8 - 2 + .2, 0) and .8 - .4 + .2.
    assert loss.item() == pytest.approx(0.3, abs=1e-6)
    unsquared_result = (math.sqrt(0.8) - math.sqrt(0.4) + 0.2) / 2
    assert abs(loss.item() - unsquared_result) > 0.05
    assert stats["positive_pairs"] == stats["selected_triplets"] == 2
    assert stats["active_triplets"] == 1
    assert stats["mean_positive_distance"] == pytest.approx(0.8, abs=1e-6)
    assert stats["mean_negative_distance"] == pytest.approx(1.2, abs=1e-6)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_semi_hard_missing_pairs_remain_in_denominator():
    angles = torch.tensor([0.0, math.pi / 6, math.radians(70.0)])
    embeddings = torch.stack((angles.cos(), angles.sin()), dim=1).requires_grad_()
    labels = torch.tensor([0, 0, 1])
    loss, stats = TripletMetricLoss(margin=0.5)(
        embeddings, labels, strategy="semi_hard"
    )
    dap = float((embeddings[0] - embeddings[1]).square().sum().detach())
    dan = float((embeddings[1] - embeddings[2]).square().sum().detach())
    assert dap < dan < dap + 0.5
    assert stats["positive_pairs"] == 2
    assert stats["selected_triplets"] == stats["active_triplets"] == 1
    assert stats["candidate_coverage"] == 0.5
    assert loss.item() == pytest.approx((dap - dan + 0.5) / 2, abs=1e-6)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize("labels", [[0, 0, 1], [0, 0, 0], [0, 1, 2]])
def test_empty_triplet_batches_return_differentiable_zero(labels):
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    loss, stats = TripletMetricLoss(margin=0.2)(
        embeddings, torch.tensor(labels), strategy="semi_hard"
    )
    assert loss.dtype == torch.float32 and loss.item() == 0.0
    assert stats["selected_triplets"] == 0 and stats["empty_batch"] == 1
    loss.backward()
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_semi_hard_inequalities_exclude_both_endpoints():
    # Both anchors have squared d_ap=1. Available negatives lie exactly at
    # the lower and upper endpoints of the required open interval (1, 2).
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    _, stats = TripletMetricLoss(margin=1.0)(
        embeddings, torch.tensor([0, 0, 1]), strategy="semi_hard"
    )
    assert stats["selected_triplets"] == 0


def test_random_mining_is_reproducible_and_never_selects_same_identity():
    embeddings = torch.eye(8, requires_grad=True)
    labels = torch.arange(8) // 2
    criterion = TripletMetricLoss(margin=0.3)
    loss_a, stats_a = criterion(
        embeddings, labels, generator=torch.Generator().manual_seed(91)
    )
    loss_b, stats_b = criterion(
        embeddings, labels, generator=torch.Generator().manual_seed(91)
    )
    # Every distinct pair has squared distance two, so both runs have a
    # deterministic analytical loss even though sampled indices may vary.
    assert stats_a == stats_b
    assert torch.equal(loss_a, loss_b)
    assert stats_a["selected_triplets"] == stats_a["positive_pairs"] == 8
    assert loss_a.item() == pytest.approx(0.3, abs=1e-6)
    clustered = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    zero_loss, separated_stats = criterion(
        clustered,
        torch.tensor([0, 0, 1, 1]),
        generator=torch.Generator().manual_seed(91),
    )
    assert zero_loss.item() == 0.0
    assert separated_stats["mean_negative_distance"] == 4.0


def test_arcface_initialization_is_reproducible_and_preserves_rng():
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    head_a = ArcFaceHead(num_classes=5, embedding_dim=3, seed=91)
    assert torch.equal(before, torch.get_rng_state())
    head_b = ArcFaceHead(num_classes=5, embedding_dim=3, seed=91)
    head_c = ArcFaceHead(num_classes=5, embedding_dim=3, seed=92)
    assert torch.equal(head_a.weight, head_b.weight)
    assert not torch.equal(head_a.weight, head_c.weight)
    assert not hasattr(head_a, "bias")


def test_arcface_changes_only_target_logits_and_uses_monotonic_fallback():
    margin = 0.5
    head = ArcFaceHead(num_classes=3, embedding_dim=2, margin=margin, scale=64)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    embeddings = torch.tensor([[0.6, 0.8], [-0.99, 0.1]], requires_grad=True)
    labels = torch.tensor([0, 0])
    logits = head(embeddings, labels)
    cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(head.weight, dim=1))
    cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
    assert torch.allclose(logits[:, 1:], 64 * cosine[:, 1:])
    assert logits[0, 0].item() == pytest.approx(
        64 * math.cos(math.acos(cosine[0, 0].item()) + margin), abs=1e-5
    )
    assert cosine[1, 0].item() <= math.cos(math.pi - margin)
    assert logits[1, 0].item() == pytest.approx(
        64 * (cosine[1, 0].item() - math.sin(math.pi - margin) * margin), abs=1e-5
    )


def test_arcface_boundary_gradients_and_autocast_are_finite():
    head = ArcFaceHead(num_classes=3, embedding_dim=2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    angles = torch.tensor([0.0, math.pi, math.pi - 0.5 - 1e-4, math.pi - 0.5 + 1e-4])
    embeddings = torch.stack((angles.cos(), angles.sin()), dim=1).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = head(embeddings, torch.zeros(4, dtype=torch.long))
        loss = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(head.weight.grad).all()


def test_curriculum_gate_is_one_way_and_resumable():
    gate = CurriculumGate()
    passing = {"ranking_rate": 0.75, "mean_gap": 0.05, "feature_spread": 0.002}
    assert not gate.update(1, passing)
    assert not gate.update(2, passing)
    # Equality does not satisfy the strict spread threshold.
    assert not gate.update(3, dict(passing, feature_spread=0.001))
    assert not gate.update(4, passing)
    assert not gate.update(5, passing)
    restored = CurriculumGate()
    restored.load_state_dict(gate.state_dict())
    assert restored.update(6, passing)
    assert restored.strategy == "semi_hard" and restored.switch_epoch == 6
    assert not restored.update(7, {"q": 0.0, "g": 0.0, "v": 0.0})
    with pytest.raises(ValueError, match="increasing"):
        restored.update(7, passing)


def test_iresnet34_normalization_fp32_head_and_backbone_gradient():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        model = IResNet34(embedding_dim=256, dropout=0.0)
        assert [
            len(model.layer1),
            len(model.layer2),
            len(model.layer3),
            len(model.layer4),
        ] == [3, 4, 6, 3]
        assert model.fc.in_features == 512 * 7 * 7
        assert model.features.affine and not model.features.weight.requires_grad
        assert model.features.bias.requires_grad
        assert torch.equal(model.features.weight, torch.ones(256))
        images = torch.randn(2, 3, 112, 112)
        output = model(images)
        assert output.shape == (2, 256) and output.dtype == torch.float32
        assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)
        (output * torch.randn_like(output)).sum().backward()
        assert model.conv1.weight.grad is not None
        assert torch.isfinite(model.conv1.weight.grad).all()
        assert model.features.weight.grad is None
        assert model.features.bias.grad is not None
        model.zero_grad(set_to_none=True)
        model.eval()
        head_dtypes = []
        handle = model.fc.register_forward_pre_hook(
            lambda module, args: head_dtypes.append(args[0].dtype)
        )
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            features = model.forward_features(images)
            output = model(images)
        handle.remove()
        assert features.dtype == output.dtype == torch.float32
        assert head_dtypes == [torch.float32, torch.float32]
        assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triplet_mining_accepts_a_matching_cuda_generator():
    embeddings = torch.eye(8, device="cuda", requires_grad=True)
    labels = torch.arange(8, device="cuda") // 2
    generator = torch.Generator(device="cuda").manual_seed(91)
    loss, stats = TripletMetricLoss(margin=0.3)(embeddings, labels, generator=generator)
    assert loss.item() == pytest.approx(0.3, abs=1e-6)
    assert stats["selected_triplets"] == 8
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
