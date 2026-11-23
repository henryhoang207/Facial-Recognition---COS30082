"""Regression coverage for the Emotion checkpoint architectures."""

import cv2
import numpy as np
import pytest
import timm
import torch
import torchvision
from torch import nn

from app.backend.models.emotion import EmotionGRUPredictor


@pytest.mark.parametrize(
    ("model_name", "build_backbone", "feature_dim", "legacy"),
    [
        (
            "mobilenet_v3_large_gru",
            lambda: torchvision.models.mobilenet_v3_large(weights=None).features,
            960,
            False,
        ),
        (
            "mobilenetv3_gru",
            lambda: torchvision.models.mobilenet_v3_large(weights=None).features,
            960,
            True,
        ),
        (
            "mobilenetv4_conv_small_gru",
            lambda: timm.create_model(
                "mobilenetv4_conv_small", pretrained=False, num_classes=0
            ),
            1280,
            False,
        ),
        (
            "efficientnet_b0_gru",
            lambda: torchvision.models.efficientnet_b0(weights=None).features,
            1280,
            False,
        ),
    ],
)
def test_emotion_checkpoint_streaming_matches_clip(
    tmp_path, model_name, build_backbone, feature_dim, legacy
):
    labels = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear"]
    reference = nn.ModuleDict(
        {
            "backbone": build_backbone(),
            "gru": nn.GRU(feature_dim, 128, batch_first=True),
            "head": nn.Sequential(nn.Dropout(0.3), nn.Linear(128, len(labels))),
        }
    ).eval()
    config = {
        "model": model_name,
        "class_names": labels,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }
    if not legacy:
        config.update(task="emotion", image_size=64)
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": config, "model": reference.state_dict()}, path)

    predictor = EmotionGRUPredictor(path)
    size = predictor.img_size
    frames = np.random.default_rng(42).integers(
        0, 256, (3, size, size, 3), dtype=np.uint8
    )
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        features = reference["backbone"](images)
        if model_name != "mobilenetv4_conv_small_gru":
            features = features.mean(dim=(2, 3))
        output, _ = reference["gru"](features.unsqueeze(0))
        expected = reference["head"](output)[0].softmax(dim=-1).numpy()

    for index, frame in enumerate(frames):
        label, scores = predictor.predict_track(1, frame)
        assert list(scores) == labels
        np.testing.assert_allclose(list(scores.values()), expected[index], atol=1e-6)
        assert label == labels[expected[index].argmax()]
    predictor.drop_track(1)
    assert predictor.predict_track(1, frames[-1]) == predictor.predict(frames[-1])
