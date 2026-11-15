import cv2
import numpy as np
import pytest
import torch
from torch import nn

from app.backend.models.emotion import EmotionGRUPredictor


def test_mobilenet_v4_checkpoint_streaming_matches_clip(tmp_path):
    timm = pytest.importorskip("timm")
    labels = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear"]
    reference = nn.ModuleDict(
        {
            "backbone": timm.create_model(
                "mobilenetv4_conv_small", pretrained=False, num_classes=0
            ),
            "gru": nn.GRU(1280, 8, batch_first=True),
            "head": nn.Sequential(nn.Dropout(0.3), nn.Linear(8, len(labels))),
        }
    ).eval()
    path = tmp_path / "emotionv4.pt"
    torch.save(
        {
            "config": {
                "task": "emotion",
                "model": "mobilenetv4_conv_small_gru",
                "class_names": labels,
                "gru_hidden_size": 8,
                "image_size": 64,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "model": reference.state_dict(),
        },
        path,
    )
    predict = EmotionGRUPredictor(path)
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        sequence = reference["backbone"](images).unsqueeze(0)
        output, _ = reference["gru"](sequence)
        expected = reference["head"](output)[0].softmax(dim=-1).numpy()
    for index, frame in enumerate(frames):
        label, scores = predict.predict_track(1, frame)
        assert list(scores) == labels
        np.testing.assert_allclose(list(scores.values()), expected[index], atol=1e-6)
        assert label == labels[expected[index].argmax()]
    assert predict.predict_track(2, frames[0]) == predict.predict(frames[0])
    predict.drop_track(1)
    assert predict.predict_track(1, frames[-1]) == predict.predict(frames[-1])
