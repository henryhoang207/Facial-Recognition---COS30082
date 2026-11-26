"""Face parsing for the web app (Mạnh Đức's checkpoint, model/individual/Duc/best.pt).

The network is rebuilt from the checkpoint's weights: a BiSeNet-style context path on ResNet-18
(attention refinement at 1/16 and 1/32, global context, refinement convs), a gated fusion with the
1/8 ResNet feature, and an 11-class head (aux heads are kept only so the weights load strictly).

    parser = FaceParsingPredictor("model/individual/Duc/best.pt")
    mask, region = parser.predict_frame(frame_bgr, box)   # mask: class ids over `region`
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_SCALE = 1.9  # khung mặt YuNet chỉ ôm mặt; nới ra để có cả tóc như ảnh train


def conv_bn_relu(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class AttentionRefinement(nn.Module):
    def __init__(self, c_in: int, c_out: int = 128):
        super().__init__()
        self.features = conv_bn_relu(c_in, c_out, 3)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return x * self.attention(x)


def _stage(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    down = None
    if stride != 1 or c_in != c_out:
        down = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
    return nn.Sequential(BasicBlock(c_in, c_out, stride, down), BasicBlock(c_out, c_out))


def _head(c_in: int, mid: int, classes: int) -> nn.Sequential:
    return nn.Sequential(conv_bn_relu(c_in, mid, 3), nn.Conv2d(mid, classes, 1))


class FaceParsingNet(nn.Module):
    def __init__(self, classes: int = 11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stages = nn.Sequential(
            _stage(64, 64, 1), _stage(64, 128, 2), _stage(128, 256, 2), _stage(256, 512, 2)
        )
        self.arm16 = AttentionRefinement(256)
        self.arm32 = AttentionRefinement(512)
        self.global_context = conv_bn_relu(512, 128, 1)
        self.refine16 = conv_bn_relu(128, 128, 3)
        self.refine8 = conv_bn_relu(128, 128, 3)
        self.fusion = conv_bn_relu(256, 256, 1)
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, bias=False),
            nn.Sigmoid(),
        )
        self.main = _head(256, 256, classes)
        self.aux8 = _head(128, 64, classes)
        self.aux16 = _head(128, 64, classes)

    def forward(self, x):
        size = x.shape[2:]
        x = self.stem(x)
        feat4 = self.stages[0](x)
        feat8 = self.stages[1](feat4)
        feat16 = self.stages[2](feat8)
        feat32 = self.stages[3](feat16)

        context = self.global_context(F.adaptive_avg_pool2d(feat32, 1))
        up32 = self.arm32(feat32) + context
        up32 = self.refine16(F.interpolate(up32, size=feat16.shape[2:], mode="nearest"))
        up16 = self.arm16(feat16) + up32
        up16 = self.refine8(F.interpolate(up16, size=feat8.shape[2:], mode="nearest"))

        fused = self.fusion(torch.cat([feat8, up16], dim=1))
        fused = fused + fused * self.fusion_gate(fused)
        logits = self.main(fused)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def face_region(box, width: int, height: int, scale: float = CROP_SCALE) -> tuple[int, int, int, int]:
    """Square around the detector box, scaled up and shifted up a little for the hair."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - 0.08 * (y2 - y1)
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = int(round(cx + side / 2)), int(round(cy + side / 2))
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def resolve_device(device: str | None) -> torch.device:
    """'auto' -> cuda when available, else cpu."""
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pick_precision(run, device: torch.device) -> str:
    """fp16 on a GPU (fp32 if fp16 gives inf / nan); on a CPU, bf16 only when it is clearly faster
    (CPUs without bf16 units are slower). A GPU that cannot run the model raises here."""
    if device.type == "cuda":
        run("fp32")
        return "fp16" if bool(torch.isfinite(run("fp16")).all()) else "fp32"
    timings = {}
    for mode in ("fp32", "bf16"):
        try:
            run(mode)
            started = time.perf_counter()
            for _ in range(2):
                run(mode)
            timings[mode] = time.perf_counter() - started
        except RuntimeError:
            continue
    return "bf16" if timings.get("bf16", float("inf")) < 0.85 * timings["fp32"] else "fp32"


def setup_device(predictor, net: nn.Module, shape: tuple, device: str | None):
    """Move `net` to the requested device and test it there. If the GPU cannot run it (for
    example a card too old for this PyTorch build), fall back to the CPU instead of failing
    later on every frame. Returns (device, precision, gpu_error)."""
    target = resolve_device(device)
    gpu_error = None
    for candidate in ([target, torch.device("cpu")] if target.type == "cuda" else [target]):
        try:
            predictor.device = candidate
            predictor.net = net.to(candidate).to(memory_format=torch.channels_last)
            example = torch.zeros(*shape, device=candidate)
            precision = pick_precision(lambda mode: predictor._run(example, mode), candidate)
            return candidate, precision, gpu_error
        except (RuntimeError, AssertionError) as exc:  # CUDA lỗi kernel / bản torch không có CUDA
            if candidate.type != "cuda":
                raise
            gpu_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    raise RuntimeError("no usable device")


def autocast(device: torch.device, precision: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


class FaceParsingPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.classes: list[str] = list(ckpt["classes"])
        self.side: int = int(ckpt.get("side", 512))
        net = FaceParsingNet(len(self.classes))
        net.load_state_dict(ckpt["weights"])
        # channels_last nhanh hơn khoảng 20% trên CPU; độ phân giải giữ 512 vì giảm xuống làm mất mắt / môi
        self.device, self.precision, self.gpu_error = setup_device(
            self, net.eval(), (1, 3, self.side, self.side), device
        )

    @torch.inference_mode()
    def _run(self, tensor: torch.Tensor, precision: str) -> torch.Tensor:
        with autocast(self.device, precision):
            return self.net(tensor.contiguous(memory_format=torch.channels_last)).float()

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        """`face_bgr`: crop of any size. Returns class ids at the crop's size."""
        height, width = face_bgr.shape[:2]
        image = cv2.resize(face_bgr, (self.side, self.side), interpolation=cv2.INTER_LINEAR)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        labels = self._run(tensor, self.precision).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

    def predict_frame(self, frame: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        region = face_region(box, frame.shape[1], frame.shape[0])
        x1, y1, x2, y2 = region
        return self.predict(frame[y1:y2, x1:x2]), region
