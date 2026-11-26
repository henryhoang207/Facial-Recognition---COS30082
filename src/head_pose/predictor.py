"""Head pose for the web app from Thành's landmark checkpoint (model/individual/Thanh/model.safetensors).

The checkpoint has no inference code in the repo, so the network is rebuilt from its weights:

    net.b      HRNet-W48 (timm, features_only, incre modules) - matches the weights exactly
    net.neck   1x1 conv 256 -> 32 on the 1/8 feature, then GELU
    net.h      flatten (32 x 32 x 32) -> Linear 1024 -> ReLU -> Linear 1860 (dense head, 620 x (x, y, s))
    side.300w  Linear 1024 -> 68 x (x, y, s) on the same 1024 hidden vector
    side.wflw  Linear 1024 -> 98 x (x, y, s)
    side_rf.*  per-point refinement heads (not used here)

Decoding was chosen by testing on real faces against YuNet's eye, nose and mouth points: a square
256 px crop at 1.4 x the detector box, ImageNet normalisation, 300W points in crop pixels
(about 0.09 of the eye distance from YuNet's points, close to YuNet's own noise). Yaw, pitch and
roll come from solvePnP of six 300W points on a generic 3D face.

    pose = HeadPosePredictor("model/individual/Thanh/model.safetensors")
    pose.predict_frame(frame_bgr, box) -> {"yaw", "pitch", "roll", "landmarks"}
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

INPUT = 256
CROP_SCALE = 1.4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# điểm 300W dùng cho solvePnP: mũi, cằm, khoé mắt ngoài trái/phải, khoé miệng trái/phải
PNP_INDEX = [30, 8, 36, 45, 48, 54]
# mô hình mặt 3D chung (đơn vị tuỳ ý), trục x sang phải ảnh, y lên trên, z hướng về camera
FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    np.float64,
)


class LandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.b = timm.create_model(
            "hrnet_w48", pretrained=False, features_only=True, feature_location="incre"
        )
        self.neck = nn.Sequential(nn.Conv2d(256, 32, 1), nn.GELU())
        self.h = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 32 * 32, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1860)
        )
        self.side_300w = nn.Linear(1024, 68 * 3)
        self.side_wflw = nn.Linear(1024, 98 * 3)

    def feature8(self, x):
        """Only the 1/8 feature the heads use: skips the unused fusions of the last HRNet module
        and three of the four incre modules (same output as self.b(x)[2])."""
        b = self.b
        x = b.act1(b.bn1(b.conv1(x)))
        x = b.act2(b.bn2(b.conv2(x)))
        x = b.layer1(x)
        yl = b.stage2([t(x) for t in b.transition1])
        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(b.transition2)]
        yl = b.stage3(xl)
        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(b.transition3)]
        for module in list(b.stage4)[:-1]:
            xl = module(xl)
        last = b.stage4[-1]
        xl = [branch(v) for branch, v in zip(last.branches, xl, strict=True)]
        fused = None
        for j, layer in enumerate(last.fuse_layers[1]):
            fused = layer(xl[j]) if fused is None else fused + layer(xl[j])
        return b.incre_modules[1](last.fuse_act(fused))

    def landmarks_300w(self, x):
        return self.side_300w(self.h[:3](self.neck(self.feature8(x)))).view(-1, 68, 3)

    def forward(self, x):
        features = self.b(x)
        hidden = self.h[:3](self.neck(features[2]))
        return {
            "300w": self.side_300w(hidden).view(-1, 68, 3),
            "wflw": self.side_wflw(hidden).view(-1, 98, 3),
            "dense": self.h[3](hidden).view(-1, 620, 3),
        }


SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
}


def read_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Minimal .safetensors reader (8-byte header size, JSON header, raw little-endian data).

    Kept here so the app does not need the `safetensors` package.
    """
    with open(path, "rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_size))
        data = bytearray(stream.read())
    header.pop("__metadata__", None)
    tensors = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        dtype = SAFETENSORS_DTYPES[entry["dtype"]]
        flat = torch.frombuffer(data, dtype=dtype, count=(end - start) // dtype.itemsize, offset=start)
        tensors[name] = flat.reshape(entry["shape"]).clone()
    return tensors


def load_landmark_net(path: str | Path) -> LandmarkNet:
    weights = read_safetensors(path)
    renamed = {}
    for key, value in weights.items():
        if key.startswith("side_rf."):
            continue  # khối tinh chỉnh không dùng
        key = key.removeprefix("net.").replace("side.300w.", "side_300w.")
        renamed[key.replace("side.wflw.", "side_wflw.")] = value
    net = LandmarkNet()
    net.load_state_dict(renamed)  # strict: mọi trọng số còn lại phải khớp
    return net.eval()


def crop_transform(box, scale: float = CROP_SCALE) -> np.ndarray:
    """2x3 affine from frame pixels to the square network input."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * scale
    k = INPUT / side
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return np.array([[k, 0, INPUT / 2 - cx * k], [0, k, INPUT / 2 - cy * k]], np.float32)


def pose_from_landmarks(points: np.ndarray, frame_size: tuple[int, int]) -> dict[str, float]:
    """68 points in frame pixels -> yaw, pitch, roll in degrees (app convention, see head_pose.py)."""
    width, height = frame_size
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], np.float64)
    image = points[PNP_INDEX].astype(np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, image, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    # ảnh có trục y hướng xuống: đổi sang hệ y lên để đọc góc như mô hình 3D
    flip = np.diag([1.0, -1.0, -1.0])
    r = flip @ rotation
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))  # quay về bên trái của người > 0
    pitch = -math.degrees(math.atan2(r[2, 1], r[2, 2]))  # ngẩng cằm lên > 0
    roll = -math.degrees(math.atan2(r[1, 0], r[0, 0]))  # nghiêng về vai trái của người > 0
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


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


class HeadPosePredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device, self.precision, self.gpu_error = setup_device(
            self, load_landmark_net(checkpoint), (1, 3, INPUT, INPUT), device
        )

    @torch.inference_mode()
    def _run(self, tensor: torch.Tensor, precision: str) -> torch.Tensor:
        with autocast(self.device, precision):
            return self.net.landmarks_300w(tensor.contiguous(memory_format=torch.channels_last)).float()

    def landmarks(self, frame: np.ndarray, box) -> np.ndarray:
        """68 300W points (x, y) in frame pixels."""
        matrix = crop_transform(box)
        crop = cv2.warpAffine(frame, matrix, (INPUT, INPUT), flags=cv2.INTER_LINEAR)
        image = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        points = self._run(tensor, self.precision)[0, :, :2].cpu().numpy()
        inverse = cv2.invertAffineTransform(matrix)
        return points @ inverse[:, :2].T + inverse[:, 2]

    def predict_frame(self, frame: np.ndarray, box) -> dict:
        points = self.landmarks(frame, box)
        angles = pose_from_landmarks(points, (frame.shape[1], frame.shape[0]))
        return {**angles, "landmarks": points.tolist()}
