"""Rough head angles from YuNet's 5 points (eyes, nose tip, mouth corners).

No model: the nose tip sits in front of the eye plane, so turning the head shifts it along the
eye line. Good enough to check "did the person turn left / right", not a head pose estimator.
A flat photo turned in front of the camera does not move the nose relative to the eyes.
"""

from __future__ import annotations

import math

import numpy as np

# độ nhô của đầu mũi so với mặt phẳng hai mắt, tính theo khoảng cách hai mắt (mặt người trung bình)
NOSE_DEPTH = 0.55
# vị trí đầu mũi giữa đường mắt và đường miệng khi nhìn thẳng
NOSE_HEIGHT = 0.55


def head_angles(points: np.ndarray) -> dict[str, float]:
    """`points`: (5, 2) in frame pixels, YuNet order. Angles in degrees.

    yaw > 0: the person turned to their own left. pitch > 0: chin up. roll > 0: head tilted
    towards the person's left shoulder (in the unmirrored frame).
    """
    right_eye, left_eye, nose, right_mouth, left_mouth = points.astype(np.float64)
    eye_axis = left_eye - right_eye
    eye_dist = float(np.linalg.norm(eye_axis))
    if eye_dist < 1e-6:
        raise ValueError("degenerate landmarks")
    across = eye_axis / eye_dist
    down = np.array([-across[1], across[0]])  # vuông góc với đường mắt, hướng xuống cằm
    eye_mid = (right_eye + left_eye) / 2
    mouth_mid = (right_mouth + left_mouth) / 2
    offset = float(np.dot(nose - eye_mid, across)) / eye_dist
    face_height = float(np.dot(mouth_mid - eye_mid, down))
    height = float(np.dot(nose - eye_mid, down)) / face_height if face_height > 1e-6 else NOSE_HEIGHT
    return {
        "yaw": math.degrees(math.atan2(offset, NOSE_DEPTH)),
        "pitch": math.degrees(math.atan2(NOSE_HEIGHT - height, NOSE_DEPTH)),
        "roll": math.degrees(math.atan2(eye_axis[1], eye_axis[0])),
    }
