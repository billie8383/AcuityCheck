from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from pathlib import Path


def run_yunet_cv2(
    frame_bgr: np.ndarray,
    model_path: Path,
    *,
    score_thresh: float = 0.3,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[List[Tuple[float, float]]]]:
    """Detect face using OpenCV's YuNet wrapper.

    Returns (box, keypoints) in pixel coordinates, or (None, None) if not found.
    """
    try:
        if not model_path.exists():
            return None, None
        h, w = frame_bgr.shape[:2]
        detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (w, h),
            score_threshold=float(score_thresh),
            nms_threshold=0.3,
            top_k=5000,
        )
        detector.setInputSize((w, h))
        retval, faces = detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None, None
        faces = faces.reshape(-1, 15)
        idx = int(np.argmax(faces[:, 14]))
        row = faces[idx]
        x, y, bw, bh = [float(v) for v in row[:4]]
        kps = [
            (float(row[4]), float(row[5])),
            (float(row[6]), float(row[7])),
            (float(row[8]), float(row[9])),
            (float(row[10]), float(row[11])),
            (float(row[12]), float(row[13])),
        ]
        return (x, y, bw, bh), kps
    except Exception:
        return None, None


def detect_eyes_in_roi(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
) -> Optional[List[Tuple[float, float]]]:
    """Fallback: detect two eyes inside the provided face box using Haar cascade."""
    x, y, w, h = [int(v) for v in box]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    try:
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
    except Exception:
        return None
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
    pts: List[Tuple[float, float]] = []
    for (ex, ey, ew, eh) in eyes:
        cx = x + ex + ew / 2.0
        cy = y + ey + eh / 2.0
        pts.append((float(cx), float(cy)))
    pts = sorted(pts, key=lambda p: p[0])
    return pts


