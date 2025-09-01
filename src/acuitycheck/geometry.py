from __future__ import annotations

from typing import Tuple
import numpy as np

def compute_distance_mm(pixel_ipd: float, ipd_mm: float, f_px: float, offset_mm: float) -> Tuple[float, float]:
    """Compute camera->eye and eye->screen distance in millimetres.

    D_cam_eye = (IPD_mm * f_px) / pixel_ipd
    D_eye_screen = max(0, D_cam_eye - offset_mm)
    """
    cam_to_eye = (float(ipd_mm) * float(f_px)) / max(1e-6, float(pixel_ipd))
    eye_to_screen = max(0.0, cam_to_eye - max(0.0, float(offset_mm)))
    return cam_to_eye, eye_to_screen


def fov_from_fpx(f_px: float, w_px: int, h_px: int):
    """Return (HFOV, VFOV, DFOV) in degrees from focal length in pixels."""
    hfov = 2.0 * np.degrees(np.arctan((w_px / 2.0) / float(f_px)))
    vfov = 2.0 * np.degrees(np.arctan((h_px / 2.0) / float(f_px)))
    dfov = 2.0 * np.degrees(np.arctan((np.hypot(w_px, h_px) / 2.0) / float(f_px)))
    return hfov, vfov, dfov
