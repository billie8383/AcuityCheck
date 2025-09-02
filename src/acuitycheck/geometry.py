from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_distance_mm(
    pixel_ipd: float,
    ipd_mm: float,
    f_px: float,
    offset_mm: float,
) -> Tuple[float, float]:
    """Computes distances between camera, eye, and screen in millimetres.

    The function estimates the camera-to-eye distance and the eye-to-screen distance
    using interpupillary distance (IPD), focal length, and a fixed offset.

    Formulas:
        - Camera-to-eye distance:
          D_cam_eye = (IPD_mm * f_px) / pixel_ipd
        - Eye-to-screen distance:
          D_eye_screen = max(0, D_cam_eye - offset_mm)

    Args:
        pixel_ipd (float): Interpupillary distance measured in pixels.
        ipd_mm (float): Interpupillary distance in millimetres.
        f_px (float): Camera focal length in pixels.
        offset_mm (float): Distance offset in millimetres (e.g., screen thickness).

    Returns:
        Tuple[float, float]:
            - Camera-to-eye distance in millimetres.
            - Eye-to-screen distance in millimetres (clamped at zero).
    """
    cam_to_eye = (float(ipd_mm) * float(f_px)) / max(1e-6, float(pixel_ipd))
    eye_to_screen = max(0.0, cam_to_eye - max(0.0, float(offset_mm)))
    return cam_to_eye, eye_to_screen


def fov_from_fpx(
    f_px: float,
    w_px: int,
    h_px: int,
) -> Tuple[float, float, float]:
    """Computes horizontal, vertical, and diagonal field of view (FOV) in degrees.

    Args:
        f_px (float): Camera focal length in pixels.
        w_px (int): Image width in pixels.
        h_px (int): Image height in pixels.

    Returns:
        Tuple[float, float, float]:
            - Horizontal field of view (HFOV) in degrees.
            - Vertical field of view (VFOV) in degrees.
            - Diagonal field of view (DFOV) in degrees.
    """
    hfov = 2.0 * np.degrees(np.arctan((w_px / 2.0) / float(f_px)))
    vfov = 2.0 * np.degrees(np.arctan((h_px / 2.0) / float(f_px)))
    dfov = 2.0 * np.degrees(np.arctan((np.hypot(w_px, h_px) / 2.0) / float(f_px)))
    return hfov, vfov, dfov
