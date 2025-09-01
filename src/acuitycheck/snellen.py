from __future__ import annotations

from typing import List, Tuple


def snellen_letter_height_mm(distance_mm: float, snellen_den: float) -> float:
    """Compute optotype height in mm for a given Snellen denominator at a given test distance.

    Using the 5-arcmin rule: height_mm ≈ distance_mm * 0.001454 * (den/6).
    """
    return float(distance_mm) * 0.001454 * (float(snellen_den) / 6.0)


def build_snellen_lines(distance_mm: float, pixels_per_mm: float, denominators: List[int]) -> List[Tuple[str, float]]:
    """Return [(label, letter_size_px), ...] for the given denominators."""
    out: List[Tuple[str, float]] = []
    for den in denominators:
        mm = snellen_letter_height_mm(distance_mm, den)
        px = mm * pixels_per_mm
        out.append((f"6/{int(den)}", px))
    return out

