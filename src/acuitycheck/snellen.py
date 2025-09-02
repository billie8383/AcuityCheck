from typing import List, Tuple


def snellen_letter_height_mm(
    distance_mm: float,
    snellen_den: float,
) -> float:
    """Computes optotype (Snellen chart letter) height in millimetres.

    The calculation follows the 5-arcminute rule, which relates visual angle
    to letter height. Each Snellen letter subtends 5 arcminutes at the specified
    distance.

    Formula:
        height_mm ≈ distance_mm * 0.001454 * (den / 6)

    Args:
        distance_mm (float): Viewing distance in millimetres.
        snellen_den (float): Snellen denominator (e.g., 6, 9, 12).

    Returns:
        float: Letter height in millimetres.
    """
    return float(distance_mm) * 0.001454 * (float(snellen_den) / 6.0)


def build_snellen_lines(
    distance_mm: float,
    pixels_per_mm: float,
    denominators: List[int],
) -> List[Tuple[str, float]]:
    """Generates Snellen chart line specifications.

    Each line is represented by a Snellen fraction (e.g., "6/6", "6/12") 
    and the corresponding letter size in pixels.

    Args:
        distance_mm (float): Viewing distance in millimetres.
        pixels_per_mm (float): Display resolution in pixels per millimetre.
        denominators (List[int]): List of Snellen denominators.

    Returns:
        List[Tuple[str, float]]:
            A list of tuples, where each tuple contains:
            - Snellen label as a string (e.g., "6/12").
            - Letter size in pixels as a float.
    """
    snellen_lines: List[Tuple[str, float]] = []
    for den in denominators:
        letter_height_mm = snellen_letter_height_mm(distance_mm, den)
        letter_size_px = letter_height_mm * pixels_per_mm
        snellen_lines.append((f"6/{int(den)}", letter_size_px))
    return snellen_lines
