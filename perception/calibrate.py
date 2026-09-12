"""Court calibration: 4 user-clicked landmarks -> pixel-to-court homography.

Deliberately NOT automatic. The user clicks 4 known court landmarks once
(e.g. the two near baseline corners and the two free-throw line corners of a
half court) in calibration_ui, and we solve for the homography that maps
pixel coordinates to real-world court coordinates in feet.
"""
from __future__ import annotations

import numpy as np
import cv2

from perception.schemas import Calibration


def compute_homography(
    image_points: list[tuple[float, float]],
    court_points_ft: list[tuple[float, float]],
) -> np.ndarray:
    """Solve for the 3x3 homography mapping image pixels -> court feet.

    Requires exactly 4 point correspondences (a minimal, well-conditioned
    solve for a planar court surface — no RANSAC needed since these are
    hand-clicked, not detected).
    """
    if len(image_points) != 4 or len(court_points_ft) != 4:
        raise ValueError("Calibration requires exactly 4 point correspondences")

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(court_points_ft, dtype=np.float64)
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise ValueError("Could not solve homography from the given points")
    return homography


def pixel_to_court(
    homography: np.ndarray, pixel_xy: tuple[float, float]
) -> tuple[float, float]:
    """Map a single pixel coordinate to court feet using a precomputed homography."""
    point = np.array([[pixel_xy]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(point, homography)
    x, y = mapped[0][0]
    return float(x), float(y)


def pixels_to_court_batch(
    homography: np.ndarray, pixel_points: np.ndarray
) -> np.ndarray:
    """Vectorized version of pixel_to_court for an (N, 2) array of points."""
    pts = pixel_points.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(pts, homography)
    return mapped.reshape(-1, 2)


def build_calibration(
    image_points: list[tuple[float, float]],
    court_points_ft: list[tuple[float, float]],
    landmarks_used: str,
) -> Calibration:
    homography = compute_homography(image_points, court_points_ft)
    return Calibration(
        image_points=image_points,
        court_points_ft=court_points_ft,
        homography=homography.tolist(),
        landmarks_used=landmarks_used,
    )
