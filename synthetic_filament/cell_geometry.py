from __future__ import annotations

import math

import numpy as np
from skimage.draw import polygon


def generate_cell_mask(
    image_shape: tuple[int, int],
    center_yx: tuple[float, float],
    equivalent_radius: float,
    eccentricity: float,
    boundary_smoothness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image_shape
    cy, cx = center_yx
    eccentricity = float(np.clip(eccentricity, 0.0, 0.95))
    a = equivalent_radius / max((1.0 - eccentricity**2) ** 0.25, 1e-6)
    b = max(equivalent_radius**2 / max(a, 1e-6), 2.0)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))

    angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    base_x = a * np.cos(angles)
    base_y = b * np.sin(angles)

    irregularity_scale = float(np.clip(1.0 - boundary_smoothness, 0.0, 0.2))
    phase = rng.uniform(0, 2 * math.pi, size=3)
    radial_mod = 1.0 + irregularity_scale * (
        0.7 * np.sin(angles * 2 + phase[0]) +
        0.2 * np.sin(angles * 3 + phase[1]) +
        0.1 * np.sin(angles * 5 + phase[2])
    )

    x = radial_mod * base_x
    y = radial_mod * base_y
    rot_x = x * math.cos(theta) - y * math.sin(theta)
    rot_y = x * math.sin(theta) + y * math.cos(theta)

    rr, cc = polygon(cy + rot_y, cx + rot_x, shape=image_shape)
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    return mask
