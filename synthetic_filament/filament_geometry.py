from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def _sample_centerline(
    center: tuple[float, float],
    length: float,
    orientation: float,
    curvature_strength: float,
    rng: np.random.Generator,
    num_points: int = 64,
) -> np.ndarray:
    x = np.linspace(-length / 2.0, length / 2.0, num_points)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    y = curvature_strength * (
        np.sin(np.linspace(0, math.pi, num_points) + phase) +
        0.35 * np.sin(np.linspace(0, 2 * math.pi, num_points) - phase)
    )
    y += rng.normal(scale=0.1 * max(curvature_strength, 1.0), size=num_points)
    local = np.stack([y, x], axis=1)
    rot = np.asarray(
        [
            [math.cos(orientation), -math.sin(orientation)],
            [math.sin(orientation), math.cos(orientation)],
        ]
    )
    rotated = local @ rot.T
    rotated[:, 0] += center[0]
    rotated[:, 1] += center[1]
    return rotated


def _distance_to_polyline(image_shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    rr, cc = np.indices(image_shape)
    grid = np.stack([rr, cc], axis=-1).astype(float)
    min_dist2 = np.full(image_shape, np.inf, dtype=float)
    for p0, p1 in zip(points[:-1], points[1:]):
        segment = p1 - p0
        seg_len2 = float(np.dot(segment, segment))
        if seg_len2 <= 1e-9:
            continue
        rel = grid - p0
        t = np.clip((rel[..., 0] * segment[0] + rel[..., 1] * segment[1]) / seg_len2, 0.0, 1.0)
        proj = p0 + t[..., None] * segment
        dist2 = np.sum((grid - proj) ** 2, axis=-1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.sqrt(min_dist2)


def generate_filament_mask(
    image_shape: tuple[int, int],
    cell_mask: np.ndarray,
    cell_center: tuple[float, float],
    length: float,
    width: float,
    orientation: float,
    radial_position_norm: float,
    curvature_strength: float,
    rng: np.random.Generator,
    max_tries: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    cell_dist = ndi.distance_transform_edt(cell_mask)
    radius = max(float(np.sqrt(np.sum(cell_mask) / math.pi)), 1.0)
    direction = np.asarray([math.sin(orientation), math.cos(orientation)])

    for _ in range(max_tries):
        angle = rng.uniform(0, 2 * math.pi)
        offset = radial_position_norm * radius * np.asarray([math.sin(angle), math.cos(angle)])
        center = np.asarray(cell_center) + offset
        points = _sample_centerline(tuple(center), length=length, orientation=orientation, curvature_strength=curvature_strength, rng=rng)
        dist = _distance_to_polyline(image_shape, points)
        width_profile = width * (0.85 + 0.15 * np.sin(np.linspace(0, math.pi, len(points[:-1] if len(points) > 1 else points))))
        # v1 uses mean width; tapering remains mild and implicit in blur.
        mask = dist <= max(float(np.mean(width_profile)) / 2.0, 1.0)
        inside_fraction = float(np.mean(cell_mask[mask])) if np.any(mask) else 0.0
        if inside_fraction >= 0.9 and np.all(cell_dist[tuple(np.clip(points.astype(int), [0, 0], np.array(image_shape) - 1).T)] > 0):
            return mask, points

    fallback = dist <= max(width / 2.0, 1.0)
    return fallback & cell_mask, points
