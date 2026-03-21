#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile


DEFAULT_EPS_VALUES = np.linspace(1.0, 12.0, 30)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def round_value(value: float | int | bool | None, digits: int = 4) -> float | int | bool | None:
    if isinstance(value, (bool, int)) or value is None:
        return value
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def round_points(points: np.ndarray, digits: int = 2) -> list[list[float]]:
    if len(points) == 0:
        return []
    return np.round(np.asarray(points, dtype=float), digits).tolist()


def is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points
    keep = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[keep]


def save_grayscale_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image, cmap="gray", vmin=0.0, vmax=1.0)


def normalization_bounds_from_volume_stack(
    volume_stack: np.ndarray,
    mode: str = "timeseries_percentile",
    percentiles: tuple[float, float] = (1.0, 99.5),
) -> tuple[float, float]:
    if mode == "timeseries_minmax":
        lower_bound = float(volume_stack.min())
        upper_bound = float(volume_stack.max())
    elif mode == "timeseries_percentile":
        lower_bound, upper_bound = np.percentile(volume_stack, percentiles)
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
    else:
        raise ValueError("mode must be 'timeseries_minmax' or 'timeseries_percentile'.")

    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1e-12
    return lower_bound, upper_bound


def normalize_volume_to_unit_interval(volume: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    normalized_volume = (volume - lower_bound) / (upper_bound - lower_bound + 1e-12)
    return np.clip(normalized_volume, 0.0, 1.0)


def epsilon_graph_from_sq_dists(sq_dists: np.ndarray, epsilon: float) -> np.ndarray:
    return sq_dists <= epsilon**2


def connected_components_from_adjacency(adjacency: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(len(adjacency), dtype=bool)
    components: list[np.ndarray] = []

    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = np.where(adjacency[node])[0]
            for neighbour in neighbours:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(np.array(component, dtype=int))
    return components


def component_dense_line_stats_nd(
    point_cloud: np.ndarray,
    adjacency: np.ndarray,
    component: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    component_points = point_cloud[component]
    size = len(component_points)
    if size <= 1:
        return {
            "size": size,
            "major_span": 0.0,
            "minor_span": 0.0,
            "linearity": 0.0,
            "linear_density": 0.0,
            "mean_degree": 0.0,
            "score": float("-inf"),
        }

    centred = component_points - component_points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ vh.T
    spans = rotated.max(axis=0) - rotated.min(axis=0)
    major_span = float(spans[0]) if len(spans) else 0.0
    transverse_span = float(np.sqrt(np.mean(np.square(spans[1:])))) if len(spans) > 1 else 0.0
    linearity = major_span / (transverse_span + 1e-6)

    component_adjacency = adjacency[np.ix_(component, component)]
    degrees = component_adjacency.sum(axis=1) - 1
    mean_degree = float(np.mean(degrees)) if len(degrees) else 0.0
    linear_density = size / (major_span + epsilon + 1e-6)
    span_bonus = np.tanh(major_span / 6.0)

    score = (
        linear_density**2
        * np.sqrt(np.log1p(size))
        * np.log1p(linearity)
        * np.log1p(mean_degree + 1.0)
        * span_bonus
        / max(epsilon, 1e-6)
    )
    return {
        "size": size,
        "major_span": major_span,
        "minor_span": transverse_span,
        "linearity": float(linearity),
        "linear_density": float(linear_density),
        "mean_degree": mean_degree,
        "score": float(score),
    }


def strongest_dense_line_component_nd_from_sq_dists(
    point_cloud: np.ndarray,
    sq_dists: np.ndarray,
    epsilon: float,
    min_component_size: int = 25,
    min_linearity: float = 2.0,
    min_linear_density: float = 1.0,
    min_mean_degree: float = 2.0,
) -> tuple[np.ndarray | None, dict[str, float]]:
    adjacency = epsilon_graph_from_sq_dists(sq_dists, epsilon)
    components = connected_components_from_adjacency(adjacency)
    stats = [component_dense_line_stats_nd(point_cloud, adjacency, component, epsilon) for component in components]

    valid_indices = [
        index
        for index, stat in enumerate(stats)
        if stat["size"] >= min_component_size
        and stat["linearity"] >= min_linearity
        and stat["linear_density"] >= min_linear_density
        and stat["mean_degree"] >= min_mean_degree
    ]
    if not valid_indices:
        return None, {
            "size": 0,
            "major_span": 0.0,
            "minor_span": 0.0,
            "linearity": 0.0,
            "linear_density": 0.0,
            "mean_degree": 0.0,
            "score": float("-inf"),
        }

    best_index = max(valid_indices, key=lambda index: stats[index]["score"])
    mask = np.zeros(len(point_cloud), dtype=bool)
    mask[components[best_index]] = True
    return mask, stats[best_index]


def principal_basis_from_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n_points, n_dims = points.shape
    if n_points == 0:
        return np.eye(n_dims)

    centred = points - points.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order].T

    if n_dims == 3 and np.linalg.det(basis) < 0:
        basis[-1] *= -1
    return basis


def subsample_points_along_axis(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points

    basis = principal_basis_from_points(points)
    centred = points - points.mean(axis=0, keepdims=True)
    order = np.argsort(centred @ basis[0])
    keep = np.linspace(0, len(order) - 1, max_points).astype(int)
    return points[order[keep]]


def minimum_spanning_tree_adjacency(points: np.ndarray) -> list[list[tuple[int, float]]]:
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_points)]
    if n_points <= 1:
        return adjacency

    sq_dists = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    dists = np.sqrt(np.maximum(sq_dists, 0.0))

    in_tree = np.zeros(n_points, dtype=bool)
    in_tree[0] = True
    min_dist = dists[0].copy()
    parent = np.zeros(n_points, dtype=int)
    min_dist[0] = np.inf

    for _ in range(n_points - 1):
        next_node = int(np.argmin(min_dist))
        edge_parent = int(parent[next_node])
        edge_weight = float(dists[next_node, edge_parent])
        adjacency[next_node].append((edge_parent, edge_weight))
        adjacency[edge_parent].append((next_node, edge_weight))

        in_tree[next_node] = True
        min_dist[next_node] = np.inf

        update_mask = (~in_tree) & (dists[next_node] < min_dist)
        parent[update_mask] = next_node
        min_dist[update_mask] = dists[next_node, update_mask]

    return adjacency


def farthest_node_in_tree(
    adjacency: list[list[tuple[int, float]]],
    start_node: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    n_nodes = len(adjacency)
    parent = np.full(n_nodes, -1, dtype=int)
    distances = np.full(n_nodes, -np.inf, dtype=float)
    distances[start_node] = 0.0
    stack = [start_node]

    while stack:
        node = stack.pop()
        for neighbour, weight in adjacency[node]:
            if neighbour == parent[node]:
                continue
            parent[neighbour] = node
            distances[neighbour] = distances[node] + weight
            stack.append(neighbour)

    farthest_node = int(np.argmax(distances))
    return farthest_node, parent, distances


def diameter_path_from_points(points: np.ndarray, max_points: int) -> tuple[np.ndarray, float]:
    sampled_points = subsample_points_along_axis(points, max_points=max_points)
    if len(sampled_points) <= 1:
        return sampled_points, 0.0

    adjacency = minimum_spanning_tree_adjacency(sampled_points)
    end_a, _, _ = farthest_node_in_tree(adjacency, start_node=0)
    end_b, parent, distances = farthest_node_in_tree(adjacency, start_node=end_a)

    path_indices = [end_b]
    while path_indices[-1] != end_a:
        path_indices.append(parent[path_indices[-1]])
    path_indices.reverse()

    return sampled_points[np.array(path_indices, dtype=int)], float(distances[end_b])


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return float("nan")
    complex_mean = np.mean(np.exp(1j * np.deg2rad(angles_deg)))
    if np.abs(complex_mean) < 1e-12:
        return float("nan")
    return float(np.rad2deg(np.angle(complex_mean)))


def circular_variance_deg(angles_deg: np.ndarray) -> float:
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return float("nan")
    complex_mean = np.mean(np.exp(1j * np.deg2rad(angles_deg)))
    return float(1.0 - np.abs(complex_mean))


def signed_angle_deg(reference: np.ndarray, vector: np.ndarray, axis: np.ndarray) -> float:
    cross_term = np.cross(reference, vector)
    sin_term = np.dot(cross_term, axis)
    cos_term = np.clip(np.dot(reference, vector), -1.0, 1.0)
    return float(np.rad2deg(np.arctan2(sin_term, cos_term)))


def orientation_angles_from_basis(basis: np.ndarray) -> dict[str, float]:
    axis = basis[0] / (np.linalg.norm(basis[0]) + 1e-12)
    yaw_deg = float(np.rad2deg(np.arctan2(axis[1], axis[0])))
    pitch_deg = float(np.rad2deg(np.arctan2(axis[2], np.linalg.norm(axis[:2])))) if len(axis) >= 3 else float("nan")

    roll_deg = float("nan")
    if basis.shape[0] >= 3 and basis.shape[1] >= 3:
        secondary = basis[1] / (np.linalg.norm(basis[1]) + 1e-12)
        reference = None
        for candidate in (
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ):
            projected = candidate - np.dot(candidate, axis) * axis
            projected_norm = np.linalg.norm(projected)
            if projected_norm > 1e-8:
                reference = projected / projected_norm
                break
        if reference is not None:
            roll_deg = signed_angle_deg(reference, secondary, axis)

    return {
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
    }


def align_basis_to_previous(current_basis: np.ndarray | None, previous_basis: np.ndarray | None) -> np.ndarray | None:
    if current_basis is None or previous_basis is None:
        return current_basis

    aligned_basis = current_basis.copy()
    if np.dot(previous_basis[0], aligned_basis[0]) < 0:
        aligned_basis[0] *= -1
        if aligned_basis.shape[0] >= 2:
            aligned_basis[1] *= -1

    if aligned_basis.shape[0] >= 3 and np.dot(previous_basis[1], aligned_basis[1]) < 0:
        aligned_basis[1] *= -1
        aligned_basis[2] *= -1

    return aligned_basis


def angle_moments_from_backbone(backbone_points: np.ndarray) -> dict[str, float]:
    if len(backbone_points) < 2:
        return {
            "tangent_yaw_mean_deg": float("nan"),
            "tangent_yaw_circular_variance": float("nan"),
            "tangent_pitch_mean_deg": float("nan"),
            "tangent_pitch_variance_deg2": float("nan"),
            "turning_angle_mean_deg": float("nan"),
            "turning_angle_variance_deg2": float("nan"),
        }

    segment_vectors = np.diff(backbone_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_mask = segment_lengths > 1e-8
    if not np.any(valid_mask):
        return {
            "tangent_yaw_mean_deg": float("nan"),
            "tangent_yaw_circular_variance": float("nan"),
            "tangent_pitch_mean_deg": float("nan"),
            "tangent_pitch_variance_deg2": float("nan"),
            "turning_angle_mean_deg": float("nan"),
            "turning_angle_variance_deg2": float("nan"),
        }

    unit_tangents = segment_vectors[valid_mask] / segment_lengths[valid_mask, None]
    tangent_yaw_deg = np.rad2deg(np.arctan2(unit_tangents[:, 1], unit_tangents[:, 0]))
    tangent_pitch_deg = np.rad2deg(
        np.arctan2(unit_tangents[:, 2], np.linalg.norm(unit_tangents[:, :2], axis=1) + 1e-12)
    )

    turning_angles_deg = np.array([], dtype=float)
    if len(unit_tangents) >= 2:
        tangent_dots = np.sum(unit_tangents[:-1] * unit_tangents[1:], axis=1)
        turning_angles_deg = np.rad2deg(np.arccos(np.clip(tangent_dots, -1.0, 1.0)))

    return {
        "tangent_yaw_mean_deg": circular_mean_deg(tangent_yaw_deg),
        "tangent_yaw_circular_variance": circular_variance_deg(tangent_yaw_deg),
        "tangent_pitch_mean_deg": float(np.mean(tangent_pitch_deg)) if len(tangent_pitch_deg) else float("nan"),
        "tangent_pitch_variance_deg2": float(np.var(tangent_pitch_deg)) if len(tangent_pitch_deg) else float("nan"),
        "turning_angle_mean_deg": float(np.mean(turning_angles_deg)) if len(turning_angles_deg) else float("nan"),
        "turning_angle_variance_deg2": float(np.var(turning_angles_deg)) if len(turning_angles_deg) else float("nan"),
    }


def empty_frame_metrics() -> dict[str, float | None]:
    return {
        "centroid_x": float("nan"),
        "centroid_y": float("nan"),
        "centroid_z": float("nan"),
        "filament_length": float("nan"),
        "end_to_end_distance": float("nan"),
        "tortuosity": float("nan"),
        "yaw_deg": float("nan"),
        "pitch_deg": float("nan"),
        "roll_deg": float("nan"),
        "tangent_yaw_mean_deg": float("nan"),
        "tangent_yaw_circular_variance": float("nan"),
        "tangent_pitch_mean_deg": float("nan"),
        "tangent_pitch_variance_deg2": float("nan"),
        "turning_angle_mean_deg": float("nan"),
        "turning_angle_variance_deg2": float("nan"),
        "_basis": None,
        "_backbone_points": np.empty((0, 3), dtype=float),
    }


def filament_frame_metrics(recovered_points: np.ndarray, max_points_for_length: int = 250) -> dict[str, object]:
    recovered_points = np.asarray(recovered_points, dtype=float)
    if len(recovered_points) < 2:
        return empty_frame_metrics()

    centroid = recovered_points.mean(axis=0)
    basis = principal_basis_from_points(recovered_points)
    backbone_points, filament_length = diameter_path_from_points(recovered_points, max_points=max_points_for_length)
    end_to_end_distance = (
        float(np.linalg.norm(backbone_points[-1] - backbone_points[0])) if len(backbone_points) >= 2 else float("nan")
    )

    metrics: dict[str, object] = {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "filament_length": float(filament_length),
        "end_to_end_distance": end_to_end_distance,
        "tortuosity": float(filament_length / (end_to_end_distance + 1e-12)) if np.isfinite(end_to_end_distance) else float("nan"),
        "_basis": basis,
        "_backbone_points": backbone_points,
    }
    metrics.update(angle_moments_from_backbone(backbone_points))
    metrics.update(orientation_angles_from_basis(basis))
    return metrics


def compute_centroid_msd(
    time_indices: np.ndarray,
    positions: np.ndarray,
    frame_interval: float = 1.0,
) -> list[dict[str, float]]:
    time_indices = np.asarray(time_indices, dtype=int)
    positions = np.asarray(positions, dtype=float)
    if len(time_indices) < 2:
        return []

    displacements_by_lag: dict[int, list[float]] = defaultdict(list)
    for start in range(len(time_indices) - 1):
        for stop in range(start + 1, len(time_indices)):
            lag_frames = int(time_indices[stop] - time_indices[start])
            if lag_frames <= 0:
                continue
            sq_displacement = float(np.sum((positions[stop] - positions[start]) ** 2))
            displacements_by_lag[lag_frames].append(sq_displacement)

    msd_rows = []
    for lag_frames in sorted(displacements_by_lag):
        sq_displacements = np.asarray(displacements_by_lag[lag_frames], dtype=float)
        msd_rows.append(
            {
                "lag_frames": lag_frames,
                "lag_time": float(lag_frames * frame_interval),
                "msd": float(np.mean(sq_displacements)),
                "n_pairs": int(len(sq_displacements)),
            }
        )
    return msd_rows


def summarize_track(
    records: list[dict[str, object]],
    msd_rows: list[dict[str, float]],
    file_name: str,
    channel_index: int,
) -> dict[str, object]:
    valid_records = [record for record in records if is_finite_number(record["filament_length"])]
    summary: dict[str, object] = {
        "file_name": file_name,
        "channel_index": channel_index,
        "n_frames_total": int(len(records)),
        "n_frames_with_filament": int(sum(record["has_filament"] for record in records)),
        "mean_length": None,
        "std_length": None,
        "mean_tortuosity": None,
        "yaw_mean_deg": None,
        "yaw_circular_variance": None,
        "pitch_mean_deg": None,
        "pitch_std_deg": None,
        "roll_mean_deg": None,
        "roll_circular_variance": None,
        "mean_turning_angle_deg": None,
        "mean_turning_angle_std_deg": None,
        "max_msd": None,
    }
    if not valid_records:
        return summary

    lengths = np.array([record["filament_length"] for record in valid_records], dtype=float)
    tortuosities = np.array([record["tortuosity"] for record in valid_records], dtype=float)
    yaw_angles = np.array([record["yaw_deg"] for record in valid_records], dtype=float)
    pitch_angles = np.array([record["pitch_deg"] for record in valid_records], dtype=float)
    roll_angles = np.array([record["roll_deg"] for record in valid_records], dtype=float)
    turning_means = np.array([record["turning_angle_mean_deg"] for record in valid_records], dtype=float)
    turning_stds = np.sqrt(
        np.maximum(
            0.0,
            np.array([record["turning_angle_variance_deg2"] for record in valid_records], dtype=float),
        )
    )

    summary.update(
        {
            "mean_length": round_value(np.nanmean(lengths)),
            "std_length": round_value(np.nanstd(lengths)),
            "mean_tortuosity": round_value(np.nanmean(tortuosities)),
            "yaw_mean_deg": round_value(circular_mean_deg(yaw_angles)),
            "yaw_circular_variance": round_value(circular_variance_deg(yaw_angles)),
            "pitch_mean_deg": round_value(np.nanmean(pitch_angles)),
            "pitch_std_deg": round_value(np.nanstd(pitch_angles)),
            "roll_mean_deg": round_value(circular_mean_deg(roll_angles)),
            "roll_circular_variance": round_value(circular_variance_deg(roll_angles)),
            "mean_turning_angle_deg": round_value(np.nanmean(turning_means)),
            "mean_turning_angle_std_deg": round_value(np.nanmean(turning_stds)),
            "max_msd": round_value(max((row["msd"] for row in msd_rows), default=float("nan"))),
        }
    )
    return summary


def principal_axis_line(centroid: np.ndarray, basis: np.ndarray | None, half_length: float) -> np.ndarray:
    if basis is None:
        return np.empty((0, 3), dtype=float)
    direction = basis[0] / (np.linalg.norm(basis[0]) + 1e-12)
    return np.vstack([centroid - direction * half_length, centroid + direction * half_length])


def sample_volume_points(
    volume: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    brightness_floor: float,
    gamma: float,
    z_scale: float,
) -> np.ndarray:
    base_weights = np.clip(volume - brightness_floor, a_min=0.0, a_max=None)
    weights = base_weights**gamma
    flat_weights = weights.ravel()
    if flat_weights.sum() == 0:
        return np.empty((0, 3), dtype=float)

    probabilities = flat_weights / flat_weights.sum()
    flat_indices = rng.choice(len(probabilities), size=n_samples, replace=True, p=probabilities)
    z_indices, row_indices, col_indices = np.unravel_index(flat_indices, volume.shape)
    x_coords = col_indices + rng.uniform(-0.5, 0.5, size=n_samples)
    y_coords = -(row_indices + rng.uniform(-0.5, 0.5, size=n_samples))
    z_coords = z_scale * (z_indices + rng.uniform(-0.5, 0.5, size=n_samples))
    return np.column_stack((x_coords, y_coords, z_coords))


def projection_images_from_volume(volume: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "xy": volume.max(axis=0),
        "xz": volume.max(axis=1),
        "yz": volume.max(axis=2),
    }


def detect_dense_signal(
    point_cloud: np.ndarray,
    eps_values: np.ndarray,
    min_linearity: float,
    min_linear_density: float,
    min_mean_degree: float,
    score_fraction_threshold: float,
) -> dict[str, object]:
    if len(point_cloud) == 0:
        return {
            "point_cloud": point_cloud,
            "component_counts": [0 for _ in eps_values],
            "diagnostics": [
                {
                    "epsilon": float(eps),
                    "score": float("-inf"),
                    "size": 0,
                    "major_span": 0.0,
                    "linearity": 0.0,
                    "linear_density": 0.0,
                    "mean_degree": 0.0,
                    "is_valid": False,
                }
                for eps in eps_values
            ],
            "has_dense_signal": False,
            "chosen": {
                "epsilon": float("nan"),
                "score": float("-inf"),
                "size": 0,
                "major_span": 0.0,
                "linearity": 0.0,
                "linear_density": 0.0,
                "mean_degree": 0.0,
            },
            "recovered_points": np.empty((0, 3), dtype=float),
        }

    sq_dists = np.sum((point_cloud[:, None, :] - point_cloud[None, :, :]) ** 2, axis=2)
    min_component_size = max(30, int(0.01 * len(point_cloud)))
    component_counts = []
    diagnostics: list[dict[str, object]] = []

    for epsilon in eps_values:
        adjacency = epsilon_graph_from_sq_dists(sq_dists, float(epsilon))
        component_counts.append(len(connected_components_from_adjacency(adjacency)))

        recovered_mask_eps, stat = strongest_dense_line_component_nd_from_sq_dists(
            point_cloud,
            sq_dists,
            float(epsilon),
            min_component_size=min_component_size,
            min_linearity=min_linearity,
            min_linear_density=min_linear_density,
            min_mean_degree=min_mean_degree,
        )
        diagnostics.append(
            {
                "epsilon": float(epsilon),
                "mask": recovered_mask_eps,
                "score": stat["score"],
                "size": stat["size"],
                "major_span": stat["major_span"],
                "linearity": stat["linearity"],
                "linear_density": stat["linear_density"],
                "mean_degree": stat["mean_degree"],
                "is_valid": recovered_mask_eps is not None,
            }
        )

    valid_indices = [index for index, diagnostic in enumerate(diagnostics) if diagnostic["is_valid"]]
    if valid_indices:
        valid_scores = np.array([diagnostics[index]["score"] for index in valid_indices], dtype=float)
        max_valid_score = valid_scores.max()
        chosen_index = next(
            index
            for index in valid_indices
            if diagnostics[index]["score"] >= score_fraction_threshold * max_valid_score
        )
        chosen = diagnostics[chosen_index]
        recovered_mask = chosen["mask"]
        has_dense_signal = True
    else:
        chosen = {
            "epsilon": float("nan"),
            "score": float("-inf"),
            "size": 0,
            "major_span": 0.0,
            "linearity": 0.0,
            "linear_density": 0.0,
            "mean_degree": 0.0,
        }
        recovered_mask = np.zeros(len(point_cloud), dtype=bool)
        has_dense_signal = False

    return {
        "point_cloud": point_cloud,
        "component_counts": component_counts,
        "diagnostics": diagnostics,
        "has_dense_signal": has_dense_signal,
        "chosen": chosen,
        "recovered_points": point_cloud[recovered_mask],
    }


def projection_paths_for_track(track_dir: Path, time_index: int) -> dict[str, str]:
    return {
        "xy": f"./assets/3d/{track_dir.name}/xy_{time_index:03d}.png",
        "xz": f"./assets/3d/{track_dir.name}/xz_{time_index:03d}.png",
        "yz": f"./assets/3d/{track_dir.name}/yz_{time_index:03d}.png",
    }


def export_3d_dataset(
    input_paths: list[Path],
    output_dir: Path,
    channel_indices: list[int],
    time_stride: int,
    n_samples: int,
    brightness_floor: float,
    gamma: float,
    z_scale: float,
    normalization_mode: str,
    normalization_percentiles: tuple[float, float],
    eps_values: np.ndarray,
    min_linearity: float,
    min_linear_density: float,
    min_mean_degree: float,
    score_fraction_threshold: float,
    frame_interval: float,
    max_points_for_length: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    all_track_frames: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    all_track_dimensions: dict[tuple[str, int], tuple[int, int, int]] = {}

    assets_root = output_dir / "assets" / "3d"
    assets_root.mkdir(parents=True, exist_ok=True)

    for tif_path in input_paths:
        with tifffile.TiffFile(tif_path) as tif:
            hyperstack = tif.asarray()

        if hyperstack.ndim != 5:
            raise ValueError(f"Expected a TZCYX hyperstack in {tif_path.name}, got shape {hyperstack.shape}.")

        n_timepoints, depth, n_channels, height, width = hyperstack.shape
        invalid_channel_indices = [index for index in channel_indices if index < 0 or index >= n_channels]
        if invalid_channel_indices:
            raise ValueError(
                f"Selected channel indices {invalid_channel_indices} are invalid for {tif_path.name}. "
                f"Available channels are 0 to {n_channels - 1}."
            )

        time_indices = list(range(0, n_timepoints, time_stride)) or [0]

        for channel_index in channel_indices:
            channel_stack = hyperstack[:, :, channel_index].astype(float)
            if normalization_mode == "per_volume":
                normalization_bounds = None
            else:
                normalization_bounds = normalization_bounds_from_volume_stack(
                    channel_stack,
                    mode=normalization_mode,
                    percentiles=normalization_percentiles,
                )

            track_key = (tif_path.name, channel_index)
            all_track_dimensions[track_key] = (depth, height, width)
            track_dir = assets_root / slugify(f"{tif_path.stem}-c{channel_index}")
            track_dir.mkdir(parents=True, exist_ok=True)

            for time_index in time_indices:
                raw_volume = hyperstack[time_index, :, channel_index].astype(float)
                if normalization_mode == "per_volume":
                    normalized_volume = normalize_volume_to_unit_interval(
                        raw_volume,
                        float(raw_volume.min()),
                        float(raw_volume.max()),
                    )
                else:
                    assert normalization_bounds is not None
                    normalized_volume = normalize_volume_to_unit_interval(
                        raw_volume,
                        normalization_bounds[0],
                        normalization_bounds[1],
                    )

                projections = projection_images_from_volume(normalized_volume)
                projection_paths = projection_paths_for_track(track_dir, time_index)
                save_grayscale_png(projections["xy"], output_dir / projection_paths["xy"].replace("./", ""))
                save_grayscale_png(projections["xz"], output_dir / projection_paths["xz"].replace("./", ""))
                save_grayscale_png(projections["yz"], output_dir / projection_paths["yz"].replace("./", ""))

                point_cloud = sample_volume_points(
                    normalized_volume,
                    rng=rng,
                    n_samples=n_samples,
                    brightness_floor=brightness_floor,
                    gamma=gamma,
                    z_scale=z_scale,
                )
                detection = detect_dense_signal(
                    point_cloud=point_cloud,
                    eps_values=eps_values,
                    min_linearity=min_linearity,
                    min_linear_density=min_linear_density,
                    min_mean_degree=min_mean_degree,
                    score_fraction_threshold=score_fraction_threshold,
                )

                all_track_frames[track_key].append(
                    {
                        "file_name": tif_path.name,
                        "channel_index": channel_index,
                        "time_index": time_index,
                        "time": time_index * frame_interval,
                        "label": f"{tif_path.name} | t={time_index}, c={channel_index}",
                        "projection_paths": projection_paths,
                        "volume_shape": {"depth": depth, "height": height, "width": width},
                        "point_cloud": detection["point_cloud"],
                        "component_counts": detection["component_counts"],
                        "diagnostics": detection["diagnostics"],
                        "has_dense_signal": detection["has_dense_signal"],
                        "chosen": detection["chosen"],
                        "recovered_points": detection["recovered_points"],
                    }
                )

    track_payloads = []
    summary_rows = []
    volumes_total = 0
    volumes_with_filament = 0

    for (file_name, channel_index), frames in sorted(all_track_frames.items()):
        frames = sorted(frames, key=lambda item: item["time_index"])
        track_id = slugify(f"{Path(file_name).stem}-c{channel_index}")
        previous_basis: np.ndarray | None = None
        time_series = []
        volume_payloads = []

        for frame in frames:
            volumes_total += 1
            volumes_with_filament += int(frame["has_dense_signal"])

            if frame["has_dense_signal"] and len(frame["recovered_points"]) >= 2:
                metrics = filament_frame_metrics(frame["recovered_points"], max_points_for_length=max_points_for_length)
                basis = align_basis_to_previous(metrics["_basis"], previous_basis)
                metrics["_basis"] = basis
                if basis is not None:
                    metrics.update(orientation_angles_from_basis(basis))
                    previous_basis = basis
            else:
                metrics = empty_frame_metrics()

            basis = metrics["_basis"] if isinstance(metrics.get("_basis"), np.ndarray) else None
            centroid = np.array(
                [
                    metrics["centroid_x"] if is_finite_number(metrics["centroid_x"]) else 0.0,
                    metrics["centroid_y"] if is_finite_number(metrics["centroid_y"]) else 0.0,
                    metrics["centroid_z"] if is_finite_number(metrics["centroid_z"]) else 0.0,
                ],
                dtype=float,
            )
            axis_length = max(float(frame["chosen"]["major_span"]) / 2.0, 8.0) if frame["has_dense_signal"] else 0.0
            principal_axis = principal_axis_line(centroid, basis, half_length=axis_length)
            backbone_points = metrics["_backbone_points"] if isinstance(metrics.get("_backbone_points"), np.ndarray) else np.empty((0, 3), dtype=float)

            frame_row = {
                "file_name": file_name,
                "channel_index": channel_index,
                "track_id": track_id,
                "label": frame["label"],
                "time_index": int(frame["time_index"]),
                "time": round_value(frame["time"]),
                "has_filament": bool(frame["has_dense_signal"]),
                "n_recovered_points": int(len(frame["recovered_points"])),
                "chosen_eps": round_value(frame["chosen"]["epsilon"]) if frame["has_dense_signal"] else None,
                "chosen_score": round_value(frame["chosen"]["score"]) if frame["has_dense_signal"] else None,
                "chosen_span": round_value(frame["chosen"]["major_span"]) if frame["has_dense_signal"] else None,
                "chosen_linearity": round_value(frame["chosen"]["linearity"]) if frame["has_dense_signal"] else None,
                "chosen_linear_density": round_value(frame["chosen"]["linear_density"]) if frame["has_dense_signal"] else None,
                "chosen_mean_degree": round_value(frame["chosen"]["mean_degree"]) if frame["has_dense_signal"] else None,
                "filament_length": round_value(metrics["filament_length"]),
                "end_to_end_distance": round_value(metrics["end_to_end_distance"]),
                "tortuosity": round_value(metrics["tortuosity"]),
                "yaw_deg": round_value(metrics["yaw_deg"]),
                "pitch_deg": round_value(metrics["pitch_deg"]),
                "roll_deg": round_value(metrics["roll_deg"]),
                "tangent_yaw_mean_deg": round_value(metrics["tangent_yaw_mean_deg"]),
                "tangent_yaw_circular_variance": round_value(metrics["tangent_yaw_circular_variance"]),
                "tangent_pitch_mean_deg": round_value(metrics["tangent_pitch_mean_deg"]),
                "tangent_pitch_variance_deg2": round_value(metrics["tangent_pitch_variance_deg2"]),
                "turning_angle_mean_deg": round_value(metrics["turning_angle_mean_deg"]),
                "turning_angle_variance_deg2": round_value(metrics["turning_angle_variance_deg2"]),
                "centroid_x": round_value(metrics["centroid_x"]),
                "centroid_y": round_value(-metrics["centroid_y"] if is_finite_number(metrics["centroid_y"]) else float("nan")),
                "centroid_z": round_value(metrics["centroid_z"]),
            }
            time_series.append(frame_row)

            volume_payloads.append(
                {
                    "time_index": int(frame["time_index"]),
                    "time": round_value(frame["time"]),
                    "label": frame["label"],
                    "has_filament": bool(frame["has_dense_signal"]),
                    "image_paths": frame["projection_paths"],
                    "volume_shape": frame["volume_shape"],
                    "metrics": frame_row,
                    "point_cloud_preview": round_points(downsample_points(frame["point_cloud"], 500)),
                    "recovered_points": round_points(downsample_points(frame["recovered_points"], 900)),
                    "backbone_path": round_points(backbone_points),
                    "principal_axis": round_points(principal_axis),
                    "diagnostics": {
                        "eps_values": [round_value(value) for value in eps_values],
                        "component_counts": [int(value) for value in frame["component_counts"]],
                        "dense_line_scores": [round_value(item["score"]) for item in frame["diagnostics"]],
                        "line_sizes": [int(item["size"]) for item in frame["diagnostics"]],
                        "line_spans": [round_value(item["major_span"]) for item in frame["diagnostics"]],
                        "line_linearities": [round_value(item["linearity"]) for item in frame["diagnostics"]],
                        "dense_line_densities": [round_value(item["linear_density"]) for item in frame["diagnostics"]],
                        "dense_line_degrees": [round_value(item["mean_degree"]) for item in frame["diagnostics"]],
                        "chosen_eps": round_value(frame["chosen"]["epsilon"]) if frame["has_dense_signal"] else None,
                    },
                }
            )

        valid_time_rows = [row for row in time_series if is_finite_number(row["centroid_x"]) and is_finite_number(row["centroid_y"]) and is_finite_number(row["centroid_z"])]
        if valid_time_rows:
            centroid_positions = np.array(
                [[row["centroid_x"], row["centroid_y"], row["centroid_z"]] for row in valid_time_rows],
                dtype=float,
            )
            time_indices = np.array([int(row["time_index"]) for row in valid_time_rows], dtype=int)
            msd_rows = compute_centroid_msd(time_indices, centroid_positions, frame_interval=frame_interval)
        else:
            msd_rows = []

        msd_rows_payload = [
            {
                "lag_frames": int(row["lag_frames"]),
                "lag_time": round_value(row["lag_time"]),
                "msd": round_value(row["msd"]),
                "n_pairs": int(row["n_pairs"]),
            }
            for row in msd_rows
        ]

        track_summary = summarize_track(time_series, msd_rows, file_name=file_name, channel_index=channel_index)
        summary_rows.append(track_summary)
        track_payloads.append(
            {
                "track_id": track_id,
                "track_label": f"{file_name} | c={channel_index}",
                "file_name": file_name,
                "channel_index": channel_index,
                "volume_shape": {
                    "depth": all_track_dimensions[(file_name, channel_index)][0],
                    "height": all_track_dimensions[(file_name, channel_index)][1],
                    "width": all_track_dimensions[(file_name, channel_index)][2],
                },
                "summary": track_summary,
                "frames": volume_payloads,
                "time_series": time_series,
                "msd": msd_rows_payload,
            }
        )

    global_summary = {
        "tracks_total": len(track_payloads),
        "volumes_total": volumes_total,
        "volumes_with_filament": volumes_with_filament,
        "detection_rate": round_value(volumes_with_filament / max(volumes_total, 1)),
    }

    payload = {
        "metadata": {
            "title": "3D TDA Volume Explorer",
            "source_files": [str(path) for path in input_paths],
            "time_units": "frames",
            "frame_interval": frame_interval,
            "summary": global_summary,
            "parameters": {
                "channels": channel_indices,
                "time_stride": time_stride,
                "n_samples": n_samples,
                "brightness_floor": brightness_floor,
                "gamma": gamma,
                "z_scale": z_scale,
                "normalization_mode": normalization_mode,
                "normalization_percentiles": list(normalization_percentiles),
                "min_linearity": min_linearity,
                "min_linear_density": min_linear_density,
                "min_mean_degree": min_mean_degree,
                "score_fraction_threshold": score_fraction_threshold,
                "max_points_for_length": max_points_for_length,
                "seed": seed,
            },
            "notebook_mapping": [
                "Weighted sampling is performed in 3D from each TZCYX volume so bright voxels contribute more points to the cloud.",
                "For every timepoint, the site reuses the notebook's dense elongated component heuristic across an epsilon sweep.",
                "Recovered 3D filaments are summarized by backbone length, tortuosity, centroid MSD, and PCA-based yaw, pitch, and roll proxies.",
            ],
        },
        "summary": global_summary,
        "tracks": track_payloads,
        "summary_rows": summary_rows,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export notebook-inspired 3D TDA filament results for the website.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("3D/ch20_URA7_URA8_002_hyperstack_crop_45.tif")],
        help="One or more 3D TIFF hyperstacks to analyze.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("site"))
    parser.add_argument("--channels", type=int, nargs="+", default=[1])
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--brightness-floor", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument(
        "--normalization-mode",
        choices=["timeseries_minmax", "timeseries_percentile", "per_volume"],
        default="timeseries_minmax",
    )
    parser.add_argument("--normalization-percentiles", type=float, nargs=2, default=(1.0, 99.5))
    parser.add_argument("--min-linearity", type=float, default=2.5)
    parser.add_argument("--min-linear-density", type=float, default=20.0)
    parser.add_argument("--min-mean-degree", type=float, default=4.0)
    parser.add_argument("--score-fraction-threshold", type=float, default=0.95)
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--max-points-for-length", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = export_3d_dataset(
        input_paths=args.input,
        output_dir=args.output_dir,
        channel_indices=args.channels,
        time_stride=args.time_stride,
        n_samples=args.samples,
        brightness_floor=args.brightness_floor,
        gamma=args.gamma,
        z_scale=args.z_scale,
        normalization_mode=args.normalization_mode,
        normalization_percentiles=tuple(args.normalization_percentiles),
        eps_values=np.asarray(DEFAULT_EPS_VALUES, dtype=float),
        min_linearity=args.min_linearity,
        min_linear_density=args.min_linear_density,
        min_mean_degree=args.min_mean_degree,
        score_fraction_threshold=args.score_fraction_threshold,
        frame_interval=args.frame_interval,
        max_points_for_length=args.max_points_for_length,
        seed=args.seed,
    )

    output_path = data_dir / "tda_3d_volumes.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote 3D projection PNGs to {args.output_dir / 'assets' / '3d'}")


if __name__ == "__main__":
    main()
