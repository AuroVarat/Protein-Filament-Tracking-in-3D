#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile


DEFAULT_EPS_VALUES = np.linspace(1.0, 12.0, 30)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    return (image - image.min()) / (image.max() - image.min() + 1e-12)


def weighted_point_cloud(
    image_2d: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    brightness_floor: float,
    gamma: float,
) -> np.ndarray:
    base_weights = np.clip(image_2d - brightness_floor, a_min=0.0, a_max=None)
    weights = base_weights**gamma
    flat_weights = weights.ravel()
    if flat_weights.sum() == 0:
        return np.empty((0, 2), dtype=float)

    probabilities = flat_weights / flat_weights.sum()
    flat_indices = rng.choice(len(probabilities), size=n_samples, replace=True, p=probabilities)
    rows, cols = np.unravel_index(flat_indices, image_2d.shape)
    x = cols + rng.uniform(-0.5, 0.5, size=n_samples)
    y = rows + rng.uniform(-0.5, 0.5, size=n_samples)
    return np.column_stack((x, -y))


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


def component_dense_line_stats(
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
    major_span = float(spans[0])
    minor_span = float(spans[1]) if spans.shape[0] > 1 else 0.0
    linearity = major_span / (minor_span + 1e-6)

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
        "minor_span": minor_span,
        "linearity": float(linearity),
        "linear_density": float(linear_density),
        "mean_degree": mean_degree,
        "score": float(score),
    }


def strongest_dense_line_component_from_sq_dists(
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
    stats = [component_dense_line_stats(point_cloud, adjacency, component, epsilon) for component in components]

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


def principal_basis(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.eye(points.shape[1] if points.ndim == 2 else 2)

    centred = points - points.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order].T


def subsample_points_along_axis(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points

    basis = principal_basis(points)
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


def angle_moments_from_backbone_2d(backbone_points: np.ndarray) -> dict[str, float]:
    if len(backbone_points) < 2:
        return {
            "tangent_angle_mean_deg": float("nan"),
            "tangent_angle_circular_variance": float("nan"),
            "turning_angle_mean_deg": float("nan"),
            "turning_angle_variance_deg2": float("nan"),
        }

    segment_vectors = np.diff(backbone_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_mask = segment_lengths > 1e-8
    if not np.any(valid_mask):
        return {
            "tangent_angle_mean_deg": float("nan"),
            "tangent_angle_circular_variance": float("nan"),
            "turning_angle_mean_deg": float("nan"),
            "turning_angle_variance_deg2": float("nan"),
        }

    unit_tangents = segment_vectors[valid_mask] / segment_lengths[valid_mask, None]
    tangent_angles_deg = np.rad2deg(np.arctan2(unit_tangents[:, 1], unit_tangents[:, 0]))

    turning_angles_deg = np.array([], dtype=float)
    if len(tangent_angles_deg) >= 2:
        wrapped_turns = (np.diff(tangent_angles_deg) + 180.0) % 360.0 - 180.0
        turning_angles_deg = np.abs(wrapped_turns)

    return {
        "tangent_angle_mean_deg": circular_mean_deg(tangent_angles_deg),
        "tangent_angle_circular_variance": circular_variance_deg(tangent_angles_deg),
        "turning_angle_mean_deg": float(np.mean(turning_angles_deg)) if len(turning_angles_deg) else float("nan"),
        "turning_angle_variance_deg2": float(np.var(turning_angles_deg)) if len(turning_angles_deg) else float("nan"),
    }


def filament_frame_metrics(recovered_points: np.ndarray, max_points_for_length: int) -> dict[str, float | np.ndarray]:
    recovered_points = np.asarray(recovered_points, dtype=float)
    if len(recovered_points) < 2:
        return {
            "centroid_x": float("nan"),
            "centroid_y": float("nan"),
            "filament_length": float("nan"),
            "end_to_end_distance": float("nan"),
            "tortuosity": float("nan"),
            "orientation_deg": float("nan"),
            "tangent_angle_mean_deg": float("nan"),
            "tangent_angle_circular_variance": float("nan"),
            "turning_angle_mean_deg": float("nan"),
            "turning_angle_variance_deg2": float("nan"),
            "backbone_points": np.empty((0, 2), dtype=float),
            "principal_axis": np.empty((0, 2), dtype=float),
        }

    centroid = recovered_points.mean(axis=0)
    basis = principal_basis(recovered_points)
    backbone_points, filament_length = diameter_path_from_points(recovered_points, max_points=max_points_for_length)
    end_to_end_distance = (
        float(np.linalg.norm(backbone_points[-1] - backbone_points[0])) if len(backbone_points) >= 2 else float("nan")
    )
    orientation_deg = float(np.rad2deg(np.arctan2(basis[0, 1], basis[0, 0])))
    principal_axis = np.vstack([centroid - basis[0] * 14.0, centroid + basis[0] * 14.0])

    metrics: dict[str, float | np.ndarray] = {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "filament_length": float(filament_length),
        "end_to_end_distance": end_to_end_distance,
        "tortuosity": float(filament_length / (end_to_end_distance + 1e-12)) if np.isfinite(end_to_end_distance) else float("nan"),
        "orientation_deg": orientation_deg,
        "backbone_points": backbone_points,
        "principal_axis": principal_axis,
    }
    metrics.update(angle_moments_from_backbone_2d(backbone_points))
    return metrics


def compute_centroid_msd(frame_rows: list[dict[str, float]], frame_interval: float) -> list[dict[str, float]]:
    valid_rows = [
        row for row in frame_rows if is_finite_number(row["centroid_x"]) and is_finite_number(row["centroid_y"])
    ]
    if len(valid_rows) < 2:
        return []

    time_indices = np.array([int(row["frame_index"]) for row in valid_rows], dtype=int)
    positions = np.array([[row["centroid_x"], row["centroid_y"]] for row in valid_rows], dtype=float)

    displacements_by_lag: dict[int, list[float]] = defaultdict(list)
    for start in range(len(time_indices) - 1):
        for stop in range(start + 1, len(time_indices)):
            lag_frames = int(time_indices[stop] - time_indices[start])
            if lag_frames <= 0:
                continue
            sq_displacement = float(np.sum((positions[stop] - positions[start]) ** 2))
            displacements_by_lag[lag_frames].append(sq_displacement)

    rows = []
    for lag_frames in sorted(displacements_by_lag):
        values = np.asarray(displacements_by_lag[lag_frames], dtype=float)
        rows.append(
            {
                "lag_frames": lag_frames,
                "lag_time": float(lag_frames * frame_interval),
                "msd": float(np.mean(values)),
                "n_pairs": int(len(values)),
            }
        )
    return rows


def round_points(points: np.ndarray, digits: int = 2) -> list[list[float]]:
    if len(points) == 0:
        return []
    return np.round(np.asarray(points, dtype=float), digits).tolist()


def round_value(value: float | int | bool | None, digits: int = 4) -> float | int | bool | None:
    if isinstance(value, (bool, int)) or value is None:
        return value
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def to_image_coordinates(points: np.ndarray) -> list[list[float]]:
    if len(points) == 0:
        return []
    converted = np.asarray(points, dtype=float).copy()
    converted[:, 1] *= -1.0
    return round_points(converted)


def save_frame_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image, cmap="gray", vmin=0.0, vmax=1.0)


def analyze_stack(
    tif_path: Path,
    output_dir: Path,
    n_samples: int,
    brightness_floor: float,
    gamma: float,
    min_linearity: float,
    min_linear_density: float,
    min_mean_degree: float,
    score_fraction_threshold: float,
    frame_interval: float,
    max_points_for_length: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    eps_values = np.asarray(DEFAULT_EPS_VALUES, dtype=float)

    with tifffile.TiffFile(tif_path) as tif:
        pages = [normalize_image(page.asarray()) for page in tif.pages]

    frames_dir = output_dir / "assets" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    frame_payloads = []
    chosen_scores = []
    detection_count = 0

    for frame_index, image_2d in enumerate(pages):
        point_cloud = weighted_point_cloud(
            image_2d=image_2d,
            rng=rng,
            n_samples=n_samples,
            brightness_floor=brightness_floor,
            gamma=gamma,
        )
        if len(point_cloud) == 0:
            sq_dists = np.empty((0, 0), dtype=float)
        else:
            sq_dists = np.sum((point_cloud[:, None, :] - point_cloud[None, :, :]) ** 2, axis=2)

        component_counts = []
        diagnostics = []
        min_component_size = max(30, int(0.01 * max(len(point_cloud), 1)))

        for epsilon in eps_values:
            if len(point_cloud) == 0:
                component_counts.append(0)
                diagnostics.append(
                    {
                        "epsilon": float(epsilon),
                        "mask": None,
                        "score": float("-inf"),
                        "size": 0,
                        "major_span": 0.0,
                        "linearity": 0.0,
                        "linear_density": 0.0,
                        "mean_degree": 0.0,
                        "is_valid": False,
                    }
                )
                continue

            adjacency = epsilon_graph_from_sq_dists(sq_dists, float(epsilon))
            component_counts.append(len(connected_components_from_adjacency(adjacency)))

            recovered_mask_eps, stat = strongest_dense_line_component_from_sq_dists(
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

        valid_indices = [index for index, item in enumerate(diagnostics) if item["is_valid"]]
        if valid_indices:
            valid_scores = np.array([diagnostics[index]["score"] for index in valid_indices], dtype=float)
            max_valid_score = valid_scores.max()
            chosen_idx = next(
                index
                for index in valid_indices
                if diagnostics[index]["score"] >= score_fraction_threshold * max_valid_score
            )
            chosen = diagnostics[chosen_idx]
            recovered_mask = chosen["mask"]
            has_filament = True
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
            has_filament = False

        recovered_points = point_cloud[recovered_mask]
        metrics = filament_frame_metrics(recovered_points, max_points_for_length=max_points_for_length)

        detection_count += int(has_filament)
        chosen_scores.append(float(chosen["score"]) if np.isfinite(chosen["score"]) else np.nan)

        image_path = frames_dir / f"frame_{frame_index:03d}.png"
        save_frame_png(image_2d, image_path)

        frame_row = {
            "frame_index": frame_index,
            "time": frame_index * frame_interval,
            "has_filament": has_filament,
            "n_recovered_points": int(len(recovered_points)),
            "chosen_eps": round_value(chosen["epsilon"]) if has_filament else None,
            "chosen_score": round_value(chosen["score"]) if has_filament else None,
            "chosen_span": round_value(chosen["major_span"]) if has_filament else None,
            "chosen_linearity": round_value(chosen["linearity"]) if has_filament else None,
            "chosen_linear_density": round_value(chosen["linear_density"]) if has_filament else None,
            "chosen_mean_degree": round_value(chosen["mean_degree"]) if has_filament else None,
            "filament_length": round_value(metrics["filament_length"]),
            "end_to_end_distance": round_value(metrics["end_to_end_distance"]),
            "tortuosity": round_value(metrics["tortuosity"]),
            "orientation_deg": round_value(metrics["orientation_deg"]),
            "tangent_angle_mean_deg": round_value(metrics["tangent_angle_mean_deg"]),
            "tangent_angle_circular_variance": round_value(metrics["tangent_angle_circular_variance"]),
            "turning_angle_mean_deg": round_value(metrics["turning_angle_mean_deg"]),
            "turning_angle_variance_deg2": round_value(metrics["turning_angle_variance_deg2"]),
            "centroid_x": round_value(metrics["centroid_x"]),
            "centroid_y": round_value(-metrics["centroid_y"] if np.isfinite(metrics["centroid_y"]) else float("nan")),
        }
        frame_rows.append(frame_row)

        frame_payloads.append(
            {
                "frame_index": frame_index,
                "time": round_value(frame_index * frame_interval),
                "image_path": f"./assets/frames/frame_{frame_index:03d}.png",
                "has_filament": has_filament,
                "metrics": frame_row,
                "filament_points": to_image_coordinates(recovered_points),
                "backbone_path": to_image_coordinates(np.asarray(metrics["backbone_points"])),
                "principal_axis": to_image_coordinates(np.asarray(metrics["principal_axis"])),
                "diagnostics": {
                    "eps_values": [round_value(value) for value in eps_values],
                    "component_counts": [int(value) for value in component_counts],
                    "dense_line_scores": [round_value(item["score"]) for item in diagnostics],
                    "line_sizes": [int(item["size"]) for item in diagnostics],
                    "line_spans": [round_value(item["major_span"]) for item in diagnostics],
                    "line_linearities": [round_value(item["linearity"]) for item in diagnostics],
                    "dense_line_densities": [round_value(item["linear_density"]) for item in diagnostics],
                    "dense_line_degrees": [round_value(item["mean_degree"]) for item in diagnostics],
                    "chosen_eps": round_value(chosen["epsilon"]),
                },
            }
        )

    msd_rows = compute_centroid_msd(frame_rows, frame_interval=frame_interval)

    valid_lengths = np.array(
        [row["filament_length"] for row in frame_rows if row["has_filament"] and row["filament_length"] is not None],
        dtype=float,
    )
    valid_linearity = np.array(
        [row["chosen_linearity"] for row in frame_rows if row["has_filament"] and row["chosen_linearity"] is not None],
        dtype=float,
    )
    valid_density = np.array(
        [
            row["chosen_linear_density"]
            for row in frame_rows
            if row["has_filament"] and row["chosen_linear_density"] is not None
        ],
        dtype=float,
    )

    summary = {
        "frames_total": len(frame_rows),
        "frames_with_filament": detection_count,
        "detection_rate": round_value(detection_count / max(len(frame_rows), 1)),
        "mean_length": round_value(np.nanmean(valid_lengths) if len(valid_lengths) else float("nan")),
        "mean_linearity": round_value(np.nanmean(valid_linearity) if len(valid_linearity) else float("nan")),
        "mean_density": round_value(np.nanmean(valid_density) if len(valid_density) else float("nan")),
        "best_frame_score": round_value(np.nanmax(np.asarray(chosen_scores, dtype=float))),
        "max_msd": round_value(max((row["msd"] for row in msd_rows), default=float("nan"))),
    }

    return {
        "metadata": {
            "title": "TDA Filament Timeline Explorer",
            "source_tif": str(tif_path),
            "frame_dimensions": {"width": pages[0].shape[1], "height": pages[0].shape[0]},
            "frame_interval": frame_interval,
            "time_units": "frames",
            "parameters": {
                "n_samples": n_samples,
                "brightness_floor": brightness_floor,
                "gamma": gamma,
                "min_linearity": min_linearity,
                "min_linear_density": min_linear_density,
                "min_mean_degree": min_mean_degree,
                "score_fraction_threshold": score_fraction_threshold,
                "max_points_for_length": max_points_for_length,
                "seed": seed,
            },
            "summary": summary,
            "notebook_mapping": [
                "Weighted sampling from bright voxels approximates the notebook's TIFF-to-point-cloud step.",
                "The chosen filament is the earliest dense elongated component that stays near the peak dense-line score.",
                "Timeline metrics summarize how the recovered filament changes across the 100-frame sequence.",
            ],
        },
        "summary": summary,
        "frames": frame_payloads,
        "time_series": frame_rows,
        "msd": [
            {
                "lag_frames": int(row["lag_frames"]),
                "lag_time": round_value(row["lag_time"]),
                "msd": round_value(row["msd"]),
                "n_pairs": int(row["n_pairs"]),
            }
            for row in msd_rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export notebook-inspired TDA filament results for the website.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("2D/ch20_URA7_URA8_001-crop1.tif"),
        help="Input TIFF stack to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("site"),
        help="Directory where the site data and frame images should be written.",
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of weighted samples per frame.")
    parser.add_argument("--brightness-floor", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--min-linearity", type=float, default=2.0)
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

    payload = analyze_stack(
        tif_path=args.input,
        output_dir=args.output_dir,
        n_samples=args.samples,
        brightness_floor=args.brightness_floor,
        gamma=args.gamma,
        min_linearity=args.min_linearity,
        min_linear_density=args.min_linear_density,
        min_mean_degree=args.min_mean_degree,
        score_fraction_threshold=args.score_fraction_threshold,
        frame_interval=args.frame_interval,
        max_points_for_length=args.max_points_for_length,
        seed=args.seed,
    )

    output_path = data_dir / "tda_timeline.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote frame PNGs to {args.output_dir / 'assets' / 'frames'}")


if __name__ == "__main__":
    main()
