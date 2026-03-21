from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from .dataset_index import build_dataset_index, save_dataset_index
from .io_utils import read_stack_frame, read_tiff
from .utils import ensure_dir, percentile_summary, robust_mean_std, save_json, to_builtin


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled = measure.label(mask > 0)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def _clean_binary(mask: np.ndarray, min_size: int = 8) -> np.ndarray:
    cleaned = morphology.remove_small_objects(mask > 0, min_size=min_size)
    return _largest_component(cleaned)


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return mask > 0
    footprint = morphology.disk(int(dilation_px))
    return morphology.binary_dilation(mask > 0, footprint=footprint)


def _skeleton_path(mask: np.ndarray) -> tuple[np.ndarray, int]:
    skeleton = morphology.skeletonize(mask > 0)
    coords = np.argwhere(skeleton)
    if coords.size == 0:
        return np.empty((0, 2), dtype=float), 0

    component_labels = measure.label(skeleton, connectivity=2)
    component_count = int(component_labels.max())
    if component_count > 1:
        counts = np.bincount(component_labels.ravel())
        counts[0] = 0
        skeleton = component_labels == int(np.argmax(counts))
        coords = np.argwhere(skeleton)

    coord_set = {tuple(c) for c in coords}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coord_set:
        pts = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in coord_set:
                    pts.append(nxt)
        neighbors[(r, c)] = pts

    endpoints = [node for node, nb in neighbors.items() if len(nb) == 1]
    start = endpoints[0] if endpoints else tuple(coords[0])
    ordered = [start]
    prev = None
    current = start
    visited = {start}
    while True:
        candidates = [node for node in neighbors[current] if node != prev and node not in visited]
        if not candidates:
            break
        nxt = candidates[0]
        ordered.append(nxt)
        visited.add(nxt)
        prev = current
        current = nxt

    return np.asarray(ordered, dtype=float), component_count


def _path_length(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def _path_curvature(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 3:
        return np.empty(0, dtype=float)
    deltas = np.diff(coords, axis=0)
    angles = np.arctan2(deltas[:, 0], deltas[:, 1])
    dtheta = np.diff(np.unwrap(angles))
    segment_lengths = np.maximum(np.sqrt(np.sum(deltas[1:] ** 2, axis=1)), 1e-6)
    return np.abs(dtheta) / segment_lengths


def _orientation_from_endpoints(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return float("nan")
    delta = coords[-1] - coords[0]
    return float(math.atan2(delta[0], delta[1]))


def _blur_proxy(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    lap = ndi.laplace(image.astype(float))
    values = lap[mask] if mask is not None and np.any(mask) else lap.ravel()
    return float(np.var(values))


def _radial_profile(image: np.ndarray, cell_mask: np.ndarray, centroid: tuple[float, float], radius: float) -> dict[str, float]:
    if radius <= 0 or not np.any(cell_mask):
        return {}
    rr, cc = np.indices(image.shape)
    dist = np.sqrt((rr - centroid[0]) ** 2 + (cc - centroid[1]) ** 2)
    norm_r = dist / max(radius, 1e-6)
    values = image[cell_mask]
    radii = norm_r[cell_mask]
    bins = np.linspace(0, 1.0, 6)
    means = []
    centers = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (radii >= lo) & (radii < hi)
        if np.any(sel):
            means.append(float(np.mean(values[sel])))
            centers.append((lo + hi) * 0.5)
    if len(means) < 2:
        return {}
    slope = np.polyfit(np.asarray(centers), np.asarray(means), deg=1)[0]
    return {"radial_slope": float(slope), "radial_center_minus_edge": float(means[0] - means[-1])}


def _measure_sample(
    raw_image: np.ndarray,
    cell_mask: np.ndarray | None,
    filament_mask: np.ndarray | None,
    sample_id: str,
    filament_dilation_px: int = 0,
) -> dict[str, Any]:
    image = raw_image.astype(float)
    h, w = image.shape
    metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "image_height": h,
        "image_width": w,
        "raw_p01": float(np.percentile(image, 1)),
        "raw_p50": float(np.percentile(image, 50)),
        "raw_p99": float(np.percentile(image, 99)),
        "blur_proxy": _blur_proxy(image),
    }

    if cell_mask is None:
        return metrics

    cell_mask = _clean_binary(cell_mask)
    filament_mask = _clean_binary(filament_mask) if filament_mask is not None else np.zeros_like(cell_mask, dtype=bool)
    filament_mask = _dilate_mask(filament_mask, filament_dilation_px)
    outside = ~cell_mask
    diffuse_mask = cell_mask & ~filament_mask

    if not np.any(cell_mask):
        return metrics

    props = measure.regionprops(measure.label(cell_mask.astype(np.uint8)))[0]
    centroid = tuple(float(x) for x in props.centroid)
    equiv_radius = math.sqrt(props.area / math.pi)
    bg_values = image[outside]
    diffuse_values = image[diffuse_mask] if np.any(diffuse_mask) else np.empty(0, dtype=float)
    filament_values = image[filament_mask] if np.any(filament_mask) else np.empty(0, dtype=float)

    perim = max(float(props.perimeter), 1e-6)
    convex_area = max(float(props.convex_area), 1e-6)
    radial = _radial_profile(image, diffuse_mask if np.any(diffuse_mask) else cell_mask, centroid, equiv_radius)

    metrics.update(
        {
            "bg_mean": float(np.mean(bg_values)) if bg_values.size else float("nan"),
            "bg_std": float(np.std(bg_values)) if bg_values.size else float("nan"),
            "cell_area": float(props.area),
            "cell_equivalent_radius": float(equiv_radius),
            "cell_eccentricity": float(props.eccentricity),
            "cell_circularity": float((4.0 * math.pi * props.area) / (perim**2)),
            "cell_centroid_y_norm": float(centroid[0] / max(h - 1, 1)),
            "cell_centroid_x_norm": float(centroid[1] / max(w - 1, 1)),
            "cell_boundary_smoothness": float(props.area / convex_area),
            "diffuse_mean": float(np.mean(diffuse_values)) if diffuse_values.size else float("nan"),
            "diffuse_std": float(np.std(diffuse_values)) if diffuse_values.size else float("nan"),
            "inside_texture_std": float(np.std(diffuse_values - ndi.gaussian_filter(image, 5)[diffuse_mask])) if diffuse_values.size else float("nan"),
            "cell_blur_proxy": _blur_proxy(image, cell_mask),
        }
    )
    metrics.update(radial)

    if not np.any(filament_mask):
        return metrics

    skeleton_coords, component_count = _skeleton_path(filament_mask)
    arc_length = _path_length(skeleton_coords)
    endpoint_distance = float(np.linalg.norm(skeleton_coords[-1] - skeleton_coords[0])) if len(skeleton_coords) >= 2 else 0.0
    curvature = _path_curvature(skeleton_coords)
    dist_map = ndi.distance_transform_edt(filament_mask)
    local_widths = 2.0 * dist_map[tuple(skeleton_coords.astype(int).T)] if len(skeleton_coords) else np.empty(0, dtype=float)
    skel_centroid = np.mean(skeleton_coords, axis=0) if len(skeleton_coords) else np.asarray(centroid)
    dist_to_center = float(np.linalg.norm(skel_centroid - np.asarray(centroid)))
    cell_dist_map = ndi.distance_transform_edt(cell_mask)
    boundary_dist = cell_dist_map[tuple(skeleton_coords.astype(int).T)] if len(skeleton_coords) else np.empty(0, dtype=float)

    metrics.update(
        {
            "filament_area": float(np.sum(filament_mask)),
            "filament_length": float(arc_length),
            "filament_width_mean": float(np.mean(local_widths)) if local_widths.size else float("nan"),
            "filament_width_std": float(np.std(local_widths)) if local_widths.size else float("nan"),
            "filament_intensity_mean": float(np.mean(filament_values)) if filament_values.size else float("nan"),
            "filament_intensity_median": float(np.median(filament_values)) if filament_values.size else float("nan"),
            "filament_to_diffuse_ratio": float(np.mean(filament_values) / max(np.mean(diffuse_values), 1e-6)) if filament_values.size and diffuse_values.size else float("nan"),
            "filament_minus_diffuse": float(np.mean(filament_values) - np.mean(diffuse_values)) if filament_values.size and diffuse_values.size else float("nan"),
            "filament_minus_background": float(np.mean(filament_values) - np.mean(bg_values)) if filament_values.size and bg_values.size else float("nan"),
            "filament_snr_local": float((np.mean(filament_values) - np.mean(diffuse_values)) / max(np.std(bg_values), 1e-6)) if filament_values.size and diffuse_values.size and bg_values.size else float("nan"),
            "filament_curvature_mean": float(np.mean(curvature)) if curvature.size else float("nan"),
            "filament_curvature_p95": float(np.percentile(curvature, 95)) if curvature.size else float("nan"),
            "filament_endpoint_distance": float(endpoint_distance),
            "filament_tortuosity": float(arc_length / max(endpoint_distance, 1e-6)),
            "filament_orientation_rad": _orientation_from_endpoints(skeleton_coords),
            "filament_position_radius_norm": float(dist_to_center / max(equiv_radius, 1e-6)),
            "filament_boundary_distance_norm": float(np.median(boundary_dist) / max(equiv_radius, 1e-6)) if boundary_dist.size else float("nan"),
            "filament_disconnected_components": int(component_count),
            "filament_near_boundary": float(np.mean(boundary_dist < max(2.0, 0.15 * equiv_radius))) if boundary_dist.size else float("nan"),
        }
    )
    return metrics


def _hist_plot(values: np.ndarray, title: str, path: Path, bins: int = 30) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=bins, color="#4472c4", alpha=0.9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _collect_distribution_payload(df: pd.DataFrame) -> dict[str, Any]:
    metrics_to_keep = [
        "cell_area",
        "cell_equivalent_radius",
        "cell_eccentricity",
        "cell_circularity",
        "cell_centroid_y_norm",
        "cell_centroid_x_norm",
        "cell_boundary_smoothness",
        "filament_area",
        "filament_length",
        "filament_width_mean",
        "filament_width_std",
        "filament_intensity_mean",
        "filament_intensity_median",
        "filament_to_diffuse_ratio",
        "filament_minus_diffuse",
        "filament_minus_background",
        "filament_snr_local",
        "filament_curvature_mean",
        "filament_curvature_p95",
        "filament_endpoint_distance",
        "filament_tortuosity",
        "filament_orientation_rad",
        "filament_position_radius_norm",
        "filament_boundary_distance_norm",
        "filament_near_boundary",
        "bg_mean",
        "bg_std",
        "diffuse_mean",
        "diffuse_std",
        "inside_texture_std",
        "radial_slope",
        "radial_center_minus_edge",
        "blur_proxy",
        "cell_blur_proxy",
    ]
    distributions: dict[str, Any] = {}
    for col in metrics_to_keep:
        if col not in df:
            continue
        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        distributions[col] = {
            "values": values.tolist(),
            "summary": {**robust_mean_std(values), **percentile_summary(values)},
        }
    return distributions


def measure_dataset(data_dir: Path, output_dir: Path, filament_dilation_px: int = 0) -> dict[str, Any]:
    ensure_dir(output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    index = build_dataset_index(data_dir)
    save_dataset_index(index, output_dir / "dataset_index.csv")

    rows: list[dict[str, Any]] = []
    stack_rows: list[dict[str, Any]] = []

    seen_raws: set[str] = set()
    for row in index.itertuples(index=False):
        raw_path = Path(row.raw_path)
        if str(raw_path) not in seen_raws:
            stack = read_tiff(raw_path)
            stack_rows.append(
                {
                    "raw_path": str(raw_path),
                    "ndim": int(stack.ndim),
                    "frames": int(stack.shape[0]) if stack.ndim >= 3 else 1,
                    "height": int(stack.shape[-2]),
                    "width": int(stack.shape[-1]),
                    "dtype": str(stack.dtype),
                    "stack_p01": float(np.percentile(stack.astype(float), 1)),
                    "stack_p50": float(np.percentile(stack.astype(float), 50)),
                    "stack_p99": float(np.percentile(stack.astype(float), 99)),
                }
            )
            seen_raws.add(str(raw_path))

        frame_image = read_stack_frame(raw_path, int(row.frame_index))
        cell_mask = read_tiff(Path(row.cell_mask_path)) if pd.notna(row.cell_mask_path) else None
        filament_mask = read_tiff(Path(row.filament_mask_path)) if pd.notna(row.filament_mask_path) else None

        if cell_mask is not None and cell_mask.ndim > 2:
            cell_mask = np.squeeze(cell_mask)
        if filament_mask is not None and filament_mask.ndim > 2:
            filament_mask = np.squeeze(filament_mask)

        rows.append(
            _measure_sample(
                frame_image,
                cell_mask,
                filament_mask,
                row.sample_id,
                filament_dilation_px=filament_dilation_px,
            )
        )

    sample_df = pd.DataFrame(rows)
    stack_df = pd.DataFrame(stack_rows)
    sample_df.to_csv(output_dir / "sample_measurements.csv", index=False)
    stack_df.to_csv(output_dir / "stack_measurements.csv", index=False)

    for column, title in {
        "cell_area": "Cell Area",
        "cell_equivalent_radius": "Cell Equivalent Radius",
        "filament_length": "Filament Length",
        "filament_width_mean": "Filament Width",
        "filament_intensity_mean": "Filament Intensity",
        "bg_std": "Background Noise Std",
        "filament_snr_local": "Filament SNR",
    }.items():
        if column in sample_df:
            _hist_plot(sample_df[column].to_numpy(dtype=float), title, plots_dir / f"{column}.png")

    relationships = {}
    for x_col, y_col in [
        ("cell_equivalent_radius", "filament_length"),
        ("filament_length", "filament_width_mean"),
        ("filament_width_mean", "filament_intensity_mean"),
        ("cell_equivalent_radius", "filament_position_radius_norm"),
    ]:
        if x_col in sample_df and y_col in sample_df:
            pair = sample_df[[x_col, y_col]].dropna()
            if len(pair) >= 3:
                relationships[f"{x_col}__{y_col}"] = {
                    "pearson_r": float(pair[x_col].corr(pair[y_col])),
                    "count": int(len(pair)),
                }

    summary = {
        "dataset_summary": {
            "raw_stack_count": int(len(stack_df)),
            "frame_record_count": int(len(index)),
            "fully_annotated_count": int(index["has_full_annotation"].sum()) if not index.empty else 0,
            "filament_mask_count": int(index["has_filament_mask"].sum()) if not index.empty else 0,
            "cell_mask_count": int(index["has_cell_mask"].sum()) if not index.empty else 0,
            "filament_dilation_px": int(filament_dilation_px),
        },
        "stack_level": {
            "height": robust_mean_std(stack_df["height"]) if not stack_df.empty else {},
            "width": robust_mean_std(stack_df["width"]) if not stack_df.empty else {},
            "frames": robust_mean_std(stack_df["frames"]) if not stack_df.empty else {},
            "stack_p01": robust_mean_std(stack_df["stack_p01"]) if not stack_df.empty else {},
            "stack_p50": robust_mean_std(stack_df["stack_p50"]) if not stack_df.empty else {},
            "stack_p99": robust_mean_std(stack_df["stack_p99"]) if not stack_df.empty else {},
        },
        "distributions": _collect_distribution_payload(sample_df),
        "relationships": relationships,
    }
    save_json(to_builtin(summary), output_dir / "stats_summary.json")
    return summary
