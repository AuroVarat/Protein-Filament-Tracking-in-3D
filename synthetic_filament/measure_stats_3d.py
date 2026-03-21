from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .dataset_index_3d import build_dataset_index_3d, save_dataset_index_3d
from .utils import ensure_dir, percentile_summary, robust_mean_std, save_json, to_builtin


VOXEL_SIZE_UM = 0.3
Z_STEP_UM = 0.3


def load_fluorescence_volume(path: Path, time_index: int, channel_index: int = 1) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D hyperstack (T,Z,C,Y,X), got shape {arr.shape} for {path}")
    return arr[time_index, :, channel_index].astype(np.float32)


def weak_cell_masks_3d(volume: np.ndarray, expected_max_cells: int = 3) -> tuple[np.ndarray, np.ndarray]:
    smooth = ndi.gaussian_filter(volume, sigma=(0.8, 1.2, 1.2))
    positive = smooth[smooth > 0]
    if positive.size == 0:
        return np.zeros_like(volume, dtype=np.int32), np.zeros_like(volume, dtype=bool)

    threshold = filters.threshold_otsu(positive) if positive.size >= 128 else float(np.percentile(positive, 70))
    binary = smooth > threshold
    binary = morphology.remove_small_objects(binary, min_size=80)
    binary = ndi.binary_fill_holes(binary)
    binary = morphology.binary_closing(binary, footprint=morphology.ball(1))

    if np.sum(binary) == 0:
        return np.zeros_like(volume, dtype=np.int32), np.zeros_like(volume, dtype=bool)

    distance = ndi.distance_transform_edt(binary)
    markers = measure.label(morphology.local_maxima(distance))
    labels = segmentation.watershed(-distance, markers=markers, mask=binary)

    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.area, reverse=True)[:expected_max_cells]
    keep = {int(p.label) for p in props}
    filtered = np.where(np.isin(labels, list(keep)), labels, 0)

    # Relabel compactly
    relabeled = np.zeros_like(filtered, dtype=np.int32)
    for new_label, old_label in enumerate(sorted(keep), start=1):
        relabeled[filtered == old_label] = new_label
    return relabeled, relabeled > 0


def filament_host_cell_labels(cell_labels: np.ndarray, filament_mask: np.ndarray) -> list[int]:
    if np.sum(filament_mask) == 0:
        return []
    labels = cell_labels[filament_mask > 0]
    labels = labels[labels > 0]
    if labels.size == 0:
        return []
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [int(unique[i]) for i in order]


def skeleton_length_3d(mask: np.ndarray) -> float:
    skeleton = morphology.skeletonize(mask > 0)
    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords[np.argsort(coords[:, 0] * 10_000 + coords[:, 1] * 100 + coords[:, 2])], axis=0)
    spacing = np.array([Z_STEP_UM, VOXEL_SIZE_UM, VOXEL_SIZE_UM], dtype=float)
    lengths = np.sqrt(np.sum((diffs * spacing) ** 2, axis=1))
    return float(np.sum(lengths))


def local_width_um(mask: np.ndarray) -> dict[str, float]:
    dist = ndi.distance_transform_edt(mask > 0, sampling=(Z_STEP_UM, VOXEL_SIZE_UM, VOXEL_SIZE_UM))
    skeleton = morphology.skeletonize(mask > 0)
    widths = 2.0 * dist[skeleton]
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return {}
    return {
        "filament_width_um_mean": float(np.mean(widths)),
        "filament_width_um_p95": float(np.percentile(widths, 95)),
    }


def measure_timepoint(
    volume: np.ndarray,
    sample_id: str,
    filament_mask: np.ndarray | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "z_dim": int(volume.shape[0]),
        "y_dim": int(volume.shape[1]),
        "x_dim": int(volume.shape[2]),
        "raw_p01": float(np.percentile(volume, 1)),
        "raw_p50": float(np.percentile(volume, 50)),
        "raw_p99": float(np.percentile(volume, 99)),
    }

    cell_labels, cell_union = weak_cell_masks_3d(volume)
    outside = ~cell_union
    bg_values = volume[outside]
    metrics.update(
        {
            "cell_count_auto": int(cell_labels.max()),
            "background_mean": float(np.mean(bg_values)) if bg_values.size else float("nan"),
            "background_std": float(np.std(bg_values)) if bg_values.size else float("nan"),
            "cell_mask_source": "auto",
        }
    )

    cell_props = measure.regionprops(cell_labels, intensity_image=volume)
    cell_volumes = []
    cell_diffuse_means = []
    for prop in cell_props:
        cell_volumes.append(float(prop.area) * (VOXEL_SIZE_UM**2) * Z_STEP_UM)
        coords = tuple(prop.coords.T)
        cell_values = volume[coords]
        if filament_mask is not None and np.any(filament_mask):
            filament_vals = filament_mask[coords] > 0
            diffuse_vals = cell_values[~filament_vals]
        else:
            diffuse_vals = cell_values
        if diffuse_vals.size:
            cell_diffuse_means.append(float(np.mean(diffuse_vals)))

    if cell_volumes:
        metrics["cell_volume_um3_mean"] = float(np.mean(cell_volumes))
        metrics["cell_volume_um3_sum"] = float(np.sum(cell_volumes))
    if cell_diffuse_means:
        metrics["cell_diffuse_mean"] = float(np.mean(cell_diffuse_means))

    if filament_mask is None:
        metrics["filament_label_state"] = "unlabeled"
        return metrics

    filament_mask = filament_mask > 0
    filament_sum = int(np.sum(filament_mask))
    if filament_sum == 0:
        metrics["filament_label_state"] = "explicit_negative"
        return metrics

    metrics["filament_label_state"] = "positive"
    host_labels = filament_host_cell_labels(cell_labels, filament_mask)
    filament_values = volume[filament_mask]
    metrics.update(
        {
            "filament_voxel_count": filament_sum,
            "filament_volume_um3": float(filament_sum * (VOXEL_SIZE_UM**2) * Z_STEP_UM),
            "filament_intensity_mean": float(np.mean(filament_values)),
            "filament_intensity_median": float(np.median(filament_values)),
            "filament_z_occupancy": int(np.sum(np.any(filament_mask, axis=(1, 2)))),
            "filament_length_um": skeleton_length_3d(filament_mask),
            "filament_host_cell_count": int(len(host_labels)),
        }
    )
    metrics.update(local_width_um(filament_mask))

    if host_labels:
        host_mask = np.isin(cell_labels, host_labels)
        host_diffuse_mask = host_mask & ~filament_mask
        host_diffuse_values = volume[host_diffuse_mask]
        if host_diffuse_values.size:
            metrics["host_cell_diffuse_mean"] = float(np.mean(host_diffuse_values))
            metrics["filament_minus_host_diffuse"] = float(np.mean(filament_values) - np.mean(host_diffuse_values))
            total_budget = float(np.sum(volume[host_mask]))
            filament_budget = float(np.sum(filament_values))
            metrics["filament_budget_fraction"] = float(filament_budget / max(total_budget, 1e-6))
            metrics["host_cell_volume_um3"] = float(np.sum(host_mask) * (VOXEL_SIZE_UM**2) * Z_STEP_UM)

    return metrics


def _hist_plot(values: np.ndarray, title: str, path: Path, bins: int = 30) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=bins, color="#4c78a8", alpha=0.9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def measure_dataset_3d(
    data_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    fluorescence_channel: int = 1,
    max_timepoints: int | None = None,
    labeled_only: bool = False,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    index = build_dataset_index_3d(data_dir, mask_dir, fluorescence_channel=fluorescence_channel)
    if labeled_only:
        index = index[index["has_filament_mask"]].reset_index(drop=True)
    if max_timepoints is not None:
        index = index.head(max_timepoints).reset_index(drop=True)
    save_dataset_index_3d(index, output_dir / "dataset_index_3d.csv")

    rows = []
    for row in index.itertuples(index=False):
        volume = load_fluorescence_volume(Path(row.raw_path), int(row.time_index), channel_index=int(row.fluorescence_channel))
        filament_mask = np.load(row.mask_path).astype(np.float32) if isinstance(row.mask_path, str) and row.mask_path else None
        rows.append(measure_timepoint(volume, row.sample_id, filament_mask))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "timepoint_measurements_3d.csv", index=False)

    for column, title in {
        "cell_count_auto": "Auto Cell Count",
        "cell_volume_um3_mean": "Cell Volume (um^3)",
        "cell_diffuse_mean": "Cell Diffuse Mean",
        "filament_length_um": "Filament Length (um)",
        "filament_width_um_mean": "Filament Width (um)",
        "filament_budget_fraction": "Filament Budget Fraction",
    }.items():
        if column in df:
            _hist_plot(df[column].to_numpy(dtype=float), title, plots_dir / f"{column}.png")

    labeled_df = df[df["filament_label_state"].isin(["positive", "explicit_negative"])] if "filament_label_state" in df else df.iloc[0:0]
    summary = {
        "dataset_summary": {
            "crop_count": int(index["raw_stem"].nunique()) if not index.empty else 0,
            "timepoint_count": int(len(index)),
            "mask_count": int(index["has_filament_mask"].sum()) if not index.empty else 0,
            "positive_mask_count": int((df.get("filament_label_state", pd.Series(dtype=str)) == "positive").sum()) if not df.empty else 0,
            "explicit_negative_count": int((df.get("filament_label_state", pd.Series(dtype=str)) == "explicit_negative").sum()) if not df.empty else 0,
            "unlabeled_count": int((df.get("filament_label_state", pd.Series(dtype=str)) == "unlabeled").sum()) if not df.empty else 0,
            "fluorescence_channel": int(fluorescence_channel),
            "labeled_only": bool(labeled_only),
            "max_timepoints": int(max_timepoints) if max_timepoints is not None else None,
            "voxel_size_um_xy": VOXEL_SIZE_UM,
            "voxel_size_um_z": Z_STEP_UM,
        },
        "distributions": {},
        "filament_presence_comparison": {},
    }

    for col in [
        "cell_count_auto",
        "cell_volume_um3_mean",
        "cell_diffuse_mean",
        "background_mean",
        "background_std",
        "filament_volume_um3",
        "filament_length_um",
        "filament_width_um_mean",
        "filament_intensity_mean",
        "host_cell_diffuse_mean",
        "filament_minus_host_diffuse",
        "filament_budget_fraction",
        "filament_z_occupancy",
    ]:
        if col in df:
            vals = df[col].dropna().to_numpy(dtype=float)
            summary["distributions"][col] = {
                "values": vals.tolist(),
                "summary": {**robust_mean_std(vals), **percentile_summary(vals)},
            }

    if not labeled_df.empty and "cell_diffuse_mean" in labeled_df:
        positive = labeled_df[labeled_df["filament_label_state"] == "positive"]["cell_diffuse_mean"].dropna().to_numpy(dtype=float)
        negative = labeled_df[labeled_df["filament_label_state"] == "explicit_negative"]["cell_diffuse_mean"].dropna().to_numpy(dtype=float)
        summary["filament_presence_comparison"]["cell_diffuse_mean"] = {
            "positive_mean": float(np.mean(positive)) if positive.size else float("nan"),
            "negative_mean": float(np.mean(negative)) if negative.size else float("nan"),
            "delta_positive_minus_negative": float(np.mean(positive) - np.mean(negative)) if positive.size and negative.size else float("nan"),
        }

    save_json(to_builtin(summary), output_dir / "stats_summary_3d.json")
    return summary
