from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from .measure_stats_3d import VOXEL_SIZE_UM, Z_STEP_UM
from .utils import ensure_dir


@dataclass
class StatsSampler3D:
    summary: dict[str, Any]
    measurements: pd.DataFrame

    @classmethod
    def from_stats_dir(cls, stats_dir: Path) -> "StatsSampler3D":
        with (stats_dir / "stats_summary_3d.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        measurements = pd.read_csv(stats_dir / "timepoint_measurements_3d.csv")
        return cls(summary=summary, measurements=measurements)

    def sample_scalar(self, name: str, rng: np.random.Generator, default: float) -> float:
        payload = self.summary.get("distributions", {}).get(name, {})
        values = np.asarray(payload.get("values", []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(default)
        return float(rng.choice(values))

    def sample_int(self, name: str, rng: np.random.Generator, default: int) -> int:
        return int(round(self.sample_scalar(name, rng, default)))

    def sample_shape(self, rng: np.random.Generator) -> tuple[int, int, int]:
        cols = self.measurements[["z_dim", "y_dim", "x_dim"]].dropna()
        if cols.empty:
            return 5, 128, 128
        row = cols.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        return int(row["z_dim"]), int(row["y_dim"]), int(row["x_dim"])


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center_zyx: tuple[float, float, float],
    radii_zyx: tuple[float, float, float],
    rotation_xy: float,
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    cz, cy, cx = center_zyx
    rz, ry, rx = radii_zyx
    y = yy - cy
    x = xx - cx
    xr = x * math.cos(rotation_xy) - y * math.sin(rotation_xy)
    yr = x * math.sin(rotation_xy) + y * math.cos(rotation_xy)
    zr = zz - cz
    norm = (zr / max(rz, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2 + (xr / max(rx, 1e-6)) ** 2
    return norm <= 1.0


def _distance_to_polyline_3d(shape: tuple[int, int, int], points: np.ndarray) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    grid = np.stack([zz, yy, xx], axis=-1)
    min_dist2 = np.full(shape, np.inf, dtype=float)
    spacing = np.asarray([Z_STEP_UM, VOXEL_SIZE_UM, VOXEL_SIZE_UM], dtype=float)
    for p0, p1 in zip(points[:-1], points[1:]):
        p0s = p0 * spacing
        p1s = p1 * spacing
        seg = p1s - p0s
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 <= 1e-9:
            continue
        rel = grid * spacing - p0s
        t = np.clip((rel[..., 0] * seg[0] + rel[..., 1] * seg[1] + rel[..., 2] * seg[2]) / seg_len2, 0.0, 1.0)
        proj = p0s + t[..., None] * seg
        dist2 = np.sum((grid * spacing - proj) ** 2, axis=-1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.sqrt(min_dist2)


def _sample_filament_centerline(
    center_zyx: tuple[float, float, float],
    length_um: float,
    curvature_um: float,
    rng: np.random.Generator,
    num_points: int = 48,
) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-6)
    ortho1 = rng.normal(size=3)
    ortho1 -= np.dot(ortho1, axis) * axis
    ortho1 /= max(np.linalg.norm(ortho1), 1e-6)
    ortho2 = np.cross(axis, ortho1)

    t = np.linspace(-0.5, 0.5, num_points)
    phase = rng.uniform(0, 2 * math.pi)
    along = t * length_um
    bend1 = curvature_um * np.sin(np.linspace(0, math.pi, num_points) + phase)
    bend2 = 0.4 * curvature_um * np.sin(np.linspace(0, 2 * math.pi, num_points) - phase)
    pts_um = (
        np.asarray(center_zyx) * np.asarray([Z_STEP_UM, VOXEL_SIZE_UM, VOXEL_SIZE_UM])[None, :]
        + along[:, None] * axis[None, :]
        + bend1[:, None] * ortho1[None, :]
        + bend2[:, None] * ortho2[None, :]
    )
    return pts_um / np.asarray([Z_STEP_UM, VOXEL_SIZE_UM, VOXEL_SIZE_UM])[None, :]


def _generate_filament_mask_for_cell(
    shape: tuple[int, int, int],
    cell_mask: np.ndarray,
    length_um: float,
    width_um: float,
    z_occupancy: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(cell_mask)
    if len(coords) == 0:
        return np.zeros(shape, dtype=bool), np.empty((0, 3), dtype=float)
    center = np.mean(coords, axis=0)
    radius_vox = np.array(
        [
            max((coords[:, 0].max() - coords[:, 0].min()) / 2.0, 1.0),
            max((coords[:, 1].max() - coords[:, 1].min()) / 2.0, 1.0),
            max((coords[:, 2].max() - coords[:, 2].min()) / 2.0, 1.0),
        ]
    )
    center[0] = np.clip(center[0] + rng.uniform(-0.2, 0.2) * radius_vox[0], 0, shape[0] - 1)
    if z_occupancy < shape[0]:
        center[0] = np.clip(rng.uniform(z_occupancy / 2.0, shape[0] - 1 - z_occupancy / 2.0), 0, shape[0] - 1)

    for _ in range(24):
        curvature_um = rng.uniform(0.0, 0.2 * max(length_um, 1.0))
        points = _sample_filament_centerline(tuple(center), length_um=length_um, curvature_um=curvature_um, rng=rng)
        dist = _distance_to_polyline_3d(shape, points)
        mask = dist <= max(width_um / 2.0, 0.3)
        inside_frac = float(np.mean(cell_mask[mask])) if np.any(mask) else 0.0
        z_occ = int(np.sum(np.any(mask, axis=(1, 2))))
        if inside_frac >= 0.9 and z_occ <= max(z_occupancy, 1):
            return mask, points
    return np.zeros(shape, dtype=bool), np.empty((0, 3), dtype=float)


def _make_diffuse_texture(mask: np.ndarray, mean_value: float, rng: np.random.Generator) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.zeros(mask.shape, dtype=float)
    field = rng.normal(size=mask.shape)
    field = ndi.gaussian_filter(field, sigma=(0.5, 1.8, 1.8))
    field = (field - np.mean(field)) / max(np.std(field), 1e-6)
    zz, yy, xx = np.indices(mask.shape, dtype=float)
    coords = np.argwhere(mask)
    center = np.mean(coords, axis=0)
    radius = np.maximum(np.std(coords, axis=0), 1.0)
    radial = (
        ((zz - center[0]) / radius[0]) ** 2
        + ((yy - center[1]) / radius[1]) ** 2
        + ((xx - center[2]) / radius[2]) ** 2
    )
    profile = mean_value * (1.0 - 0.06 * radial) + 0.12 * mean_value * field
    out = np.zeros(mask.shape, dtype=float)
    out[mask] = np.clip(profile[mask], 0.0, None)
    return out


def _apply_imaging_model_3d(
    volume: np.ndarray,
    background_mean: float,
    background_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    illumination = np.linspace(0.985, 1.015, volume.shape[2], dtype=float)[None, None, :]
    signal = np.clip(volume * illumination, 0.0, None)
    blurred_signal = ndi.gaussian_filter(signal, sigma=(0.55, 0.9, 0.9))

    # Mixed microscopy noise model:
    # observed = camera_offset + Poisson(photon_term) + Gaussian_read_noise
    # The offset gives the stable floor near the measured background baseline.
    camera_offset = float(background_mean)

    # Keep a small photon-like background term so the background is not perfectly flat,
    # but avoid letting it dominate variance.
    background_photon_term = max(0.5, 0.08 * background_mean)

    # Photon gain controls how much Poisson variance appears for a given signal.
    photon_gain = 0.22
    photon_expectation = np.clip((blurred_signal + background_photon_term) * photon_gain, 0.0, None)
    poisson_counts = rng.poisson(photon_expectation).astype(float) / max(photon_gain, 1e-6)

    # Subtract the mean background photon contribution so the baseline remains near camera_offset.
    poisson_signal = poisson_counts - background_photon_term

    read_noise_sigma = max(background_std * 0.55, 1e-3)
    read_noise = rng.normal(0.0, read_noise_sigma, size=volume.shape)
    noisy = camera_offset + poisson_signal + read_noise
    return np.clip(noisy, 0.0, None).astype(np.float32)


def generate_one_volume_3d(
    sampler: StatsSampler3D,
    rng: np.random.Generator,
    filament_prob_per_cell: float = 0.3,
    cell_size_scale: float = 1.5,
    cell_spacing_scale: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    shape = sampler.sample_shape(rng)
    cell_count = int(np.clip(sampler.sample_int("cell_count_auto", rng, 2), 1, 4))
    background_mean = sampler.sample_scalar("background_mean", rng, 120.0)
    background_std = sampler.sample_scalar("background_std", rng, 6.0)
    measured_diffuse_mean_abs = sampler.sample_scalar("cell_diffuse_mean", rng, background_mean + 18.0)
    mean_cell_volume = sampler.sample_scalar("cell_volume_um3_mean", rng, 45.0)
    filament_length_um = sampler.sample_scalar("filament_length_um", rng, 4.5)
    filament_width_um = sampler.sample_scalar("filament_width_um_mean", rng, 0.7)
    measured_host_diffuse_mean_abs = sampler.sample_scalar("host_cell_diffuse_mean", rng, measured_diffuse_mean_abs)
    measured_filament_intensity_mean_abs = sampler.sample_scalar("filament_intensity_mean", rng, measured_host_diffuse_mean_abs + 35.0)
    filament_minus_host_diffuse = sampler.sample_scalar(
        "filament_minus_host_diffuse",
        rng,
        max(10.0, measured_filament_intensity_mean_abs - measured_host_diffuse_mean_abs),
    )
    filament_budget_fraction = sampler.sample_scalar("filament_budget_fraction", rng, 0.05)
    filament_z_occupancy = int(np.clip(round(sampler.sample_scalar("filament_z_occupancy", rng, 4)), 1, shape[0]))

    # The weak automatic cell masks under-estimate cell-body intensity. Bias toward the observed
    # visual ranges so background/cell/filament contrast is realistic in the rendered volumes.
    diffuse_mean_abs = max(
        measured_diffuse_mean_abs,
        background_mean + rng.uniform(28.0, 42.0),
    )
    host_diffuse_mean_abs = max(
        measured_host_diffuse_mean_abs,
        diffuse_mean_abs - rng.uniform(2.0, 10.0),
    )
    filament_intensity_mean_abs = max(
        measured_filament_intensity_mean_abs,
        host_diffuse_mean_abs + rng.uniform(45.0, 80.0),
        rng.uniform(220.0, 250.0),
    )

    diffuse_signal_mean = max(0.0, diffuse_mean_abs - background_mean)
    host_diffuse_signal_mean = max(0.0, host_diffuse_mean_abs - background_mean)
    filament_signal_mean = max(0.0, filament_intensity_mean_abs - background_mean)

    image = np.zeros(shape, dtype=float)
    cell_labels = np.zeros(shape, dtype=np.uint16)
    filament_mask = np.zeros(shape, dtype=bool)
    cell_metadata: list[dict[str, Any]] = []

    voxel_volume = VOXEL_SIZE_UM * VOXEL_SIZE_UM * Z_STEP_UM
    target_cell_voxels = max((mean_cell_volume / voxel_volume) * cell_size_scale**3, 50.0)
    base_radius = (3.0 * target_cell_voxels / (4.0 * math.pi)) ** (1.0 / 3.0)

    centers: list[np.ndarray] = []
    scene_center = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0], dtype=float)
    for cell_idx in range(1, cell_count + 1):
        for _ in range(50):
            cluster_pull = rng.uniform(0.35, 0.8)
            jitter = np.array(
                [
                    rng.uniform(-0.8, 0.8),
                    rng.uniform(-shape[1] * 0.22, shape[1] * 0.22),
                    rng.uniform(-shape[2] * 0.22, shape[2] * 0.22),
                ]
            )
            proposed = scene_center + cluster_pull * jitter
            center = np.array(
                [
                    # Keep cells centered in z so they persist across all 5 slices.
                    np.clip(rng.normal(loc=(shape[0] - 1) / 2.0, scale=0.18), 1.8, shape[0] - 1.8),
                    np.clip(proposed[1], shape[1] * 0.12, shape[1] * 0.88),
                    np.clip(proposed[2], shape[2] * 0.12, shape[2] * 0.88),
                ]
            )
            min_spacing = max(6.0, 2.2 * base_radius * cell_spacing_scale)
            if not centers or min(np.linalg.norm(center[1:] - c[1:]) for c in centers) > min_spacing:
                centers.append(center)
                break
        else:
            centers.append(center)

        scale = rng.uniform(1.05, 1.5)
        rx = base_radius * scale * rng.uniform(0.95, 1.25)
        ry = base_radius * scale * rng.uniform(0.95, 1.25)
        # Force cell support through all z planes while still tapering near the ends.
        rz = max(2.35, min(3.2, base_radius * scale * rng.uniform(0.95, 1.25)))
        cell_mask = _ellipsoid_mask(shape, tuple(centers[-1]), (rz, ry, rx), rotation_xy=float(rng.uniform(0, math.pi)))
        cell_labels[cell_mask] = cell_idx

        has_filament = bool(rng.random() < filament_prob_per_cell)
        diffuse_target = diffuse_signal_mean
        cell_filament_mask = np.zeros(shape, dtype=bool)
        centerline = np.empty((0, 3), dtype=float)
        if has_filament:
            cell_filament_mask, centerline = _generate_filament_mask_for_cell(
                shape,
                cell_mask=cell_mask,
                length_um=max(1.2, filament_length_um * rng.uniform(0.7, 1.3)),
                width_um=max(0.3, filament_width_um * rng.uniform(0.85, 1.2)),
                z_occupancy=filament_z_occupancy,
                rng=rng,
            )
            if np.any(cell_filament_mask):
                filament_mask |= cell_filament_mask
                # Protein redistribution: filament-positive cells lose some diffuse pool.
                diffuse_target = host_diffuse_signal_mean * max(0.75, 1.0 - filament_budget_fraction * rng.uniform(0.2, 0.6))

        diffuse = _make_diffuse_texture(cell_mask, diffuse_target, rng)
        image += diffuse
        if np.any(cell_filament_mask):
            filament_target = max(diffuse_target, filament_signal_mean)
            filament_boost = max(
                0.0,
                max(
                    min(filament_minus_host_diffuse, filament_target - diffuse_target),
                    filament_target - diffuse_target,
                ),
            )
            image[cell_filament_mask] += filament_boost

        cell_metadata.append(
            {
                "cell_index": cell_idx,
                "center_zyx": centers[-1].tolist(),
                "radii_zyx_vox": [float(rz), float(ry), float(rx)],
                "has_filament": bool(np.any(cell_filament_mask)),
                "filament_voxels": int(np.sum(cell_filament_mask)),
                "centerline_zyx": centerline.tolist(),
            }
        )

    image = _apply_imaging_model_3d(image, background_mean=background_mean, background_std=background_std, rng=rng)
    metadata = {
        "shape_zyx": list(shape),
        "cell_count": cell_count,
        "background_mean": float(background_mean),
        "background_std": float(background_std),
        "diffuse_mean_abs": float(diffuse_mean_abs),
        "diffuse_signal_mean": float(diffuse_signal_mean),
        "host_diffuse_mean_abs": float(host_diffuse_mean_abs),
        "filament_intensity_mean_abs": float(filament_intensity_mean_abs),
        "filament_length_um_reference": float(filament_length_um),
        "filament_width_um_reference": float(filament_width_um),
        "filament_prob_per_cell": float(filament_prob_per_cell),
        "cell_size_scale": float(cell_size_scale),
        "cell_spacing_scale": float(cell_spacing_scale),
        "cells": cell_metadata,
    }
    return image, cell_labels, filament_mask.astype(np.uint8), metadata


def generate_dataset_3d(
    stats_dir: Path,
    output_dir: Path,
    count: int,
    seed: int = 0,
    filament_prob_per_cell: float = 0.3,
    cell_size_scale: float = 1.2,
    cell_spacing_scale: float = 0.7,
) -> pd.DataFrame:
    sampler = StatsSampler3D.from_stats_dir(stats_dir)
    rng = np.random.default_rng(seed)
    ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    cell_dir = ensure_dir(output_dir / "cell_labels")
    filament_dir = ensure_dir(output_dir / "filament_masks")
    meta_dir = ensure_dir(output_dir / "metadata")

    rows = []
    for idx in range(count):
        image, cell_labels, filament_mask, metadata = generate_one_volume_3d(
            sampler=sampler,
            rng=rng,
            filament_prob_per_cell=filament_prob_per_cell,
            cell_size_scale=cell_size_scale,
            cell_spacing_scale=cell_spacing_scale,
        )
        sample_id = f"synth3d_{idx:05d}"
        tifffile.imwrite(image_dir / f"{sample_id}.tif", image)
        tifffile.imwrite(cell_dir / f"{sample_id}_cell_labels.tif", cell_labels.astype(np.uint16))
        tifffile.imwrite(filament_dir / f"{sample_id}_filament_mask.tif", filament_mask.astype(np.uint8))
        with (meta_dir / f"{sample_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_dir / f"{sample_id}.tif"),
                "cell_labels_path": str(cell_dir / f"{sample_id}_cell_labels.tif"),
                "filament_mask_path": str(filament_dir / f"{sample_id}_filament_mask.tif"),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest
