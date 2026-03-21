from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cell_geometry import generate_cell_mask
from .filament_geometry import generate_filament_mask
from .image_formation import add_filament_signal, apply_imaging_model, make_diffuse_cell_signal
from .io_utils import write_tiff
from .sample_distributions import StatsSampler
from .utils import ensure_dir, save_json


def _sample_latents(sampler: StatsSampler, rng: np.random.Generator) -> dict[str, Any]:
    image_shape = sampler.sample_image_shape(rng)
    radius = max(6.0, sampler.sample_or_default("cell_equivalent_radius", rng, min(image_shape) * 0.22))
    eccentricity = float(np.clip(sampler.sample_or_default("cell_eccentricity", rng, 0.25), 0.0, 0.95))
    boundary_smoothness = float(np.clip(sampler.sample_or_default("cell_boundary_smoothness", rng, 0.94), 0.75, 1.0))
    cy = float(np.clip(sampler.sample_or_default("cell_centroid_y_norm", rng, 0.5), 0.2, 0.8) * (image_shape[0] - 1))
    cx = float(np.clip(sampler.sample_or_default("cell_centroid_x_norm", rng, 0.5), 0.2, 0.8) * (image_shape[1] - 1))
    length = max(4.0, sampler.conditional_resample("filament_length", "cell_equivalent_radius", radius, rng) if sampler.has("filament_length") and sampler.has("cell_equivalent_radius") else radius * rng.uniform(1.2, 2.4))
    width = max(1.0, sampler.conditional_resample("filament_width_mean", "filament_length", length, rng) if sampler.has("filament_width_mean") and sampler.has("filament_length") else rng.uniform(2.0, 4.5))
    diffuse_mean = max(0.0, sampler.sample_or_default("diffuse_mean", rng, sampler.summary["stack_level"].get("stack_p50", {}).get("median", 120.0) + 6.0))
    radial_slope = sampler.sample_or_default("radial_slope", rng, 0.0)
    texture_std = abs(sampler.sample_or_default("inside_texture_std", rng, 3.0))
    filament_delta = max(0.0, sampler.conditional_resample("filament_minus_diffuse", "filament_width_mean", width, rng) if sampler.has("filament_minus_diffuse") and sampler.has("filament_width_mean") else diffuse_mean * rng.uniform(0.12, 0.35))
    bg_mean = max(0.0, sampler.sample_or_default("bg_mean", rng, sampler.summary["stack_level"].get("stack_p01", {}).get("median", 100.0)))
    bg_std = max(1e-3, sampler.sample_or_default("bg_std", rng, 4.0))
    orientation = sampler.sample_or_default("filament_orientation_rad", rng, float(rng.uniform(0, np.pi)))
    radial_position_norm = float(np.clip(sampler.sample_or_default("filament_position_radius_norm", rng, rng.uniform(0.0, 0.55)), 0.0, 0.95))
    curvature = sampler.sample_or_default("filament_curvature_mean", rng, 0.04)

    return {
        "image_shape": image_shape,
        "cell_center": (cy, cx),
        "cell_radius": radius,
        "cell_eccentricity": eccentricity,
        "cell_boundary_smoothness": boundary_smoothness,
        "filament_length": length,
        "filament_width": width,
        "filament_orientation": orientation,
        "filament_position_radius_norm": radial_position_norm,
        "filament_curvature_strength": max(0.5, curvature * length),
        "diffuse_mean": diffuse_mean,
        "diffuse_radial_slope": radial_slope,
        "diffuse_texture_std": texture_std,
        "filament_delta": filament_delta,
        "background_mean": bg_mean,
        "read_noise_std": bg_std,
    }


def generate_one_sample(sampler: StatsSampler, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    latents = _sample_latents(sampler, rng)
    cell_mask = generate_cell_mask(
        image_shape=latents["image_shape"],
        center_yx=latents["cell_center"],
        equivalent_radius=latents["cell_radius"],
        eccentricity=latents["cell_eccentricity"],
        boundary_smoothness=latents["cell_boundary_smoothness"],
        rng=rng,
    )
    filament_mask, centerline = generate_filament_mask(
        image_shape=latents["image_shape"],
        cell_mask=cell_mask,
        cell_center=latents["cell_center"],
        length=latents["filament_length"],
        width=latents["filament_width"],
        orientation=latents["filament_orientation"],
        radial_position_norm=latents["filament_position_radius_norm"],
        curvature_strength=latents["filament_curvature_strength"],
        rng=rng,
    )
    diffuse = make_diffuse_cell_signal(
        cell_mask=cell_mask,
        diffuse_mean=latents["diffuse_mean"],
        radial_slope=latents["diffuse_radial_slope"],
        texture_std=latents["diffuse_texture_std"],
        rng=rng,
    )
    image = add_filament_signal(diffuse, filament_mask, centerline, latents["filament_delta"], rng)
    image = apply_imaging_model(
        image=image,
        background_mean=latents["background_mean"],
        read_noise_std=latents["read_noise_std"],
        rng=rng,
    )
    metadata = {**latents, "centerline": centerline.tolist()}
    return image.astype(np.float32), cell_mask.astype(np.uint8), filament_mask.astype(np.uint8), metadata


def generate_dataset(stats_dir: Path, output_dir: Path, count: int, seed: int = 0, split: bool = False) -> pd.DataFrame:
    sampler = StatsSampler(stats_dir)
    rng = np.random.default_rng(seed)
    ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    cell_dir = ensure_dir(output_dir / "cell_masks")
    filament_dir = ensure_dir(output_dir / "filament_masks")
    meta_dir = ensure_dir(output_dir / "metadata")

    rows = []
    for idx in range(count):
        image, cell_mask, filament_mask, metadata = generate_one_sample(sampler, rng)
        sample_id = f"synth_{idx:05d}"
        write_tiff(image_dir / f"{sample_id}.tif", image)
        write_tiff(cell_dir / f"{sample_id}_cellmask.tif", cell_mask)
        write_tiff(filament_dir / f"{sample_id}_filamentmask.tif", filament_mask)
        save_json(metadata, meta_dir / f"{sample_id}.json")
        split_name = "train"
        if split:
            split_name = "val" if rng.random() < 0.2 else "train"
        rows.append({"sample_id": sample_id, "image_path": str(image_dir / f"{sample_id}.tif"), "cell_mask_path": str(cell_dir / f"{sample_id}_cellmask.tif"), "filament_mask_path": str(filament_dir / f"{sample_id}_filamentmask.tif"), "split": split_name})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    strategy_table = sampler.strategy_table()
    strategy_table.to_csv(output_dir / "sampling_strategies.csv", index=False)
    return manifest
