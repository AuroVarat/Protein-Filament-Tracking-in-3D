from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def make_diffuse_cell_signal(
    cell_mask: np.ndarray,
    diffuse_mean: float,
    radial_slope: float,
    texture_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    image = np.zeros(cell_mask.shape, dtype=float)
    if not np.any(cell_mask):
        return image

    rr, cc = np.indices(cell_mask.shape)
    coords = np.argwhere(cell_mask)
    centroid = np.mean(coords, axis=0)
    radius = max(np.sqrt(np.sum(cell_mask) / np.pi), 1.0)
    norm_r = np.sqrt((rr - centroid[0]) ** 2 + (cc - centroid[1]) ** 2) / radius

    texture = rng.normal(size=cell_mask.shape)
    texture = ndi.gaussian_filter(texture, sigma=6.0)
    texture = (texture - texture.mean()) / max(texture.std(), 1e-6)

    profile = diffuse_mean + radial_slope * norm_r + texture_std * texture
    image[cell_mask] = profile[cell_mask]
    return np.clip(image, 0.0, None)


def add_filament_signal(
    base_image: np.ndarray,
    filament_mask: np.ndarray,
    centerline: np.ndarray,
    filament_delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = base_image.copy()
    if not np.any(filament_mask):
        return out

    if len(centerline) >= 2:
        cumulative = np.concatenate([[0.0], np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0) ** 2, axis=1)))])
        cumulative /= max(cumulative[-1], 1e-6)
        mod = 1.0 + 0.1 * np.sin(2 * np.pi * cumulative + rng.uniform(0, 2 * np.pi))
    else:
        mod = np.asarray([1.0])
    filament_boost = filament_delta * np.mean(mod)
    out[filament_mask] += filament_boost
    return out


def apply_imaging_model(
    image: np.ndarray,
    background_mean: float,
    read_noise_std: float,
    rng: np.random.Generator,
    blur_sigma: float = 1.1,
    poisson_scale: float = 1.0,
    illumination_strength: float = 0.05,
) -> np.ndarray:
    rr, cc = np.indices(image.shape)
    grad = 1.0 + illumination_strength * ((cc / max(image.shape[1] - 1, 1)) - 0.5)
    signal = np.clip((image + background_mean) * grad, 0.0, None)
    blurred = ndi.gaussian_filter(signal, sigma=max(blur_sigma, 0.5))
    poisson_ready = np.clip(blurred * poisson_scale, 0.0, None)
    shot = rng.poisson(poisson_ready) / max(poisson_scale, 1e-6)
    noisy = shot + rng.normal(0.0, max(read_noise_std, 1e-6), size=image.shape)
    return np.clip(noisy, 0.0, None)
