#!/usr/bin/env python3
"""
Visualize a brightfield frame with the tracked cell IDs overlayed.

Usage:
  python scripts/visualize_cell_masks.py \
    --tif tiffs3d/ch20_URA7_URA8_001_hyperstack_crop_01.tif \
    --mask results/masks/ch20_URA7_URA8_001_hyperstack_crop_01/ch20_URA7_URA8_001_hyperstack_crop_01_mask_t000.tif \
    --plane 2 \
    --output output/cell_mask_overlay_t000.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay tracked cell IDs on a brightfield frame.")
    parser.add_argument("--tif", required=True, help="5D TIFF (T,Z,C,H,W) for the video.")
    parser.add_argument("--mask", required=True, help="Mask TIFF (Z,H,W) for a specific timepoint.")
    parser.add_argument("--time", type=int, default=0, help="Time index (default: 0).")
    parser.add_argument("--plane", type=int, default=None, help="Z-plane index (default: middle).")
    parser.add_argument(
        "--output", default="output/cell_mask_overlay.png", help="PNG path for the overlay."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha for the mask overlay."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tiff_path = Path(args.tif)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(tiff_path)
    mask = tifffile.imread(mask_path)

    if img.ndim < 4:
        raise ValueError(f"Expected TIFF with >=4 dims (T,Z,C,H,W), got {img.shape}")
    bf = img[..., 0, :, :]

    if bf.ndim != 4:
        raise ValueError(f"Expected BF channel to be (T,Z,H,W), got {bf.shape}")

    if mask.ndim == 3:
        mask = mask[np.newaxis]
    if mask.ndim != 4:
        raise ValueError(f"Expected mask to be (T,Z,H,W), got {mask.shape}")

    T, Z, H, W = mask.shape
    t_idx = int(np.clip(args.time, 0, T - 1))
    if args.plane is None:
        z_idx = Z // 2
    else:
        z_idx = int(np.clip(args.plane, 0, Z - 1))

    image = bf[t_idx, z_idx]
    label_mask = mask[t_idx, z_idx]
    max_id = int(label_mask.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", interpolation="nearest")
    overlay = ax.imshow(
        label_mask,
        cmap="tab20",
        alpha=args.alpha,
        interpolation="nearest",
        vmin=0,
        vmax=max(max_id, 1),
    )
    cbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell ID")
    ax.set_title(f"Cell IDs (plane {z_idx})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Overlay saved to {output_path}")


if __name__ == "__main__":
    main()
