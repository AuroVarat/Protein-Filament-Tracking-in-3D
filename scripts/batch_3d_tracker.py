#!/usr/bin/env python3
"""
Batch 3D Filament Tracker (Cell-Based)

Logic:
- Segment cells in Channel 0 (Brightfield) for each frame.
- Track the same cell ID across planes/time and reject the two pillar-like BF outlines
  on frame 0 using low channel-1 signal and matched outline area.
- Perform filament inference in Channel 1 (Filament) and associate every voxel inside a cell’s mask (across planes) to that cell’s filament.
- One filament per cell, regardless of gaps or plane jumps.
- Output a CSV of pixel coordinates plus a companion mask TIFF per frame with the IDs (pillars zeroed out).
"""

import sys
import os
import glob
import tifffile
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm
import argparse
from collections import defaultdict
from pathlib import Path

from unet3d import TinyUNet3D
from utils import best_device

# Tracking parameters
MAX_CELL_DISTANCE = 40.0


class CellTracker:
    def __init__(self, max_distance: float = MAX_CELL_DISTANCE):
        self.max_distance = max_distance
        self.next_id = 1
        self.centroids: dict[int, tuple[float, float]] = {}

    def update(self, new_centroids: dict[int, tuple[float, float]]) -> dict[int, int]:
        assignment = {}
        used = set()
        for label, center in new_centroids.items():
            best_id = None
            best_dist = float("inf")
            for cid, prev_center in self.centroids.items():
                if cid in used:
                    continue
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = cid

            if best_id is not None:
                assignment[label] = best_id
                self.centroids[best_id] = center
                used.add(best_id)
            else:
                assignment[label] = self.next_id
                self.centroids[self.next_id] = center
                used.add(self.next_id)
                self.next_id += 1

        return assignment


def compute_label_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    centroids = {}
    labels = np.unique(mask)
    for label in labels:
        if label == 0:
            continue
        centroid = ndimage.center_of_mass(mask == label)
        if np.isnan(centroid[0]):
            continue
        centroids[label] = centroid
    return centroids


def segment_cells_2d(vol_3d, cellpose_model):
    """
    Segment cells on the middle Z-slice of a 3D volume.
    Returns 2D mask (H, W).
    """
    z_idx = vol_3d.shape[0] // 2
    ref_2d = vol_3d[z_idx]
    masks, _, _ = cellpose_model.eval(ref_2d, diameter=25)
    return masks


def identify_pillar_ids(
    track_stats: dict[int, dict[str, list[float]]],
) -> list[int]:
    """
    Choose the two pillar-like outlines using temporal statistics:
    1. persistently weak fluorescence
    2. similar area
    3. low centroid motion / area variation across time
    """
    stats: list[dict] = []
    for global_id, values in track_stats.items():
        if not values["area"] or not values["mean_signal"]:
            continue
        stats.append(
            {
                "global_id": int(global_id),
                "mean_area": float(np.mean(values["area"])),
                "area_std": float(np.std(values["area"])),
                "mean_signal": float(np.mean(values["mean_signal"])),
                "q90_signal": float(np.mean(values["q90_signal"])),
                "centroid_motion": float(
                    np.std(values["centroid_y"]) + np.std(values["centroid_x"])
                ),
            }
        )

    if len(stats) < 2:
        return []

    best_pair: tuple[int, int] | None = None
    best_score: tuple[float, float] | None = None
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            a = stats[i]
            b = stats[j]
            pair_signal = a["mean_signal"] + b["mean_signal"]
            area_similarity = abs(a["mean_area"] - b["mean_area"])
            stability_penalty = (
                a["area_std"] + b["area_std"] + a["centroid_motion"] + b["centroid_motion"]
            )
            score = (pair_signal, area_similarity + stability_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (a["global_id"], b["global_id"])

    return list(best_pair) if best_pair is not None else []

def collect_filament_voxels(
    pred_mask: np.ndarray,
    cell_mask_2d: np.ndarray,
    raw_3d: np.ndarray,
    prob_3d: np.ndarray,
) -> list[dict]:
    """
    Emit one row per predicted filament voxel, keyed by the tracked parent cell ID.
    """
    z_idx, y_idx, x_idx = np.where(pred_mask > 0)
    if len(z_idx) == 0:
        return []

    voxel_cell_ids = cell_mask_2d[y_idx, x_idx]
    keep = voxel_cell_ids > 0
    if not np.any(keep):
        return []

    z_idx = z_idx[keep]
    y_idx = y_idx[keep]
    x_idx = x_idx[keep]
    voxel_cell_ids = voxel_cell_ids[keep]

    rows = []
    for z, y, x, cid in zip(z_idx, y_idx, x_idx, voxel_cell_ids):
        rows.append(
            {
                "filament_id": int(cid),
                "z": int(z),
                "y": int(y),
                "x": int(x),
                "raw_intensity": float(raw_3d[z, y, x]),
                "probability": float(prob_3d[z, y, x]),
            }
        )
    return rows

def process_video(filepath, model, cellpose_model, device, tracking_dir, mask_dir, cell_mask_dir):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_csv = tracking_dir / f"{base_name}_tracking.csv"
    mask_path = mask_dir / f"{base_name}_mask.tif"
    cell_mask_path = cell_mask_dir / f"{base_name}_mask.tif"

    print(f"\nProcessing {base_name}...")
    try:
        img = tifffile.imread(filepath).astype(np.float32)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if len(img.shape) != 5:
        print(f"Skipping {base_name}: Expected 5D TIFF (T, Z, C, H, W), got {img.shape}")
        return

    T, Z, C, H, W = img.shape
    vol_bf = img[:, :, 0, :, :]
    vol_fil = img[:, :, 1, :, :]
    norm_fil = np.zeros_like(vol_fil)
    for t in range(T):
        mn, mx = vol_fil[t].min(), vol_fil[t].max()
        if mx > mn:
            norm_fil[t] = (vol_fil[t] - mn) / (mx - mn)

    all_voxels = []
    tracker = CellTracker()
    frame_masks: list[np.ndarray] = []
    frame_filament_masks: list[np.ndarray] = []
    pillar_ids: list[int] = []
    track_stats: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "area": [],
            "mean_signal": [],
            "q90_signal": [],
            "centroid_y": [],
            "centroid_x": [],
        }
    )

    print(f"Inference & Segmentation ({T} frames)...")
    with torch.no_grad():
        for t in range(T):
            t_prev, t_curr, t_next = max(0, t - 1), t, min(T - 1, t + 1)
            window = np.stack([norm_fil[t_prev], norm_fil[t_curr], norm_fil[t_next]], axis=0)
            inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
            logits = model(inp).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits[1]))
            pred_mask = (probs > 0.5).astype(np.uint8)

            cell_mask = segment_cells_2d(vol_bf[t], cellpose_model)
            label_centroids = compute_label_centroids(cell_mask)
            label_to_global = tracker.update(label_centroids)

            global_mask = np.zeros_like(cell_mask, dtype=np.uint16)
            for label, gid in label_to_global.items():
                global_mask[cell_mask == label] = gid
                region = cell_mask == label
                values = vol_fil[t][:, region].reshape(-1)
                if values.size > 0:
                    centroid = ndimage.center_of_mass(region)
                    track_stats[gid]["area"].append(float(region.sum()))
                    track_stats[gid]["mean_signal"].append(float(values.mean()))
                    track_stats[gid]["q90_signal"].append(float(np.percentile(values, 90)))
                    track_stats[gid]["centroid_y"].append(float(centroid[0]))
                    track_stats[gid]["centroid_x"].append(float(centroid[1]))

            frame_mask = np.repeat(global_mask[np.newaxis], Z, axis=0).astype(np.uint16)
            filament_frame_mask = frame_mask.copy()
            filament_frame_mask[pred_mask == 0] = 0
            frame_masks.append(frame_mask)
            frame_filament_masks.append(filament_frame_mask)

            frame_voxels = collect_filament_voxels(pred_mask, global_mask, vol_fil[t], probs)
            for voxel in frame_voxels:
                voxel["frame"] = t
                all_voxels.append(voxel)

    if not all_voxels:
        print(f"No cell-associated filaments detected in {base_name}.")
        return

    pillar_ids = identify_pillar_ids(track_stats)
    if pillar_ids:
        print(f"Rejected pillar IDs from temporal BF/GFP logic: {pillar_ids}")

    filtered_voxels = [v for v in all_voxels if v["filament_id"] not in pillar_ids]

    cell_stack = np.stack(frame_masks, axis=0).astype(np.uint16)
    filament_stack = np.stack(frame_filament_masks, axis=0).astype(np.uint16)
    if pillar_ids:
        for pillar_id in pillar_ids:
            filament_stack[filament_stack == pillar_id] = 0
            cell_stack[cell_stack == pillar_id] = 0
    tifffile.imwrite(mask_path, filament_stack, imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(cell_mask_path, cell_stack, imagej=True, metadata={"axes": "TZYX"})
    print(f"Filament mask saved to {mask_path}")
    print(f"Cell mask saved to {cell_mask_path}")

    if not filtered_voxels:
        print(f"All detections filtered out (pillar removal removed all tracked IDs) for {base_name}.")
        return

    df = pd.DataFrame(filtered_voxels)
    cols = ["frame", "filament_id", "z", "y", "x", "raw_intensity", "probability"]
    df = df[cols]
    df.sort_values(by=["frame", "filament_id", "z", "y", "x"], inplace=True)
    df.to_csv(out_csv, index=False)
    print(
        f"Saved {len(df)} raw filament voxels "
        f"(from {df['filament_id'].nunique()} tracked filaments) to {out_csv}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="tiffs3d", help="Directory containing .tif files")
    parser.add_argument("--output_dir", default="results", help="Directory to save CSVs")
    parser.add_argument("--model_path", default="models/filament_unet3d_temporal_auto.pt", help="Path to model")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many files")
    args = parser.parse_args()
    
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tracking_dir = output_root / "tracking_csvs"
    mask_dir = output_root / "masks"
    cell_mask_dir = output_root / "cell_masks"
    for d in [tracking_dir, mask_dir, cell_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    device = best_device()
    print(f"Using device: {device}")
    
    # Load Filament Model
    model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Initialize Cellpose
    print("Initializing Cellpose model...")
    from cellpose import models
    cellpose_model = models.CellposeModel(model_type='cyto', gpu=torch.cuda.is_available() or torch.backends.mps.is_available())
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    if not files:
        print(f"No .tif files found in {args.input_dir}")
        sys.exit(0)
    if args.limit is not None:
        files = files[: args.limit]
        
    print(f"Found {len(files)} files to process.")
    for f in tqdm(files):
        process_video(
            Path(f),
            model,
            cellpose_model,
            device,
            tracking_dir,
            mask_dir,
            cell_mask_dir,
        )

if __name__ == "__main__":
    main()
