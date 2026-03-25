#!/usr/bin/env python3
"""
Filament 3D Tracker — Quantify and track 3D filaments in 5D TIFFs

Features:
- Brightfield cell segmentation (Channel 0)
- 3D Filament tracking (Channel 1) via Temporal U-Net
- Z-localization to reject out-of-focus multi-plane stretching
- Cross-frame tracking of individual filaments per cell
- Statistical export (CSV + Summary plot)

Usage:
    python scripts/filament_3d_tracker.py <tiffs3d/volume.tif> [--auto]
"""

import sys
import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
from tqdm import tqdm
import argparse

from unet3d import TinyUNet3D
from utils import best_device

# Physical parameters
PIXEL_SIZE_UM = 0.183
TIME_INTERVAL_MIN = 15.0
MIN_AREA_PX = 5
MAX_TRACK_DIST_PX = 15.0

def otsu(img):
    bins = 256
    hist, bin_centers = np.histogram(img, bins)
    hist = hist.astype(float)
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * bin_centers[1:]) / (w1 + 1e-6)
    m2 = (np.cumsum((hist * bin_centers[1:])[::-1]) / (w2[::-1] + 1e-6))[::-1]
    variance12 = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    if len(variance12) == 0: return img.mean()
    return bin_centers[1:][np.argmax(variance12)]

def segment_cells(bf_vol, seed_mask=None, pillar_mask=None, cellpose_model=None):
    """
    Given a 3D Brightfield volume, find cell labels using Cellpose on the middle Z-slice.
    Returns all objects found by Cellpose without strict filtering.
    """
    if cellpose_model is None:
        from cellpose import models
        cellpose_model = models.CellposeModel(model_type='cyto', gpu=True)
        
    z_idx = bf_vol.shape[0] // 2
    ref_2d = bf_vol[z_idx]
    
    # Run Cellpose
    masks, _, _ = cellpose_model.eval(ref_2d, diameter=25, channels=[0,0])

    # Remove pillars from the cellpose mask
    if pillar_mask is not None:
        masks[pillar_mask > 0] = 0
        
    # Only keep cells that contain at least one filament seed
    if seed_mask is not None:
        valid_cells = np.zeros_like(masks, dtype=np.int32)
        unique_cells = np.unique(masks)
        for i in unique_cells:
            if i == 0: continue
            cell_region = (masks == i)
            if np.any(cell_region & (seed_mask > 0)):
                valid_cells[cell_region] = i
        return valid_cells
        
    return masks

def z_localize(mask_3d, raw_3d):
    """
    If a filament appears in multiple Z-planes at the same (Y,X),
    keep only the Z-plane where the raw intensity is highest.
    """
    Z, H, W = mask_3d.shape
    localized = np.zeros_like(mask_3d)
    
    # Z-projection to find active (Y,X) pixels
    active_xy = np.sum(mask_3d, axis=0) > 0
    y_idx, x_idx = np.where(active_xy)
    
    for y, x in zip(y_idx, x_idx):
        z_candidates = np.where(mask_3d[:, y, x] > 0)[0]
        if len(z_candidates) > 0:
            intensities = raw_3d[z_candidates, y, x]
            best_z = z_candidates[np.argmax(intensities)]
            localized[best_z, y, x] = 1.0
            
    return localized

def measure_filaments(localized_mask, raw_3d, cell_mask):
    """
    Extract properties for each distinct 3D filament.
    """
    labeled, n_filaments = ndimage.label(localized_mask)
    blobs = []
    
    for i in range(1, n_filaments + 1):
        z_idx, y_idx, x_idx = np.where(labeled == i)
        size_px = len(z_idx)
        if size_px < MIN_AREA_PX:
            continue
            
        z_mean = np.mean(z_idx)
        y_mean = np.mean(y_idx)
        x_mean = np.mean(x_idx)
        
        # Mean intensity
        intensities = raw_3d[z_idx, y_idx, x_idx]
        mean_int = np.mean(intensities)
        
        # Find which cell this filament belongs to
        # Sample the 2D cell mask at the filament's XY coordinates
        cell_ids = cell_mask[y_idx, x_idx]
        cell_ids = cell_ids[cell_ids > 0]
        if len(cell_ids) > 0:
            # Assign to the most frequent cell ID
            cell_id = np.bincount(cell_ids).argmax()
        else:
            cell_id = 0 # Background / outside cell
            
        # Optional: compute length via PCA in 3D
        coords = np.vstack([z_idx, y_idx, x_idx])
        if coords.shape[1] > 2:
            cov = np.cov(coords)
            evals, _ = np.linalg.eigh(cov)
            length_px = 4 * np.sqrt(max(np.max(evals), 0))
        else:
            length_px = size_px
            
        length_um = length_px * PIXEL_SIZE_UM
        
        blobs.append({
            'cell_id': cell_id,
            'size_px': size_px,
            'length_um': round(length_um, 3),
            'mean_intensity': round(mean_int, 4),
            'z': z_mean,
            'y': y_mean,
            'x': x_mean,
            'indices': (z_idx, y_idx, x_idx),
            'filament_id': None # To be assigned during tracking
        })
        
    return blobs

def track_filaments(prev_blobs, curr_blobs, next_fil_id):
    """
    Match filaments between frame T-1 and T.
    """
    for cb in curr_blobs:
        best_pb = None
        best_dist = float('inf')
        
        for pb in prev_blobs:
            # We enforce tracking within the same cell if cell_id > 0
            # If the user's segmentation is wobbly, cell_ids might change, 
            # but we prioritize spatial proximity anyway.
            if pb['cell_id'] > 0 and cb['cell_id'] > 0 and pb['cell_id'] != cb['cell_id']:
                continue
                
            dist = np.sqrt((cb['y'] - pb['y'])**2 + (cb['x'] - pb['x'])**2 + (cb['z'] - pb['z'])**2)
            if dist < best_dist and dist < MAX_TRACK_DIST_PX:
                best_dist = dist
                best_pb = pb
                
        if best_pb is not None:
            cb['filament_id'] = best_pb['filament_id']
            # Remove from consideration to prevent two filaments claiming the same parent
            prev_blobs.remove(best_pb) 
        else:
            cb['filament_id'] = next_fil_id
            next_fil_id += 1
            
    return curr_blobs, next_fil_id

def make_summary_plot(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3D Filament Tracking Summary", fontsize=16)
    
    # Size over time
    for fid in df['filament_id'].unique():
        sub = df[df['filament_id'] == fid]
        axes[0, 0].plot(sub['time_min'], sub['size_px'], marker='o', alpha=0.7)
    axes[0, 0].set_title("Filament Size (Voxels) over Time")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("Size (Pixels)")
    
    # Length over time
    for fid in df['filament_id'].unique():
        sub = df[df['filament_id'] == fid]
        axes[0, 1].plot(sub['time_min'], sub['length_um'], marker='o', alpha=0.7)
    axes[0, 1].set_title("Filament Estimated Length (µm) over Time")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Length (µm)")
    
    # Average size per frame (Aggregate)
    df_mean = df.groupby('time_min')['size_px'].mean().reset_index()
    axes[1, 0].plot(df_mean['time_min'], df_mean['size_px'], color='red', marker='s', linewidth=2)
    axes[1, 0].set_title("Average Filament Size per Frame")
    axes[1, 0].set_xlabel("Time (min)")
    axes[1, 0].set_ylabel("Mean Size (Pixels)")
    axes[1, 0].grid(True)
    
    # Filament count per frame
    df_count = df.groupby('time_min')['filament_id'].count().reset_index()
    axes[1, 1].bar(df_count['time_min'], df_count['filament_id'], width=TIME_INTERVAL_MIN*0.8, color='teal', alpha=0.6)
    axes[1, 1].set_title("Number of Filaments Detected per Frame")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved summary plot to {out_path}")
    plt.close()

def save_segmentation_check(vol_bf, cell_masks, pillar_mask, out_path):
    """
    Saves a visual check showing the Brightfield Min projection, the Pillar mask, 
    and the final Cell Segmenation for the first, middle, and last frames.
    """
    T = vol_bf.shape[0]
    frames_to_check = [0, T//2, T-1]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Cell Segmentation Check (BF Channel)", fontsize=16)
    
    for i, t in enumerate(frames_to_check):
        bf_mip = np.min(vol_bf[t], axis=0)
        
        # Normalize for display
        mn, mx = bf_mip.min(), bf_mip.max()
        if mx > mn: bf_mip = (bf_mip - mn) / (mx - mn)
        
        axes[i, 0].imshow(bf_mip, cmap='gray')
        axes[i, 0].set_title(f"T={t}: BF Min Projection")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pillar_mask, cmap='magma')
        axes[i, 1].set_title(f"T={t}: Pillar Mask")
        axes[i, 1].axis('off')
        
        # Colorize the integer cell mask
        cmap = plt.cm.get_cmap('tab20', np.max(cell_masks[t])+1)
        axes[i, 2].imshow(cell_masks[t], cmap=cmap, interpolation='nearest')
        axes[i, 2].set_title(f"T={t}: Cell Labels")
        axes[i, 2].axis('off')
        
        # Composite
        rgb = np.stack([bf_mip]*3, axis=-1)
        # Overlay cell outlines
        edges = ndimage.morphological_gradient(cell_masks[t], size=(3,3)) > 0
        rgb[edges] = [1.0, 0.0, 0.0] # Red edges
        axes[i, 3].imshow(rgb)
        axes[i, 3].set_title(f"T={t}: Overlay")
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved segmentation visual check to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser("Filament 3D Tracker")
    parser.add_argument("input", help="Input 5D TIFF volume (T, Z, C, H, W)")
    parser.add_argument("--auto", action="store_true", help="Use auto-threshold temporal model")
    args = parser.parse_args()
    
    filepath = args.input
    if args.auto:
        model_path = "models/filament_unet3d_temporal_auto.pt"
    else:
        model_path = "models/filament_unet3d_temporal.pt"
        
    device = best_device()
    model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Error: Could not find model {model_path}")
        sys.exit(1)
    model.eval()
    
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    # Expected: (T, Z, C, H, W) -> we need C=0 (BF) and C=1 (Fil)
    if len(img.shape) != 5:
        print(f"Error: Expected 5D TIFF (T, Z, C, H, W), got {img.shape}")
        sys.exit(1)
        
    T, Z, C, H, W = img.shape
    vol_bf = img[:, :, 0, :, :]
    vol_fil = img[:, :, 1, :, :]
    
    # Normalize channels
    norm_fil = np.zeros_like(vol_fil)
    for t in range(T):
        mn, mx = vol_fil[t].min(), vol_fil[t].max()
        if mx > mn: norm_fil[t] = (vol_fil[t] - mn) / (mx - mn)
        
    # Temporal inference logic (sliding window)
    pred_masks = np.zeros((T, Z, H, W), dtype=np.float32)
    print("Running Temporal 3D Inference...")
    with torch.no_grad():
        for t in tqdm(range(T)):
            t_prev, t_curr, t_next = max(0, t-1), t, min(T-1, t+1)
            window = np.stack([norm_fil[t_prev], norm_fil[t_curr], norm_fil[t_next]], axis=0)
            inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
            logits = model(inp).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits[1])) # Middle frame of the sliding window
            pred_masks[t] = (probs > 0.5).astype(np.float32)
            
    # Compute global Pillar Mask (dark stationary objects)
    print("Identifying stationary pillars...")
    bf_min_global = np.min(vol_bf, axis=(0, 1)) # Min across time and Z
    # Pillars are extremely dark. We use a conservative threshold below the Otsu mean.
    pillar_thresh = otsu(bf_min_global) * 0.8
    pillar_mask = bf_min_global < pillar_thresh
    
    # Expand pillars slightly to be safe
    pillar_mask = ndimage.binary_dilation(pillar_mask, iterations=2)
    
    # Initialize Cellpose model once
    print("Initializing Cellpose model...")
    from cellpose import models
    cellpose_model = models.CellposeModel(model_type='cyto', gpu=torch.cuda.is_available() or torch.backends.mps.is_available())
    
    print("\nExtracting & Tracking Filaments...")
    all_data = []
    next_fil_id = 1
    prev_blobs = []
    
    # Storage for labeled volumes to show in Dashboard
    out_filament_labels = np.zeros((T, Z, H, W), dtype=np.int32)
    out_cell_masks = np.zeros((T, H, W), dtype=np.int32)
    
    for t in tqdm(range(T)):
        # 0. Filament mask (seed for cells)
        seed_mask_2d = np.max(pred_masks[t], axis=0) > 0.5
        
        # 1. Cell Segmentation from BF seeded by filaments and avoiding pillars
        cell_mask = segment_cells(vol_bf[t], seed_mask=seed_mask_2d, pillar_mask=pillar_mask, cellpose_model=cellpose_model)
        out_cell_masks[t] = cell_mask
        
        # 2. Z-Localization to remove multi-plane stretching
        localized_mask = z_localize(pred_masks[t], vol_fil[t])
        
        # 3. Measure components
        curr_blobs = measure_filaments(localized_mask, vol_fil[t], cell_mask)
        
        # 4. Cross-frame tracking
        curr_blobs, next_fil_id = track_filaments(prev_blobs, curr_blobs, next_fil_id)
        
        # Store for export
        for b in curr_blobs:
            b['time_min'] = t * TIME_INTERVAL_MIN
            b['frame'] = t
            all_data.append(dict(b))
            
            # Fill label volume
            z_i, y_i, x_i = b['indices']
            out_filament_labels[t, z_i, y_i, x_i] = b['filament_id']
            
        prev_blobs = curr_blobs.copy()
        
    # Export
    os.makedirs("results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_csv = f"results/{base_name}_tracking.csv"
    out_pdf = f"results/{base_name}_tracking_summary.pdf"
    out_png = f"results/{base_name}_tracking_summary.png"
    out_npz = f"results/{base_name}_analysis.npz"
    out_seg_check = f"results/{base_name}_segmentation_check.png"
    
    # Save segmentation check
    save_segmentation_check(vol_bf, out_cell_masks, pillar_mask, out_seg_check)
    
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        # Drop indices before saving CSV
        df_csv = df.drop(columns=['indices'])
        # Reorder columns
        cols = ['frame', 'time_min', 'cell_id', 'filament_id', 'size_px', 'length_um', 'mean_intensity', 'z', 'y', 'x']
        df_csv = df_csv[cols]
        df_csv.sort_values(by=['frame', 'filament_id'], inplace=True)
        df_csv.to_csv(out_csv, index=False)
        print(f"\n✅ Saved tracking data to {out_csv}")
        
        # Save NPZ for dashboard
        np.savez_compressed(out_npz, 
                           filament_labels=out_filament_labels,
                           cell_masks=out_cell_masks)
        print(f"✅ Saved analysis archives to {out_npz}")
        
        # Final Summary Stats and plots
        make_summary_plot(df_csv, out_pdf)
        make_summary_plot(df_csv, out_png)
        
        # Print Summary Stats
        print("\n--- Tracking Summary ---")
        print(f"Total filaments tracked: {df['filament_id'].nunique()}")
        print(f"Max filaments in one frame: {df.groupby('frame')['filament_id'].count().max()}")
        print(f"Mean size: {df['size_px'].mean():.1f} px")
        print(f"Mean length: {df['length_um'].mean():.2f} µm")
    else:
        print("\nℹ️ No filaments found to track.")

if __name__ == "__main__":
    main()
