#!/usr/bin/env python3
"""
Filament 3D Trainer (Auto-Thresholding + 3D Ridge Filter)

Trains a 2-channel 3D U-Net on annotated z-stack volumes.
Channel 1: Raw Image Volume
Channel 2: 3D Frangi Ridge Filter
Refines blob-like annotations dynamically using intensity thresholding.

Usage:
    python train_3d_ridge.py [tifs3d/volume1.tif ...]
    (If no files are specified, it will automatically find TIFFs matching the saved masks)
"""

import sys
import os
import glob
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unet3d import TinyUNet3D, SegDataset3D2ch, dice_loss
from utils import best_device

MASK_DIR = "models/masks3d"
SAVE_PATH = "models/filament_unet3d_ridge.pt"

def load_data(filepath):
    img = tifffile.imread(filepath).astype(np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    elif img.ndim >= 4:
        if img.ndim == 5:
            img = img[:, :, 1, :, :]  # Use SECOND channel
    
    T, Z, H, W = img.shape
    norm = np.zeros_like(img)
    for t in range(T):
        mn, mx = img[t].min(), img[t].max()
        if mx > mn:
            norm[t] = (img[t] - mn) / (mx - mn)
    return norm

def load_paired_volumes(tif_files):
    mask_lookup = {}
    for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask_lookup[fname] = np.load(p)

    if len(mask_lookup) == 0:
        print(f"Error: No masks found in {MASK_DIR}. Annotate some volumes first.")
        return None, None

    volumes, masks = [], []
    for fp in tif_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"Loading {fp}...")
        norm_data = load_data(fp)
        
        added_frames = set()
        for t in range(norm_data.shape[0]):
            key = f"{base}_t{t:04d}"
            if key in mask_lookup:
                volumes.append(norm_data[t])
                masks.append(mask_lookup[key])
                added_frames.add(t)
                
        # Force add first 3 frames as negative data if not already annotated
        for t in range(min(3, norm_data.shape[0])):
            if t not in added_frames:
                volumes.append(norm_data[t])
                masks.append(np.zeros_like(norm_data[t], dtype=np.float32))

    pos = sum(1 for m in masks if m.max() > 0)
    neg = len(masks) - pos
    print(f"Loaded {len(volumes)} total 3D volumes ({pos} positive explicitly annotated + {neg} empty/negative backgrounds)")
    
    if pos == 0:
        print("Error: None of the loaded videos match the available masks.")
        return None, None
        
    return volumes, masks

def discover_tifs():
    """Finds TIFF files corresponding to available masks."""
    masks = glob.glob(os.path.join(MASK_DIR, "*.npy"))
    if not masks:
        return []
    
    tif_bases = set()
    for m in masks:
        base = os.path.basename(m)
        if "_t" in base:
            tif_bases.add(base.rsplit("_t", 1)[0])
        else:
            tif_bases.add(os.path.splitext(base)[0])
            
    found_tifs = []
    search_dirs = [".", "tiffs3d"]
    for base in sorted(tif_bases):
        found = False
        for d in search_dirs:
            path = os.path.join(d, f"{base}.tif")
            if os.path.exists(path):
                found_tifs.append(path)
                found = True
                break
        if not found:
            print(f"Warning: Could not find TIFF for mask base '{base}' in {search_dirs}")
            
    return found_tifs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("tif_files", nargs="*", help="Optional list of TIFF files")
    args = parser.parse_args()

    tif_files = args.tif_files
    if not tif_files:
        print("No volumes specified. Discovering from masks...")
        tif_files = discover_tifs()
        if not tif_files:
            print(f"Error: No TIFF files found for masks in {MASK_DIR}.")
            sys.exit(1)
        print(f"Discovered {len(tif_files)} TIFF files: {', '.join(tif_files)}")

    volumes, masks = load_paired_volumes(tif_files)
    if volumes is None:
        sys.exit(1)

    device = best_device()
    # INITIALIZE 2-CHANNEL U-NET
    model = TinyUNet3D(in_ch=2).to(device)
    print(f"\n3D Model params (2-ch, Ridge): {sum(p.numel() for p in model.parameters()):,} ({device})")

    # USE 2-CHANNEL DATASET
    ds = SegDataset3D2ch(volumes, masks, augment_factor=10, intensity_thresh=0.5)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    # Calculate pos weight dynamically based on REFINED masks
    refined_masks = [((m > 0) & (v > 0.5)).astype(np.float32) for m, v in zip(masks, volumes)]
    pos_pixels = max(float(sum(m.sum() for m in refined_masks)), 1.0)
    tot_pixels = float(sum(m.size for m in refined_masks))
    raw_weight = (tot_pixels - pos_pixels) / pos_pixels
    pos_weight = torch.tensor([min(raw_weight, 10.0)]).to(device)
    print(f"Class balance pos_weight = {pos_weight.item():.1f} (raw={raw_weight:.0f})")
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining for {args.epochs} epochs...")
    model.train()
    for epoch in range(args.epochs):
        tloss, tdice, nb = 0, 0, 0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            
            # Combine BCE and Dice
            loss = 0.7 * bce_loss(out, by) + 0.3 * dice_loss(out, by)
            loss.backward()
            opt.step()
            
            with torch.no_grad():
                p = torch.sigmoid(out) > 0.5
                tdice += (2*(p*by).sum() / (p.sum() + by.sum() + 1e-8)).item()
                
            tloss += loss.item()
            nb += 1
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/30  Loss:{tloss/nb:.4f}  Dice:{tdice/nb:.3f}")
            
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nSaved trained 2-ch 3D model to {SAVE_PATH} (Final Dice: {tdice/nb:.3f})")

if __name__ == "__main__":
    main()
