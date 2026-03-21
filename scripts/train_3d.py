#!/usr/bin/env python3
"""
Filament 3D Trainer

Trains the 3D U-Net on annotated z-stack volumes.
Usage:
    python train_3d.py tifs3d/volume1.tif [tifs3d/volume2.tif ...]
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

from unet3d import TinyUNet3D, SegDataset3D, dice_loss
from utils import best_device

MASK_DIR = "models/masks3d"
SAVE_PATH = "models/filament_unet3d.pt"

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
        for z in range(Z):
            mn, mx = img[t, z].min(), img[t, z].max()
            if mx > mn:
                norm[t, z] = (img[t, z] - mn) / (mx - mn)
    return norm

def load_paired_volumes(tif_files):
    """
    Loads all timepoints from the videos. 
    Annotated volumes get their mask, unannotated get an all-zero mask 
    to serve as negative backgrounds.
    """
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
        
        for t in range(norm_data.shape[0]):
            key = f"{base}_t{t:04d}"
            if key in mask_lookup:
                volumes.append(norm_data[t])
                masks.append(mask_lookup[key])

    pos = sum(1 for m in masks if m.max() > 0)
    neg = len(masks) - pos
    print(f"Loaded {len(volumes)} explicitly annotated 3D volumes ({pos} positive + {neg} empty masks)")
    
    if pos == 0:
        print("Error: None of the loaded videos match the available masks.")
        return None, None
        
    return volumes, masks

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_3d.py <volume1.tif> [volume2.tif ...]")
        sys.exit(1)

    tif_files = sys.argv[1:]
    volumes, masks = load_paired_volumes(tif_files)
    if volumes is None:
        sys.exit(1)

    device = best_device()
    model = TinyUNet3D().to(device)
    print(f"\n3D Model params: {sum(p.numel() for p in model.parameters()):,} ({device})")

    ds = SegDataset3D(volumes, masks, augment_factor=10)
    # Batch size 2 or 4 since 3D data takes more RAM
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    pos_pixels = max(float(sum(m.sum() for m in masks)), 1.0)
    tot_pixels = float(sum(m.size for m in masks))
    raw_weight = (tot_pixels - pos_pixels) / pos_pixels
    pos_weight = torch.tensor([min(raw_weight, 10.0)]).to(device)
    print(f"Class balance pos_weight = {pos_weight.item():.1f} (raw={raw_weight:.0f})")
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining for 30 epochs...")
    model.train()
    for epoch in range(30):
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
    print(f"\nSaved trained 3D model to {SAVE_PATH} (Final Dice: {tdice/nb:.3f})")

if __name__ == "__main__":
    main()
