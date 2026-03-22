#!/usr/bin/env python3
"""
Filament 2D Temporal Trainer (Unified 2D + 3D-Slices)

Trains a 2D U-Net taking 3 temporal frames (t-1, t, t+1).
This unified version uses data from BOTH tifs2d/ and tiffs3d/ (treating each Z-plane as a 2D video).
Supports optional Auto-Thresholding with --auto.

Usage:
    python scripts/train_2d_temporal.py [--auto] [--epochs 50]
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
import argparse
import csv

from unet2d import TinyUNet2D, SegDataset2DTemporal, SegDataset2DTemporalAuto
from utils import best_device

MASK2D_DIR = "models/masks"
MASK3D_DIR = "models/masks3d"
TIFS2D_DIR = "tifs2d"
TIFS3D_DIR = "tiffs3d"
SAVE_PATH_BASE = "models/filament_unet2d_temporal"
BASE_MODEL_PATH = "models/filament_unet.pt"

def load_2d_data(filepath):
    img = tifffile.imread(filepath).astype(np.float32)
    # img is (T, H, W)
    mn, mx = img.min(), img.max()
    if mx > mn:
        norm = (img - mn) / (mx - mn)
    else:
        norm = np.zeros_like(img)
    return norm

def load_3d_data(filepath):
    img = tifffile.imread(filepath).astype(np.float32)
    # Standardise to (T, Z, H, W)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    elif img.ndim == 5:
        img = img[:, :, 1, :, :]  # Use SECOND channel
    
    # Normalize per full video volume
    mn, mx = img.min(), img.max()
    if mx > mn:
        norm = (img - mn) / (mx - mn)
    else:
        norm = np.zeros_like(img)
    return norm

def load_all_sequences():
    sequences = []
    mask_seqs = []
    valid_seqs = []

    # --- 1. Load 2D Data ---
    mask2d_lookup = {}
    for p in sorted(glob.glob(os.path.join(MASK2D_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask2d_lookup[fname] = np.load(p)

    tif2d_files = glob.glob(os.path.join(TIFS2D_DIR, "*.tif"))
    for fp in tif2d_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"Loading 2D: {fp}...")
        data = load_2d_data(fp)
        T, H, W = data.shape
        
        seq_mask = np.zeros_like(data, dtype=np.float32)
        seq_valid = np.zeros(T, dtype=np.float32)
        has_any = False
        for t in range(T):
            key = f"{base}_{t:04d}"
            if key in mask2d_lookup:
                seq_mask[t] = mask2d_lookup[key]
                seq_valid[t] = 1.0
                has_any = True
        
        if has_any:
            for t in range(min(3, T)):
                if seq_valid[t] == 0: seq_valid[t] = 1.0
            sequences.append(data)
            mask_seqs.append(seq_mask)
            valid_seqs.append(seq_valid)

    # --- 2. Load 3D Data (Slicing into 2D sequences) ---
    mask3d_lookup = {}
    for p in sorted(glob.glob(os.path.join(MASK3D_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask3d_lookup[fname] = np.load(p)

    tif3d_files = glob.glob(os.path.join(TIFS3D_DIR, "*.tif"))
    for fp in tif3d_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"Loading 3D (to slices): {fp}...")
        data_3d = load_3d_data(fp)
        T, Z, H, W = data_3d.shape
        
        # Build full mask volume for this video
        full_mask_3d = np.zeros_like(data_3d, dtype=np.float32)
        full_valid_3d = np.zeros((T, Z), dtype=np.float32)
        has_any_3d = False
        
        for t in range(T):
            key = f"{base}_t{t:04d}"
            if key in mask3d_lookup:
                full_mask_3d[t] = mask3d_lookup[key]
                full_valid_3d[t, :] = 1.0 # If one slice is annotated, we assume whole stack is checked
                has_any_3d = True
        
        if has_any_3d:
            # For each Z plane, create a sequence
            for z in range(Z):
                seq_data = data_3d[:, z, :, :] # (T, H, W)
                seq_mask = full_mask_3d[:, z, :, :] # (T, H, W)
                seq_valid = full_valid_3d[:, z].copy() # (T,)
                
                # Force first 3 frames negative
                for t in range(min(3, T)):
                    if seq_valid[t] == 0: seq_valid[t] = 1.0
                
                sequences.append(seq_data)
                mask_seqs.append(seq_mask)
                valid_seqs.append(seq_valid)

    print(f"Total 2D temporal sequences loaded: {len(sequences)}")
    return sequences, mask_seqs, valid_seqs

def temporal_loss(logits, targets, valid, pos_weight):
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    bce = bce_loss_fn(logits, targets)
    bce = (bce * valid).sum() / (valid.sum() * logits[0,0].numel() + 1e-8)
    
    probs = torch.sigmoid(logits)
    valid_probs = probs * valid
    valid_targets = targets * valid
    inter = (valid_probs * valid_targets).sum()
    dice = 1 - (2 * inter + 1.0) / (valid_probs.sum() + valid_targets.sum() + 1.0)
    return 0.7 * bce + 0.3 * dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Use dilated auto-threshold dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    args = parser.parse_args()

    sequences, masks, valid_masks = load_all_sequences()
    if not sequences:
        print("Error: No training data found.")
        sys.exit(1)

    device = best_device()
    model = TinyUNet2D(in_ch=3, out_ch=3).to(device)
    
    if os.path.exists(BASE_MODEL_PATH):
        print(f"Initializing from {BASE_MODEL_PATH}")
        state_dict = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=True)
        new_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if k == "enc1.block.0.weight": new_state_dict[k] = v.repeat(1, 3, 1, 1) / 3.0
            elif k == "out_conv.weight": new_state_dict[k] = v.repeat(3, 1, 1, 1)
            elif k == "out_conv.bias": new_state_dict[k] = v.repeat(3)
            else: new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    suffix = "_auto" if args.auto else ""
    save_path = f"{SAVE_PATH_BASE}{suffix}.pt"
    log_file = f"models/train_2d_temporal{suffix}_log.csv"

    DatasetClass = SegDataset2DTemporalAuto if args.auto else SegDataset2DTemporal
    ds = DatasetClass(sequences, masks, valid_masks, augment_factor=10)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    pos_px = sum((m * v[:, None, None]).sum() for m, v in zip(masks, valid_masks))
    tot_px = sum(v.sum() * m[0].size for m, v in zip(masks, valid_masks))
    pos_w = torch.tensor([min((float(tot_px) - float(pos_px)) / max(float(pos_px), 1.0), 10.0)]).to(device)
    
    os.makedirs("models", exist_ok=True)
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Loss", "Dice"])

    print(f"\nTraining for {args.epochs} epochs... (Auto: {args.auto})")
    for epoch in range(args.epochs):
        model.train()
        tloss, tdice, nb = 0, 0, 0
        for bx, by, bvalid in dl:
            bx, by, bvalid = bx.to(device), by.to(device), bvalid.to(device)
            if bvalid.sum() == 0: continue
            opt.zero_grad()
            out = model(bx)
            loss = temporal_loss(out, by, bvalid, pos_w)
            loss.backward(); opt.step()
            with torch.no_grad():
                p = torch.sigmoid(out) > 0.5
                dice = (2 * (p * bvalid * by * bvalid).sum()) / ((p * bvalid).sum() + (by * bvalid).sum() + 1e-8)
                tdice += dice.item()
            tloss += loss.item(); nb += 1
            
        if nb > 0:
            print(f"  Epoch {epoch+1:2d}/{args.epochs}  Loss:{tloss/nb:.4f}  Dice:{tdice/nb:.3f}")
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, tloss/nb, tdice/nb])
            
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved trained 2D Temporal model to {save_path}")

if __name__ == "__main__":
    main()
