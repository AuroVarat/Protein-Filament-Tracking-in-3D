#!/usr/bin/env python3
"""
Filament 3D Temporal Trainer

Trains a 3D U-Net taking 3 temporal frames (t-1, t, t+1) and predicting masks for all 3.
Initializes with weights from the 1-channel model.

Usage:
    python scripts/train_3d_temporal.py
"""

import sys
import os
import glob
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from unet3d import TinyUNet3D, SegDataset3DTemporal
from utils import best_device

MASK_DIR = "models/masks3d"
SAVE_PATH = "models/filament_unet3d_temporal.pt"
BASE_MODEL_PATH = "models/filament_unet3d.pt"

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

def load_temporal_volumes(tif_files):
    mask_lookup = {}
    for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask_lookup[fname] = np.load(p)

    if len(mask_lookup) == 0:
        print(f"Error: No masks found in {MASK_DIR}.")
        return None, None, None

    sequences = []
    mask_seqs = []
    valid_seqs = []
    
    for fp in tif_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"Loading {fp}...")
        norm_data = load_data(fp)
        T, Z, H, W = norm_data.shape
        
        seq_mask = np.zeros_like(norm_data, dtype=np.float32)
        seq_valid = np.zeros(T, dtype=np.float32)
        
        has_any_annotation = False
        
        for t in range(T):
            key = f"{base}_t{t:04d}"
            if key in mask_lookup:
                seq_mask[t] = mask_lookup[key]
                seq_valid[t] = 1.0
                has_any_annotation = True
                
        # If the video has annotations, enforce first 3 frames as negative if not annotated
        if has_any_annotation:
            for t in range(min(3, T)):
                if seq_valid[t] == 0:
                    seq_valid[t] = 1.0  # mask is already 0
                    
            sequences.append(norm_data)
            mask_seqs.append(seq_mask)
            valid_seqs.append(seq_valid)

    if len(sequences) == 0:
        print("Error: No matching videos found.")
        return None, None, None
        
    return sequences, mask_seqs, valid_seqs

def discover_tifs():
    masks = glob.glob(os.path.join(MASK_DIR, "*.npy"))
    if not masks: return []
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
    return found_tifs

def temporal_loss(logits, targets, valid, pos_weight):
    probs = torch.sigmoid(logits)
    
    # BCE
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    bce = bce_loss_fn(logits, targets)
    bce = (bce * valid).sum() / (valid.sum() * logits[0,0].numel() + 1e-8)
    
    # Dice
    valid_probs = probs * valid
    valid_targets = targets * valid
    inter = (valid_probs * valid_targets).sum()
    dice = 1 - (2 * inter + 1.0) / (valid_probs.sum() + valid_targets.sum() + 1.0)
    
    return 0.7 * bce + 0.3 * dice

def main():
    import argparse
    import csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("tif_files", nargs="*", help="Optional list of TIFF files")
    args = parser.parse_args()

    tif_files = args.tif_files
    if not tif_files:
        tif_files = discover_tifs()
        if not tif_files:
            print(f"Error: No TIFF files found for masks in {MASK_DIR}.")
            sys.exit(1)
        print(f"Discovered {len(tif_files)} TIFF files: {', '.join(tif_files)}")

    sequences, masks, valid_masks = load_temporal_volumes(tif_files)
    if sequences is None: sys.exit(1)

    device = best_device()
    model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
    
    if os.path.exists(BASE_MODEL_PATH):
        print(f"Initializing Temporal model from {BASE_MODEL_PATH}")
        state_dict = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=True)
        new_state_dict = model.state_dict()
        
        for k, v in state_dict.items():
            if k == "enc1.block.0.weight":
                new_state_dict[k] = v.repeat(1, 3, 1, 1, 1) / 3.0
            elif k == "out_conv.weight":
                new_state_dict[k] = v.repeat(3, 1, 1, 1, 1)
            elif k == "out_conv.bias":
                new_state_dict[k] = v.repeat(3)
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    else:
        print(f"Warning: Base model {BASE_MODEL_PATH} not found. Training from scratch.")

    print(f"\nTemporal 3D Model params: {sum(p.numel() for p in model.parameters()):,} ({device})")

    ds = SegDataset3DTemporal(sequences, masks, valid_masks, augment_factor=10)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    # Calculate pos weight from valid masks only
    pos_px = sum((m * v[:, None, None, None]).sum() for m, v in zip(masks, valid_masks))
    tot_px = sum(v.sum() * m[0].size for m, v in zip(masks, valid_masks))
    
    pos_px = max(float(pos_px), 1.0)
    tot_px = float(tot_px)
    raw_w = (tot_px - pos_px) / pos_px
    pos_w = torch.tensor([min(raw_w, 10.0)]).to(device)
    print(f"Class balance pos_weight = {pos_w.item():.1f} (raw={raw_w:.0f})")
    
    log_file = "models/train_temporal_log.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Dice"])

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        tloss, tdice, nb = 0, 0, 0
        for bx, by, bvalid in dl:
            bx, by, bvalid = bx.to(device), by.to(device), bvalid.to(device)
            
            # Skip batches with absolutely no valid frames
            if bvalid.sum() == 0:
                continue
                
            opt.zero_grad()
            out = model(bx)
            
            loss = temporal_loss(out, by, bvalid, pos_w)
            loss.backward()
            opt.step()
            
            with torch.no_grad():
                p = torch.sigmoid(out) > 0.5
                valid_p = p * bvalid
                valid_y = by * bvalid
                dice = (2 * (valid_p * valid_y).sum()) / (valid_p.sum() + valid_y.sum() + 1e-8)
                tdice += dice.item()
                
            tloss += loss.item()
            nb += 1
            
        if nb > 0 and ((epoch+1) % 5 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:2d}/{args.epochs}  Loss:{tloss/nb:.4f}  Dice:{tdice/nb:.3f}")
            
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, tloss/max(nb, 1), tdice/max(nb, 1)])
            
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nSaved trained Temporal 3D model to {SAVE_PATH} (Final Dice: {tdice/nb:.3f})")

if __name__ == "__main__":
    main()
