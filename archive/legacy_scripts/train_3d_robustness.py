#!/usr/bin/env python3
"""
Filament 3D Training Benchmarking & Robustness Script (with Error Bars)

This script evaluates the generalization performance and stability of the 3D U-Net models 
by performing multiple independent Train/Validation splits (repeated random sub-sampling).
It reports the mean and standard deviation of the Validation Dice scores across all runs.

Usage:
    python scripts/train_3d_robustness.py [--ridge] [--epochs 30] [--split 0.2] [--runs 5]
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
import random

from unet3d import TinyUNet3D, SegDataset3D, SegDataset3D2ch, dice_loss
from utils import best_device

MASK_DIR = "models/masks3d"

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
        print(f"Error: No masks found in {MASK_DIR}.")
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
                
        for t in range(min(3, norm_data.shape[0])):
            if t not in added_frames:
                volumes.append(norm_data[t])
                masks.append(np.zeros_like(norm_data[t], dtype=np.float32))

    return volumes, masks

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

def run_single_benchmark(run_idx, data, args, device):
    # Use a different seed for each run for the split
    random.seed(42 + run_idx)
    random.shuffle(data)
    
    val_size = int(len(data) * args.split)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    train_vols, train_masks = zip(*train_data)
    val_vols, val_masks = zip(*val_data)
    
    in_ch = 2 if args.ridge else 1
    model = TinyUNet3D(in_ch=in_ch).to(device)
    
    DatasetClass = SegDataset3D2ch if args.ridge else SegDataset3D
    train_ds = DatasetClass(train_vols, train_masks, augment_factor=10)
    val_ds   = DatasetClass(val_vols, val_masks, augment_factor=1)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    pos_px = max(float(sum(m.sum() for m in train_masks)), 1.0)
    tot_px = float(sum(m.size for m in train_masks))
    raw_w = (tot_px - pos_px) / pos_px
    pos_w = torch.tensor([min(raw_w, 10.0)]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_val_dice = 0.0
    print(f"\n[Run {run_idx+1}/{args.runs}] Train on {len(train_vols)}, Val on {len(val_vols)}")

    for epoch in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = 0.7 * bce_loss(out, by) + 0.3 * dice_loss(out, by)
            loss.backward()
            opt.step()
        
        model.eval()
        v_dice = 0; nb_v = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                p = torch.sigmoid(out) > 0.5
                v_dice += (2*(p*by).sum() / (p.sum() + by.sum() + 1e-8)).item()
                nb_v += 1
        
        avg_v_dice = v_dice / max(1, nb_v)
        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{args.epochs}  Current Val Dice: {avg_v_dice:.3f}")

    print(f"  ✓ Run {run_idx+1} complete. Best Val Dice: {best_val_dice:.3f}")
    return best_val_dice

def main():
    parser = argparse.ArgumentParser(description="3D U-Net Robustness Benchmark with Error Bars")
    parser.add_argument("--ridge", action="store_true", help="Use 2-channel Ridge model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs per run")
    parser.add_argument("--split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated runs")
    args = parser.parse_args()

    tif_files = discover_tifs()
    if not tif_files:
        print(f"Error: No TIFF files found for masks in {MASK_DIR}.")
        sys.exit(1)

    print(f"\n--- Loading Data ---")
    volumes, masks = load_paired_volumes(tif_files)
    if volumes is None: sys.exit(1)
    data = list(zip(volumes, masks))
    device = best_device()
    model_name = "Ridge" if args.ridge else "Original"
    
    print(f"\n--- Starting {args.runs} benchmark runs for {model_name} model ---")
    all_best_dice = []
    
    for i in range(args.runs):
        best_dice = run_single_benchmark(i, data, args, device)
        all_best_dice.append(best_dice)

    mean_dice = np.mean(all_best_dice)
    std_dice = np.std(all_best_dice)

    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY: {model_name} Model")
    print(f"  Number of runs: {args.runs}")
    print(f"  Individual Best Dices: {[f'{d:.3f}' for d in all_best_dice]}")
    print(f"\n  ➤ FINAL ROBUSTNESS SCORE: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
