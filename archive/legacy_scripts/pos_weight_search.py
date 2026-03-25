#!/usr/bin/env python3
"""
pos_weight Hyperparameter Search

Splits annotated masks 80/20 train/val, trains with a range of pos_weight values,
and reports validation Dice for each. Use the value with the best val Dice.

Usage:
    python scripts/pos_weight_search.py tifs/video1.tif tifs/video2.tif [--ridge]
"""

import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import os, sys, glob, random
from utils import best_device

sys.path.insert(0, os.path.dirname(__file__))
from filament_boxer import ridge_filter_single, TinyUNet, TinyUNet2ch, SegDataset, SegDataset2ch, MASK_DIR


# Values to search
POS_WEIGHTS  = [1, 2, 5, 10, 20, 50]
TRAIN_EPOCHS = 20   # fewer epochs to keep search fast
BATCH_SIZE   = 8
VAL_FRACTION = 0.2  # 20% held out for validation

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(tif_files):
    mask_lookup = {}
    for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask_lookup[fname] = np.load(p)

    images, masks = [], []
    for fp in tif_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        img = tifffile.imread(fp).astype(np.float32)
        for i in range(img.shape[0]):
            mn, mx = img[i].min(), img[i].max()
            frame = (img[i] - mn) / (mx - mn) if mx > mn else np.zeros_like(img[i])
            key = f"{base}_{i:04d}"
            mask = mask_lookup.get(key, np.zeros(frame.shape, dtype=np.float32))
            images.append(frame); masks.append(mask)

    return images, masks

# ── Training with a given pos_weight ─────────────────────────────────────────
def train_and_eval(train_imgs, train_masks, val_imgs, val_masks,
                   pos_weight_val, use_ridge, device):
    ModelClass = TinyUNet2ch if use_ridge else TinyUNet
    DataClass  = SegDataset2ch if use_ridge else SegDataset
    model = ModelClass().to(device)
    ds = DataClass(train_imgs, train_masks, augment_factor=8)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    pos_weight = torch.tensor([float(pos_weight_val)]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(logits, y, smooth=1.0):
        p = torch.sigmoid(logits)
        return 1 - (2*(p*y).sum() + smooth) / (p.sum() + y.sum() + smooth)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            loss = 0.7 * bce(out, by) + 0.3 * dice_loss(out, by)
            loss.backward(); opt.step()

    # Evaluate on validation set
    model.eval()
    val_dice_scores = []
    with torch.no_grad():
        for img, mask in zip(val_imgs, val_masks):
            if use_ridge:
                ridge = ridge_filter_single(img)
                inp = torch.from_numpy(np.stack([img, ridge], axis=0)).float().unsqueeze(0).to(device)
            else:
                inp = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred = (prob > 0.5).astype(np.float32)
            inter = (pred * mask).sum()
            dice = (2 * inter) / (pred.sum() + mask.sum() + 1e-8)
            val_dice_scores.append(dice)

    return float(np.mean(val_dice_scores))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args      = sys.argv[1:]
    use_ridge = '--ridge' in args
    tif_files = [a for a in args if not a.startswith('--')]

    if not tif_files:
        print("Usage: python scripts/pos_weight_search.py <video1.tif> [...] [--ridge]")
        sys.exit(1)

    device = best_device()
    label  = "2-ch (raw+ridge)" if use_ridge else "1-ch (raw only)"
    print(f"pos_weight search — {label} — device: {device}")
    print(f"Loading data from {len(tif_files)} video(s)...")

    all_imgs, all_masks = load_data(tif_files)
    n = len(all_imgs)

    # Split: use only annotated frames for val so results are meaningful
    annotated_idx = [i for i, m in enumerate(all_masks) if m.max() > 0]
    random.seed(42)
    random.shuffle(annotated_idx)
    n_val = max(1, int(len(annotated_idx) * VAL_FRACTION))
    val_idx   = set(annotated_idx[:n_val])
    train_idx = [i for i in range(n) if i not in val_idx]

    train_imgs  = [all_imgs[i]  for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    val_imgs    = [all_imgs[i]  for i in val_idx]
    val_masks   = [all_masks[i] for i in val_idx]

    pos_annot = sum(1 for m in train_masks if m.max() > 0)
    print(f"Train: {len(train_imgs)} frames ({pos_annot} annotated) | Val: {len(val_imgs)} annotated frames")
    print(f"Searching pos_weight in {POS_WEIGHTS} ...\n")
    print(f"  {'pos_weight':>12}  {'Val Dice':>10}")
    print(f"  {'─'*12}  {'─'*10}")

    best_weight, best_dice = None, -1
    results = []
    for pw in POS_WEIGHTS:
        dice = train_and_eval(train_imgs, train_masks, val_imgs, val_masks,
                              pw, use_ridge, device)
        marker = " <-- best" if dice > best_dice else ""
        print(f"  {pw:>12}  {dice:>10.4f}{marker}")
        results.append((pw, dice))
        if dice > best_dice:
            best_dice = dice; best_weight = pw

    print(f"\nBest pos_weight = {best_weight}  (Val Dice = {best_dice:.4f})")
    print(f"\nTo use this value, edit filament_boxer.py line:")
    print(f"    pos_weight = torch.tensor([min(raw_weight, 10.0)])")
    print(f"  → pos_weight = torch.tensor([min(raw_weight, {best_weight}.0)])")
    print(f"\nOr retrain with the best weight directly:")
    print(f"  from filament_boxer import train_unet{'_ridge' if use_ridge else ''}")
    print(f"  train_unet{'_ridge' if use_ridge else ''}(tif_files)")

if __name__ == "__main__":
    main()
