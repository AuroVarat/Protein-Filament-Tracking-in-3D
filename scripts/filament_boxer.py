#!/usr/bin/env python3
"""
Filament Boxer — Rectangle Annotation + Auto-Threshold Mask + U-Net Training

Instead of painting pixel-by-pixel, draw a rectangle around each filament.
The script automatically segments it using a 0.5 intensity threshold inside the box.

Usage:
    python filament_boxer.py tifs/video1.tif [tifs/video2.tif ...]

Controls:
    Click & drag on RIGHT panel = Draw a bounding box
    S                           = Save current box as mask
    C                           = Clear current box
    ← →                         = Navigate frames
"""

import tifffile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RectangleSelector
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import os
import sys
import glob


# ─── Ridge Filter ────────────────────────────────────────────────────────────

def ridge_filter_single(img, sigma=1.5, beta=0.5, c_thresh=0.02):
    """Frangi vesselness filter — highlights straight-line structures."""
    Hxx = ndimage.gaussian_filter(img, sigma, order=[2, 0])
    Hyy = ndimage.gaussian_filter(img, sigma, order=[0, 2])
    Hxy = ndimage.gaussian_filter(img, sigma, order=[1, 1])
    trace = Hxx + Hyy
    det_diff = np.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)
    l1 = 0.5 * (trace + det_diff); l2 = 0.5 * (trace - det_diff)
    mag_l1, mag_l2 = np.abs(l1), np.abs(l2)
    lambda1 = np.where(mag_l1 > mag_l2, l1, l2)
    lambda2 = np.where(mag_l1 > mag_l2, l2, l1)
    l1_safe = np.where(lambda1 == 0, 1e-10, lambda1)
    Rb = np.abs(lambda2) / np.abs(l1_safe)
    S  = np.sqrt(lambda1**2 + lambda2**2)
    c  = max(np.max(S) / 3.0, c_thresh)
    V  = np.exp(-(Rb**2) / (2*beta**2)) * (1 - np.exp(-(S**2) / (2*c**2)))
    V[lambda1 > 0] = 0
    return V.astype(np.float32)

# ─── U-Net Architectures ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class TinyUNet(nn.Module):
    """1-channel U-Net: raw frame only."""
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 16);  self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64); self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64,  32, 2, stride=2); self.dec2 = ConvBlock(64,  32)
        self.up1 = nn.ConvTranspose2d(32,  16, 2, stride=2); self.dec1 = ConvBlock(32,  16)
        self.out_conv = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

class TinyUNet2ch(nn.Module):
    """2-channel U-Net: raw frame + ridge filter."""
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 16);  self.enc2 = ConvBlock(16, 32)  # 2 input channels!
        self.enc3 = ConvBlock(32, 64); self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64,  32, 2, stride=2); self.dec2 = ConvBlock(64,  32)
        self.up1 = nn.ConvTranspose2d(32,  16, 2, stride=2); self.dec1 = ConvBlock(32,  16)
        self.out_conv = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """1-channel dataset: raw frame only."""
    def __init__(self, images, masks, augment_factor=10):
        self.images = images; self.masks = masks; self.aug = augment_factor
    def __len__(self): return len(self.images) * self.aug
    def __getitem__(self, idx):
        ri = idx // self.aug
        img, mask = self.images[ri].copy(), self.masks[ri].copy()
        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            if np.random.rand() > 0.5: img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() > 0.5: img, mask = img[::-1].copy(), mask[::-1].copy()
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, reshape=False, order=0)
            img  = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
            mask = (mask > 0.5).astype(np.float32)
        return (torch.from_numpy(img).float().unsqueeze(0),
                torch.from_numpy(mask).float().unsqueeze(0))

class SegDataset2ch(Dataset):
    """2-channel dataset: raw frame + ridge filter."""
    def __init__(self, images, masks, augment_factor=10):
        self.images = images; self.masks = masks; self.aug = augment_factor
    def __len__(self): return len(self.images) * self.aug
    def __getitem__(self, idx):
        ri = idx // self.aug
        img, mask = self.images[ri].copy(), self.masks[ri].copy()
        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            if np.random.rand() > 0.5: img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() > 0.5: img, mask = img[::-1].copy(), mask[::-1].copy()
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, reshape=False, order=0)
            img  = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
            mask = (mask > 0.5).astype(np.float32)
        ridge = ridge_filter_single(img)  # compute on (possibly augmented) frame
        two_ch = np.stack([img, ridge], axis=0)  # 2×H×W
        return (torch.from_numpy(two_ch).float(),
                torch.from_numpy(mask).float().unsqueeze(0))

# ─── Mask I/O ────────────────────────────────────────────────────────────────

MASK_DIR = "models/masks"

def mask_path(filepath, frame_idx):
    os.makedirs(MASK_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(MASK_DIR, f"{base}_{frame_idx:04d}.npy")

def save_mask(filepath, frame_idx, mask):
    np.save(mask_path(filepath, frame_idx), mask)

def load_mask(filepath, frame_idx, shape):
    p = mask_path(filepath, frame_idx)
    if os.path.exists(p): return np.load(p)
    return np.zeros(shape, dtype=np.float32)

def count_saved():
    os.makedirs(MASK_DIR, exist_ok=True)
    return sum(1 for p in glob.glob(os.path.join(MASK_DIR, "*.npy"))
               if np.load(p).max() > 0)

# ─── Training ────────────────────────────────────────────────────────────────

def _load_paired(tif_files):
    """
    Load ALL frames from training videos.
    - Annotated frames: use the saved mask.
    - Unannotated frames: use an all-zero mask (negative examples).
    This is essential so the model learns what background looks like.
    """
    # Build lookup of existing masks
    mask_lookup = {}  # basename_framestr -> mask array
    for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.npy"))):
        fname = os.path.splitext(os.path.basename(p))[0]
        mask_lookup[fname] = np.load(p)

    if len(mask_lookup) < 3:
        print(f"Only {len(mask_lookup)} masks. Need at least 3."); return None, None

    images, masks = [], []
    for fp in tif_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        img = tifffile.imread(fp).astype(np.float32)
        for i in range(img.shape[0]):
            mn, mx = img[i].min(), img[i].max()
            frame = (img[i] - mn) / (mx - mn) if mx > mn else np.zeros_like(img[i])
            key = f"{base}_{i:04d}"
            mask = mask_lookup.get(key, np.zeros(frame.shape, dtype=np.float32))
            images.append(frame)
            masks.append(mask)

    pos = sum(1 for m in masks if m.max() > 0)
    neg = len(masks) - pos
    print(f"  Frames: {len(images)} total  ({pos} annotated + {neg} background negatives)")
    return images, masks

def _run_training(model, dataset, save_path, device):
    dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    # Weighted BCE: compensate for class imbalance (filament pixels << background)
    # Cap at 10 to avoid the model predicting positives everywhere
    pos_pixels = max(float(sum(m.sum() for m in dataset.masks)), 1.0)
    tot_pixels = float(sum(m.size for m in dataset.masks))
    raw_weight = (tot_pixels - pos_pixels) / pos_pixels
    pos_weight = torch.tensor([min(raw_weight, 10.0)]).to(device)
    print(f"  Class balance: pos_weight = {min(raw_weight, 10.0):.1f} (raw={raw_weight:.0f}, capped at 10)")
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        return 1 - (2 * inter + smooth) / (probs.sum() + targets.sum() + smooth)

    model.train()
    for epoch in range(30):
        tloss, tdice, nb = 0, 0, 0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(bx)
            # Combined loss: 70% BCE + 30% Dice (standard for imbalanced segmentation)
            loss = 0.7 * bce_loss(out, by) + 0.3 * dice_loss(out, by)
            loss.backward(); opt.step()
            with torch.no_grad():
                p = torch.sigmoid(out) > 0.5
                tdice += (2*(p*by).sum() / (p.sum() + by.sum() + 1e-8)).item()
            tloss += loss.item(); nb += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/30  Loss:{tloss/nb:.4f}  Dice:{tdice/nb:.3f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to {save_path}  Dice:{tdice/nb:.3f}")
    print(f"{'─'*50}\n")



def train_unet(tif_files, save_path="models/filament_unet.pt"):
    images, masks = _load_paired(tif_files)
    if images is None: return
    print(f"\n{'─'*50}")
    print(f"[1-ch] Training on {len(images)} frames  (raw only)")
    device = best_device()
    model = TinyUNet().to(device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params on {device}")
    _run_training(model, SegDataset(images, masks), save_path, device)

def train_unet_ridge(tif_files, save_path="models/filament_unet_ridge.pt"):
    images, masks = _load_paired(tif_files)
    if images is None: return
    print(f"\n{'─'*50}")
    print(f"[2-ch] Training on {len(images)} frames  (raw + ridge filter)")
    device = best_device()
    model = TinyUNet2ch().to(device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params on {device}")
    _run_training(model, SegDataset2ch(images, masks), save_path, device)

# ─── UI ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_boxer.py <video1.tif> [video2.tif ...]"); sys.exit(1)

    tif_files = sys.argv[1:]
    all_frames = []
    for fp in tif_files:
        print(f"Loading {fp}...")
        img = tifffile.imread(fp).astype(np.float32)
        for i in range(img.shape[0]):
            mn, mx = img[i].min(), img[i].max()
            norm = (img[i] - mn) / (mx - mn) if mx > mn else np.zeros_like(img[i])
            all_frames.append((fp, i, norm))

    total = len(all_frames)
    print(f"Loaded {total} frames. Saved masks: {count_saved()}")

    fp0, fi0, frame0 = all_frames[0]
    current_idx = [0]
    current_mask = [load_mask(fp0, fi0, frame0.shape)]
    current_box  = [None]  # (x0, y0, x1, y1) in pixel coords
    thresh = [0.5]

    # ── Figure ──
    fig, (ax_raw, ax_ann) = plt.subplots(1, 2, figsize=(13, 7))
    plt.subplots_adjust(bottom=0.28)
    fig.suptitle("Filament Boxer — Draw boxes around filaments", fontsize=14, fontweight='bold')
    
    # Disable matplotlib's built-in key shortcuts that conflict with ours
    matplotlib.rcParams['keymap.save']     = []  # 's' -> our save, not OS dialog
    matplotlib.rcParams['keymap.quit']     = []  # 'q' safe
    matplotlib.rcParams['keymap.back']     = []  # left arrow -> our navigation
    matplotlib.rcParams['keymap.forward']  = []  # right arrow -> our navigation

    def composite(frame, mask, box=None):
        rgb = np.stack([frame]*3, axis=-1)
        rgba = np.concatenate([rgb, np.ones((*frame.shape, 1))], axis=-1).astype(np.float32)
        # Saved mask → bright green
        m = mask > 0.5
        rgba[m] = [0, 1, 0, 1]
        # Pending box → yellow tint
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
            region = frame[y0:y1, x0:x1]
            mask_region = region > thresh[0]
            sub = rgba[y0:y1, x0:x1]
            sub[mask_region]  = [1.0, 1.0, 0.0, 1.0]  # yellow = above threshold
            sub[~mask_region] = np.clip(sub[~mask_region] * 0.6, 0, 1)  # dim the rest
        return np.clip(rgba, 0, 1)

    raw_img = ax_raw.imshow(frame0, cmap='gray', vmin=0, vmax=1)
    ax_raw.axis('off'); ax_raw.set_title("Raw Frame")

    ann_img = ax_ann.imshow(composite(frame0, current_mask[0]))
    ax_ann.axis('off')
    ann_title = ax_ann.set_title("Draw box over filament  (drag here)", fontsize=11)

    # Info bar
    info = fig.text(0.5, 0.20, f"Saved: {count_saved()}/{total} | Frame [1/{total}]",
                    ha='center', fontsize=11, fontweight='bold')

    # Sliders
    ax_sl  = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_thr = plt.axes([0.15, 0.10, 0.7, 0.03])
    sl_frame = Slider(ax_sl,  'Frame',         0, total-1, valinit=0, valstep=1,   initcolor='none')
    sl_thr   = Slider(ax_thr, 'Pixel Thresh',  0.0, 1.0,  valinit=0.5, initcolor='none')

    # Buttons
    ax_save  = plt.axes([0.05, 0.02, 0.14, 0.06])
    ax_clear = plt.axes([0.22, 0.02, 0.14, 0.06])
    ax_train = plt.axes([0.39, 0.02, 0.27, 0.06])
    ax_trainr = plt.axes([0.69, 0.02, 0.27, 0.06])
    btn_save   = Button(ax_save,   '[S] Save Box',    color='#2d5f2d', hovercolor='#3a7a3a')
    btn_clear  = Button(ax_clear,  '[C] Clear',        color='#5f2d2d', hovercolor='#7a3a3a')
    btn_train  = Button(ax_train,  '[T] Train (raw)',  color='#2d2d5f', hovercolor='#3a3a7a')
    btn_trainr = Button(ax_trainr, '[R] Train +Ridge', color='#4d2d5f', hovercolor='#623a7a')
    for b in [btn_save, btn_clear, btn_train, btn_trainr]:
        b.label.set_color('white'); b.label.set_fontweight('bold')

    _upd = [False]

    def redraw():
        fp, fi, frame = all_frames[current_idx[0]]
        ann_img.set_data(composite(frame, current_mask[0], current_box[0]))
        fig.canvas.draw_idle()

    def refresh():
        if _upd[0]: return
        _upd[0] = True
        idx = current_idx[0]
        fp, fi, frame = all_frames[idx]
        raw_img.set_data(frame)
        current_mask[0] = load_mask(fp, fi, frame.shape)
        current_box[0]  = None
        ann_img.set_data(composite(frame, current_mask[0]))
        ann_title.set_text(f"Draw box over filament (drag here)")
        info.set_text(f"Saved: {count_saved()}/{total} | Frame [{idx+1}/{total}]")
        sl_frame.set_val(idx)
        fig.canvas.draw_idle()
        _upd[0] = False

    # RectangleSelector on the annotated panel
    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1): return
        # Convert display coords to pixel coords (image is 128x128)
        current_box[0] = (x0, y0, x1, y1)
        fp, fi, frame = all_frames[current_idx[0]]
        ix0, ix1 = sorted([int(x0), int(x1)])
        iy0, iy1 = sorted([int(y0), int(y1)])
        ix0 = max(0, ix0); ix1 = min(frame.shape[1], ix1)
        iy0 = max(0, iy0); iy1 = min(frame.shape[0], iy1)
        n_pixels = np.sum(frame[iy0:iy1, ix0:ix1] > thresh[0])
        ann_title.set_text(f"Box drawn! {n_pixels} bright pixels (thresh={thresh[0]:.2f})  — press S to save")
        redraw()

    rs = RectangleSelector(ax_ann, on_select, useblit=False, interactive=True,
                           props=dict(facecolor='yellow', edgecolor='yellow', alpha=0.2, fill=True))

    def do_save():
        if current_box[0] is None:
            ann_title.set_text("No box drawn yet! Drag to draw a box first.")
            fig.canvas.draw_idle(); return
        fp, fi, frame = all_frames[current_idx[0]]
        x0, y0, x1, y1 = current_box[0]
        ix0, ix1 = sorted([int(x0), int(x1)])
        iy0, iy1 = sorted([int(y0), int(y1)])
        ix0 = max(0, ix0); ix1 = min(frame.shape[1], ix1)
        iy0 = max(0, iy0); iy1 = min(frame.shape[0], iy1)
        # Auto-threshold: pixels inside box && above thresh -> mask
        mask = np.zeros(frame.shape, dtype=np.float32)
        region = frame[iy0:iy1, ix0:ix1]
        mask[iy0:iy1, ix0:ix1] = (region > thresh[0]).astype(np.float32)
        save_mask(fp, fi, mask)
        current_mask[0] = mask
        current_box[0]  = None
        n = int(mask.sum())
        ann_title.set_text(f"Saved! {n} px masked. Navigate to next frame.")
        info.set_text(f"Saved: {count_saved()}/{total} | Frame [{current_idx[0]+1}/{total}]")
        redraw()

    def on_save(e):  do_save()
    def on_clear(e):
        current_box[0] = None; current_mask[0] = np.zeros(all_frames[current_idx[0]][2].shape, dtype=np.float32)
        ann_title.set_text("Cleared. Draw a new box.")
        redraw()
    def on_train(e):
        print("\nTraining 1-ch U-Net (raw only)...")
        train_unet(tif_files)
        print("Done! Model: models/filament_unet.pt")
    def on_train_ridge(e):
        print("\nTraining 2-ch U-Net (raw + ridge filter)...")
        train_unet_ridge(tif_files)
        print("Done! Model: models/filament_unet_ridge.pt")
    def on_frame(v):  current_idx[0] = int(v); refresh()
    def on_thresh(v): thresh[0] = v; redraw()
    def on_key(e):
        if   e.key == 's':     do_save()
        elif e.key == 'c':     on_clear(None)
        elif e.key == 'r':     on_train_ridge(None)
        elif e.key == 't':     on_train(None)
        elif e.key == 'right' and current_idx[0] < total-1: current_idx[0] += 1; refresh()
        elif e.key == 'left'  and current_idx[0] > 0:       current_idx[0] -= 1; refresh()

    btn_save.on_clicked(on_save); btn_clear.on_clicked(on_clear)
    btn_train.on_clicked(on_train); btn_trainr.on_clicked(on_train_ridge)
    sl_frame.on_changed(on_frame); sl_thr.on_changed(on_thresh)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n" + "="*55)
    print("  BOXER UI READY")
    print("  Click & drag on RIGHT panel to draw a box")
    print("  Keys: S=Save, C=Clear, T=Train(raw), R=Train+Ridge, <->=Navigate")
    print("="*55 + "\n")
    plt.show()

if __name__ == "__main__":
    main()
