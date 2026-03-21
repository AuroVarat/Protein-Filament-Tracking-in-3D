#!/usr/bin/env python3
"""
Filament Painter — Brush-Paint Segmentation Masks + U-Net Training

Usage:
    python filament_painter.py tifs/video1.tif [tifs/video2.tif ...]

Controls:
    Left-click + drag = Paint mask (green)
    E                 = Toggle eraser mode
    S                 = Save current mask
    C                 = Clear current mask
    ← →               = Navigate frames
    Brush Size slider = Adjust brush radius
"""

import tifffile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import best_device
from scipy import ndimage
import os
import sys
import glob

# ─── U-Net Architecture ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet(nn.Module):
    """Small U-Net for 128x128 grayscale → binary mask."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)   # 64 from skip + 64 from up
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        
        # Output
        self.out_conv = nn.Conv2d(16, 1, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # 128x128
        e2 = self.enc2(self.pool(e1))  # 64x64
        e3 = self.enc3(self.pool(e2))  # 32x32
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # 16x16
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # 32x32
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 64x64
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 128x128
        
        return self.out_conv(d1)  # 128x128, raw logits

# ─── Augmented Segmentation Dataset ─────────────────────────────────────────

class SegmentationDataset(Dataset):
    """Loads paired (image, mask) and applies synchronized augmentations."""
    def __init__(self, images, masks, augment_factor=10):
        self.images = images
        self.masks = masks
        self.augment_factor = augment_factor
    
    def __len__(self):
        return len(self.images) * self.augment_factor
    
    def __getitem__(self, idx):
        real_idx = idx // self.augment_factor
        img = self.images[real_idx].copy()
        mask = self.masks[real_idx].copy()
        
        if idx % self.augment_factor != 0:
            # Random noise (image only)
            noise_sigma = np.random.uniform(0.02, 0.08)
            img = img + np.random.randn(*img.shape).astype(np.float32) * noise_sigma
            
            # Random horizontal flip (both)
            if np.random.rand() > 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            
            # Random vertical flip (both)
            if np.random.rand() > 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()
            
            # Random brightness (image only)
            brightness = np.random.uniform(0.8, 1.2)
            img = img * brightness
            
            # Random rotation (both, same angle)
            angle = np.random.uniform(-15, 15)
            img = ndimage.rotate(img, angle, reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, reshape=False, order=0)  # nearest for mask
            
            img = np.clip(img, 0, 1)
            mask = (mask > 0.5).astype(np.float32)
        
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t

# ─── Mask Manager ────────────────────────────────────────────────────────────

MASK_DIR = "models/masks"

def mask_path(filepath, frame_idx):
    os.makedirs(MASK_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(MASK_DIR, f"{basename}_{frame_idx:04d}.npy")

def save_mask(filepath, frame_idx, mask):
    np.save(mask_path(filepath, frame_idx), mask)

def load_mask(filepath, frame_idx, shape):
    p = mask_path(filepath, frame_idx)
    if os.path.exists(p):
        return np.load(p)
    return np.zeros(shape, dtype=np.float32)

def count_masks():
    os.makedirs(MASK_DIR, exist_ok=True)
    return len(glob.glob(os.path.join(MASK_DIR, "*.npy")))

def load_all_masks():
    """Returns list of (filepath_pattern, frame_idx, mask_array) for all saved masks."""
    os.makedirs(MASK_DIR, exist_ok=True)
    results = []
    for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.npy"))):
        mask = np.load(p)
        if mask.max() > 0:  # Only include masks that actually have painted pixels
            results.append((p, mask))
    return results

# ─── Training Function ──────────────────────────────────────────────────────

def train_unet(tif_files, model_save_path="models/filament_unet.pt"):
    """Train U-Net on all painted masks."""
    mask_files = load_all_masks()
    if len(mask_files) < 5:
        print(f"⚠ Only {len(mask_files)} painted masks with content. Need at least 5.")
        return None
    
    # Load TIF data to pair with masks
    tif_cache = {}
    for filepath in tif_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        img = tifffile.imread(filepath).astype(np.float32)
        # Per-frame normalize
        for i in range(img.shape[0]):
            f_min, f_max = img[i].min(), img[i].max()
            if f_max > f_min:
                img[i] = (img[i] - f_min) / (f_max - f_min)
            else:
                img[i] = 0.0
        tif_cache[basename] = img
    
    # Pair masks with their source frames
    images = []
    masks = []
    
    for mask_file_path, mask in mask_files:
        # Parse filename: {basename}_{frame_idx:04d}.npy
        fname = os.path.splitext(os.path.basename(mask_file_path))[0]
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        basename, frame_str = parts
        frame_idx = int(frame_str)
        
        if basename in tif_cache and frame_idx < tif_cache[basename].shape[0]:
            images.append(tif_cache[basename][frame_idx])
            masks.append(mask)
    
    if len(images) < 5:
        print(f"⚠ Only matched {len(images)} image-mask pairs. Need at least 5.")
        return None
    
    print(f"\n{'─'*50}")
    print(f"Training U-Net on {len(images)} painted masks")
    print(f"  Augmented dataset: {len(images) * 10} samples")
    
    dataset = SegmentationDataset(images, masks, augment_factor=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    device = best_device()
    model = TinyUNet().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters on {device}")
    
    # Dice loss + BCE for better segmentation
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for 30 epochs
    model.train()
    for epoch in range(30):
        total_loss = 0
        total_dice = 0
        n_batches = 0
        
        for batch_img, batch_mask in dataloader:
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            output = model(batch_img)
            loss = criterion(output, batch_mask)
            
            # Dice score for monitoring
            with torch.no_grad():
                pred = torch.sigmoid(output) > 0.5
                intersection = (pred * batch_mask).sum()
                dice = (2 * intersection) / (pred.sum() + batch_mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/30  Loss: {avg_loss:.4f}  Dice: {avg_dice:.3f}")
    
    # Save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"  ✓ Model saved to {model_save_path}")
    print(f"  ✓ Final Dice score: {avg_dice:.3f}")
    print(f"{'─'*50}\n")
    return model

# ─── Interactive Painting UI ────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_painter.py <video1.tif> [video2.tif ...]")
        sys.exit(1)
    
    tif_files = sys.argv[1:]
    
    # Load all videos (per-frame normalized)
    all_frames = []  # (filepath, frame_idx, frame_data)
    for filepath in tif_files:
        print(f"Loading {filepath}...")
        img = tifffile.imread(filepath).astype(np.float32)
        for i in range(img.shape[0]):
            f_min, f_max = img[i].min(), img[i].max()
            if f_max > f_min:
                norm = (img[i] - f_min) / (f_max - f_min)
            else:
                norm = np.zeros_like(img[i])
            all_frames.append((filepath, i, norm))
    
    total_frames = len(all_frames)
    print(f"Loaded {total_frames} frames from {len(tif_files)} video(s)")
    print(f"Existing painted masks: {count_masks()}")
    
    current_idx = [0]
    brush_radius = [3]
    erase_mode = [False]
    painting = [False]
    
    # Current mask for the displayed frame
    fp0, fi0, frame0 = all_frames[0]
    current_mask = [load_mask(fp0, fi0, frame0.shape)]
    
    # ── Build UI ──
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.32)
    fig.suptitle("Filament Painter — Brush Segmentation Masks", fontsize=16, fontweight='bold')
    
    # Disable matplotlib's built-in key shortcuts that conflict with ours
    matplotlib.rcParams['keymap.save']     = []  # 's' -> our save
    matplotlib.rcParams['keymap.quit']     = []  
    matplotlib.rcParams['keymap.back']     = []  # left arrow -> our navigation
    matplotlib.rcParams['keymap.forward']  = []  # right arrow -> our navigation
    
    # Left: Raw image
    img_display = ax_img.imshow(frame0, cmap='gray', vmin=0, vmax=1)
    ax_img.axis('off')
    ax_img.set_title("Raw Frame")
    
    def make_composite(frame, mask):
        """Composites a grayscale frame with green mask into RGBA array."""
        # Grayscale → RGB
        rgb = np.stack([frame, frame, frame], axis=-1)
        # Green overlay where mask is painted
        rgba = np.concatenate([rgb, np.ones((*frame.shape, 1), dtype=np.float32)], axis=-1)
        m = mask > 0.5
        rgba[m, 0] = 0.2   # R
        rgba[m, 1] = 1.0   # G
        rgba[m, 2] = 0.2   # B
        rgba[m, 3] = 1.0   # A
        return np.clip(rgba, 0, 1)
    
    # Right: Composite view (single imshow)
    composite0 = make_composite(frame0, current_mask[0])
    composite_display = ax_mask.imshow(composite0)
    ax_mask.axis('off')
    mask_title = ax_mask.set_title("Mask Overlay — PAINT mode", fontsize=12, color='#44ff44')
    
    # Status
    info_text = fig.text(0.5, 0.24, f"Masks painted: {count_masks()} | Frame [{current_idx[0]+1}/{total_frames}]",
                         ha='center', fontsize=11, fontweight='bold')
    
    # Controls
    ax_slider = plt.axes([0.2, 0.19, 0.6, 0.03])
    ax_brush  = plt.axes([0.2, 0.14, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valstep=1, initcolor='none')
    brush_slider = Slider(ax_brush, 'Brush Size', 1, 15, valinit=3, valstep=1, initcolor='none')
    
    ax_save  = plt.axes([0.10, 0.05, 0.15, 0.06])
    ax_clear = plt.axes([0.28, 0.05, 0.15, 0.06])
    ax_erase = plt.axes([0.46, 0.05, 0.15, 0.06])
    ax_train = plt.axes([0.64, 0.05, 0.20, 0.06])
    
    btn_save  = Button(ax_save,  '[S] Save', color='#2d5f2d', hovercolor='#3a7a3a')
    btn_clear = Button(ax_clear, '[C] Clear', color='#5f2d2d', hovercolor='#7a3a3a')
    btn_erase = Button(ax_erase, '[E] Paint Mode', color='#2d5f5f', hovercolor='#3a7a7a')
    btn_train = Button(ax_train, '[T] Train U-Net', color='#2d2d5f', hovercolor='#3a3a7a')
    
    for btn in [btn_save, btn_clear, btn_erase, btn_train]:
        btn.label.set_color('white')
        btn.label.set_fontweight('bold')
    
    _updating = [False]
    
    def update_mask_display(force=False):
        """Refresh the composite display."""
        fp, fi, frame = all_frames[current_idx[0]]
        composite = make_composite(frame, current_mask[0])
        composite_display.set_data(composite)
        if force:
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            fig.canvas.draw_idle()
    
    def refresh_display():
        if _updating[0]:
            return
        _updating[0] = True
        
        idx = current_idx[0]
        fp, fi, frame = all_frames[idx]
        
        img_display.set_data(frame)
        
        current_mask[0] = load_mask(fp, fi, frame.shape)
        composite = make_composite(frame, current_mask[0])
        composite_display.set_data(composite)
        
        mode_str = "ERASE mode" if erase_mode[0] else "PAINT mode"
        mode_color = '#ff4444' if erase_mode[0] else '#44ff44'
        mask_title.set_text(f"Mask Overlay — {mode_str}")
        mask_title.set_color(mode_color)
        
        info_text.set_text(f"Masks painted: {count_masks()} | Frame [{idx+1}/{total_frames}]")
        slider.set_val(idx)
        fig.canvas.draw_idle()
        
        _updating[0] = False
    
    def paint_at(x, y):
        """Paint or erase a circle at (x, y) on the current mask."""
        r = brush_radius[0]
        h, w = current_mask[0].shape
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        circle = xx**2 + yy**2 <= r**2
        
        y_start = max(0, int(y) - r)
        y_end = min(h, int(y) + r + 1)
        x_start = max(0, int(x) - r)
        x_end = min(w, int(x) + r + 1)
        
        cy_start = max(0, r - int(y))
        cy_end = cy_start + (y_end - y_start)
        cx_start = max(0, r - int(x))
        cx_end = cx_start + (x_end - x_start)
        
        if erase_mode[0]:
            current_mask[0][y_start:y_end, x_start:x_end][circle[cy_start:cy_end, cx_start:cx_end]] = 0.0
        else:
            current_mask[0][y_start:y_end, x_start:x_end][circle[cy_start:cy_end, cx_start:cx_end]] = 1.0
        
        update_mask_display(force=True)
    
    def on_press(event):
        if event.inaxes == ax_mask and event.button == 1:
            painting[0] = True
            paint_at(event.xdata, event.ydata)
    
    def on_release(event):
        painting[0] = False
    
    def on_motion(event):
        if painting[0] and event.inaxes == ax_mask and event.xdata is not None:
            paint_at(event.xdata, event.ydata)
    
    def do_save():
        fp, fi, _ = all_frames[current_idx[0]]
        save_mask(fp, fi, current_mask[0])
        info_text.set_text(f"Masks painted: {count_masks()} | Frame [{current_idx[0]+1}/{total_frames}] ✓ SAVED")
        fig.canvas.draw_idle()
    
    def on_save(event):
        do_save()
    
    def on_clear(event):
        fp, fi, frame = all_frames[current_idx[0]]
        current_mask[0] = np.zeros(frame.shape, dtype=np.float32)
        update_mask_display()
    
    def on_erase(event):
        erase_mode[0] = not erase_mode[0]
        mode_str = "ERASE mode" if erase_mode[0] else "PAINT mode"
        mode_color = '#ff4444' if erase_mode[0] else '#44ff44'
        btn_erase.label.set_text('[E] Erase Mode' if erase_mode[0] else '[E] Paint Mode')
        mask_title.set_text(f"Mask Overlay — {mode_str}")
        mask_title.set_color(mode_color)
        fig.canvas.draw_idle()
    
    def on_train(event):
        print("\n⚡ Starting U-Net training...")
        train_unet(tif_files)
        print("Training complete!")
    
    def on_slider(val):
        current_idx[0] = int(val)
        refresh_display()
    
    def on_brush(val):
        brush_radius[0] = int(val)
    
    def on_key(event):
        if event.key == 's':
            do_save()
        elif event.key == 'c':
            on_clear(None)
        elif event.key == 'e':
            on_erase(None)
        elif event.key == 'right':
            if current_idx[0] < total_frames - 1:
                current_idx[0] += 1
                refresh_display()
        elif event.key == 'left':
            if current_idx[0] > 0:
                current_idx[0] -= 1
                refresh_display()
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)
    btn_save.on_clicked(on_save)
    btn_clear.on_clicked(on_clear)
    btn_erase.on_clicked(on_erase)
    btn_train.on_clicked(on_train)
    slider.on_changed(on_slider)
    brush_slider.on_changed(on_brush)
    
    print("\n" + "="*55)
    print("  PAINTING UI READY")
    print("  Click & drag on RIGHT panel to paint masks")
    print("  Keys: S=Save, C=Clear, E=Toggle Erase, ←→=Navigate")
    print("  Click 'Train U-Net' when ~20+ masks are painted")
    print("="*55 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
