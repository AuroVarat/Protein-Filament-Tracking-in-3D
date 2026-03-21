#!/usr/bin/env python3
"""
Filament 3D Segmenter — Run Trained 3D U-Net on New Volumes

Usage:
    python filament_3d_segmenter.py tifs3d/volume.tif
"""

import sys
import os
import tifffile
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from unet3d import TinyUNet3D
from utils import best_device

def load_data(filepath):
    print(f"Loading {filepath}...")
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

def make_rgba(img_slice, mask_slice):
    rgb = np.stack([img_slice]*3, axis=-1)
    rgba = np.concatenate([rgb, np.ones((*img_slice.shape, 1))], axis=-1).astype(np.float32)
    m = mask_slice > 0.5
    rgba[m] = [0, 1, 0, 1]
    return np.clip(rgba, 0, 1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_3d_segmenter.py <volume.tif>")
        sys.exit(1)

    filepath = sys.argv[1]
    model_path = "models/filament_unet3d.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: 3D model not found at '{model_path}'. Please train first.")
        sys.exit(1)

    device = best_device()
    model = TinyUNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded 3D U-Net: {model_path} ({sum(p.numel() for p in model.parameters()):,} params, {device})")

    # Load and normalize
    normd = load_data(filepath)
    T, Z, H, W = normd.shape

    # Inference loop
    print("Running 3D pixel segmentation...")
    pred_probs = np.zeros_like(normd)
    pred_masks = np.zeros_like(normd)
    
    with torch.no_grad():
        for t in range(T):
            inp = torch.from_numpy(normd[t]).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred_probs[t] = prob
            pred_masks[t] = (prob > 0.5).astype(np.float32)
            if (t + 1) % max(1, T // 10) == 0 or t == T - 1:
                print(f"  {t+1}/{T} volumes segmented")

    pxcounts = pred_masks.sum(axis=(1, 2, 3))
    fil_frames = np.where(pxcounts > 10)[0]
    print(f"\nFilament detected in {len(fil_frames)}/{T} 3D frames")
    if len(fil_frames) > 0:
        print(f"Frames: {fil_frames.tolist()}")

    # ── Viewer ──
    cz, cy, cx = Z//2, H//2, W//2
    show_overlay = [True]

    fig = plt.figure(figsize=(16, 7))
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle(f"3D U-Net Output: {os.path.basename(filepath)}", fontsize=14, fontweight='bold')

    ax_xy = fig.add_axes([0.05, 0.4, 0.25, 0.45])
    ax_xz = fig.add_axes([0.35, 0.4, 0.25, 0.45])
    ax_yz = fig.add_axes([0.65, 0.4, 0.25, 0.45])

    ax_xy.set_title("XY Plane (Top-down)")
    ax_xz.set_title("XZ Plane (Side)")
    ax_yz.set_title("YZ Plane (Front)")
    for ax in (ax_xy, ax_xz, ax_yz):
        ax.axis('off')

    img_xy = ax_xy.imshow(np.zeros((H, W, 4)))
    img_xz = ax_xz.imshow(np.zeros((Z, W, 4)))
    img_yz = ax_yz.imshow(np.zeros((Z, H, 4)))

    info_text = fig.text(0.5, 0.32, "", ha='center', fontsize=12)

    # Sliders
    ax_sl_t = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_sl_z = plt.axes([0.15, 0.18, 0.7, 0.03])
    ax_sl_y = plt.axes([0.15, 0.14, 0.7, 0.03])
    ax_sl_x = plt.axes([0.15, 0.10, 0.7, 0.03])

    sl_t = Slider(ax_sl_t, 'Timepoint (T)', 0, T-1, valinit=0, valstep=1, initcolor='none')
    sl_z = Slider(ax_sl_z, 'Z slice (XY)', 0, Z-1, valinit=cz, valstep=1, initcolor='none')
    sl_y = Slider(ax_sl_y, 'Y slice (XZ)', 0, H-1, valinit=cy, valstep=1, initcolor='none')
    sl_x = Slider(ax_sl_x, 'X slice (YZ)', 0, W-1, valinit=cx, valstep=1, initcolor='none')

    ax_toggle = plt.axes([0.45, 0.02, 0.1, 0.05])
    btn_toggle = Button(ax_toggle, 'Toggle Overlay', hovercolor='0.8')

    def draw_slices():
        t = int(sl_t.val)
        z, y, x = int(sl_z.val), int(sl_y.val), int(sl_x.val)
        
        vol = normd[t]
        if show_overlay[0]:
            mask = pred_masks[t]
        else:
            mask = np.zeros_like(pred_masks[t])

        img_xy.set_data(make_rgba(vol[z, :, :], mask[z, :, :]))
        img_xz.set_data(make_rgba(vol[:, y, :], mask[:, y, :]))
        img_yz.set_data(make_rgba(vol[:, :, x], mask[:, :, x]))
        
        n_px = int(pxcounts[t])
        status = "FILAMENT" if n_px > 10 else "No filament"
        color = "green" if n_px > 10 else "red"
        info_text.set_text(f"Timepoint {t} — {status} ({n_px} px)")
        info_text.set_color(color)
        fig.canvas.draw_idle()

    def toggle(e):
        show_overlay[0] = not show_overlay[0]
        draw_slices()

    btn_toggle.on_clicked(toggle)

    def update(v): draw_slices()
    sl_t.on_changed(update)
    sl_z.on_changed(update)
    sl_y.on_changed(update)
    sl_x.on_changed(update)

    draw_slices()
    plt.show()

if __name__ == "__main__":
    main()
