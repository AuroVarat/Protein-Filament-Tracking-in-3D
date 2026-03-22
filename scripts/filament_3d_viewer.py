#!/usr/bin/env python3
"""
Filament 3D Viewer — Tri-planar Annotation Tool for 3D Z-stacks

Loads TIFF volumes (with time and z-stack) and provides three orthogonal
views (XY, XZ, YZ) for interactive 3D manual masking.

Usage:
    python filament_3d_viewer.py tifs3d/volume1.tif [tifs3d/volume2.tif ...]

Controls:
    Left-click + drag = Paint 3D sphere mask (green) on active slice
    Right-click + drag= Erase
    S                 = Save 3D mask for current timepoint
    C                 = Clear 3D mask for current timepoint
    ← →               = Navigate timepoints
"""

import tifffile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import glob
import os
import sys

MASK_DIR = "models/masks3d"

def load_data(filepath):
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    # Standardise to (T, Z, H, W)
    if img.ndim == 3:
        img = img[np.newaxis, ...]  # (1, Z, H, W)
    elif img.ndim >= 4:
        if img.ndim == 5:
            # Assume (T, Z, C, H, W), take second channel
            img = img[:, :, 1, :, :]
    
    T, Z, H, W = img.shape
    # Normalize per 2D slice
    norm = np.zeros_like(img)
    for t in range(T):
        mn, mx = img[t].min(), img[t].max()
        if mx > mn:
            norm[t] = (img[t] - mn) / (mx - mn)
    return norm

def mask_path(filepath, t_idx):
    os.makedirs(MASK_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(MASK_DIR, f"{base}_t{t_idx:04d}.npy")

def load_mask(filepath, t_idx, shape_3d):
    p = mask_path(filepath, t_idx)
    if os.path.exists(p):
        return np.load(p)
    return np.zeros(shape_3d, dtype=np.float32)

def save_mask(filepath, t_idx, mask_3d):
    np.save(mask_path(filepath, t_idx), mask_3d)

def count_saved():
    os.makedirs(MASK_DIR, exist_ok=True)
    return sum(1 for p in glob.glob(os.path.join(MASK_DIR, "*.npy")) if np.load(p).max() > 0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_3d_viewer.py <volume1.tif> [volume2.tif ...]")
        sys.exit(1)

    tif_files = sys.argv[1:]
    all_volumes = []  # List of (filepath, t_idx, volume_3d)
    
    for fp in tif_files:
        norm_data = load_data(fp)
        for t in range(norm_data.shape[0]):
            all_volumes.append((fp, t, norm_data[t]))

    total = len(all_volumes)
    print(f"Loaded {total} timepoints. Saved 3D masks: {count_saved()}")

    current_idx = [0]
    fp0, t0, vol0 = all_volumes[0]
    Z, H, W = vol0.shape
    current_mask = [load_mask(fp0, t0, vol0.shape)]
    
    # State variables for sliders
    cz = [Z // 2]
    cy = [H // 2]
    cx = [W // 2]
    brush_rad = [3.0]
    is_painting = [False]
    is_erasing = [False]

    # Disable matplotlib shortcuts
    matplotlib.rcParams['keymap.save'] = []
    matplotlib.rcParams['keymap.quit'] = []
    matplotlib.rcParams['keymap.back'] = []
    matplotlib.rcParams['keymap.forward'] = []

    # UI Setup
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Filament 3D Boxer — Tri-planar annotation", fontsize=14, fontweight='bold')
    
    # 3 Axes for orthogonal views
    ax_xy = fig.add_axes([0.05, 0.4, 0.25, 0.45])
    ax_xz = fig.add_axes([0.35, 0.4, 0.25, 0.45])
    ax_yz = fig.add_axes([0.65, 0.4, 0.25, 0.45])

    ax_xy.set_title("XY Plane (Top-down)")
    ax_xz.set_title("XZ Plane (Side)")
    ax_yz.set_title("YZ Plane (Front)")
    for ax in (ax_xy, ax_xz, ax_yz):
        ax.axis('off')

    def make_rgba(img_slice, mask_slice):
        # img_slice.shape = (A, B)
        rgb = np.stack([img_slice]*3, axis=-1)
        rgba = np.concatenate([rgb, np.ones((*img_slice.shape, 1))], axis=-1).astype(np.float32)
        m = mask_slice > 0.5
        rgba[m] = [0, 1, 0, 1]
        return np.clip(rgba, 0, 1)

    img_xy = ax_xy.imshow(np.zeros((H, W, 4)))
    img_xz = ax_xz.imshow(np.zeros((Z, W, 4)))
    img_yz = ax_yz.imshow(np.zeros((Z, H, 4)))
    
    info_text = fig.text(0.5, 0.28, "", ha='center', fontsize=12, fontweight='bold')

    # Controls
    ax_sl_t = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_sl_z = plt.axes([0.15, 0.18, 0.7, 0.03])
    ax_sl_y = plt.axes([0.15, 0.14, 0.7, 0.03])
    ax_sl_x = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_sl_r = plt.axes([0.15, 0.06, 0.7, 0.03])

    sl_t = Slider(ax_sl_t, 'Timepoint (T)', 0, total-1, valinit=0, valstep=1, initcolor='none')
    sl_z = Slider(ax_sl_z, 'Z slice (XY view)', 0, Z-1, valinit=cz[0], valstep=1, initcolor='none')
    sl_y = Slider(ax_sl_y, 'Y slice (XZ view)', 0, H-1, valinit=cy[0], valstep=1, initcolor='none')
    sl_x = Slider(ax_sl_x, 'X slice (YZ view)', 0, W-1, valinit=cx[0], valstep=1, initcolor='none')
    sl_r = Slider(ax_sl_r, 'Brush Radius', 1.0, 15.0, valinit=brush_rad[0], initcolor='none')

    ax_save = plt.axes([0.3, 0.01, 0.15, 0.04])
    ax_clear = plt.axes([0.5, 0.01, 0.15, 0.04])
    btn_save = Button(ax_save, '[S] Save 3D Mask', color='#2d5f2d', hovercolor='#3a7a3a')
    btn_clear = Button(ax_clear, '[C] Clear 3D Mask', color='#5f2d2d', hovercolor='#7a3a3a')
    btn_save.label.set_color('white'); btn_save.label.set_fontweight('bold')
    btn_clear.label.set_color('white'); btn_clear.label.set_fontweight('bold')

    def draw_slices():
        idx = current_idx[0]
        _, _, vol = all_volumes[idx]
        m3 = current_mask[0]
        
        # XY: index is Z, shape is (H, W)
        img_xy.set_data(make_rgba(vol[cz[0], :, :], m3[cz[0], :, :]))
        # XZ: index is Y, shape is (Z, W)
        img_xz.set_data(make_rgba(vol[:, cy[0], :], m3[:, cy[0], :]))
        # YZ: index is X, shape is (Z, H)
        img_yz.set_data(make_rgba(vol[:, :, cx[0]], m3[:, :, cx[0]]))
        
        n_px = int(m3.sum())
        info_text.set_text(f"Timepoint [{idx+1}/{total}] | Mask size: {n_px} px | Total saved: {count_saved()}/{total}")
        fig.canvas.draw_idle()

    def refresh_volume():
        idx = current_idx[0]
        fp, t, vol = all_volumes[idx]
        current_mask[0] = load_mask(fp, t, vol.shape)
        sl_t.set_val(idx)
        # update slider max ranges if volume shapes differ
        Z, H, W = vol.shape
        sl_z.valmax = max(1, Z-1); sl_z.ax.set_xlim(sl_z.valmin, sl_z.valmax)
        sl_y.valmax = max(1, H-1); sl_y.ax.set_xlim(sl_y.valmin, sl_y.valmax)
        sl_x.valmax = max(1, W-1); sl_x.ax.set_xlim(sl_x.valmin, sl_x.valmax)
        cz[0] = min(cz[0], Z-1)
        cy[0] = min(cy[0], H-1)
        cx[0] = min(cx[0], W-1)
        sl_z.set_val(cz[0]); sl_y.set_val(cy[0]); sl_x.set_val(cx[0])
        draw_slices()

    def do_save():
        idx = current_idx[0]
        fp, t, _ = all_volumes[idx]
        save_mask(fp, t, current_mask[0])
        print(f"Saved 3D mask for {os.path.basename(fp)} timepoint {t}")
        draw_slices()

    def do_clear():
        current_mask[0].fill(0)
        draw_slices()

    btn_save.on_clicked(lambda e: do_save())
    btn_clear.on_clicked(lambda e: do_clear())

    def update_z(v): cz[0] = int(v); draw_slices()
    def update_y(v): cy[0] = int(v); draw_slices()
    def update_x(v): cx[0] = int(v); draw_slices()
    def update_t(v): current_idx[0] = int(v); refresh_volume()
    def update_r(v): brush_rad[0] = v

    sl_z.on_changed(update_z)
    sl_y.on_changed(update_y)
    sl_x.on_changed(update_x)
    sl_t.on_changed(update_t)
    sl_r.on_changed(update_r)

    def paint_3d_sphere(zc, yc, xc, erase=False):
        _, _, vol = all_volumes[current_idx[0]]
        Z, H, W = vol.shape
        r = brush_rad[0]
        
        # Bounding box for the sphere to optimize numpy assignment
        z0, z1 = max(0, int(zc-r)), min(Z, int(zc+r+1))
        y0, y1 = max(0, int(yc-r)), min(H, int(yc+r+1))
        x0, x1 = max(0, int(xc-r)), min(W, int(xc+r+1))
        
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist_sq = (zz - zc)**2 + (yy - yc)**2 + (xx - xc)**2
        sphere_mask = dist_sq <= r**2
        
        if erase:
            current_mask[0][z0:z1, y0:y1, x0:x1][sphere_mask] = 0
        else:
            current_mask[0][z0:z1, y0:y1, x0:x1][sphere_mask] = 1

    def handle_mouse(event, action="paint"):
        if event.inaxes not in (ax_xy, ax_xz, ax_yz): return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        erase = is_erasing[0]

        if event.inaxes == ax_xy:
            # Displays (W, H), Z is fixed
            paint_3d_sphere(cz[0], y, x, erase)
        elif event.inaxes == ax_xz:
            # Displays (W, Z), Y is fixed
            paint_3d_sphere(y, cy[0], x, erase)
        elif event.inaxes == ax_yz:
            # Displays (H, Z), X is fixed
            paint_3d_sphere(y, x, cx[0], erase)
        
        draw_slices()

    def on_press(event):
        if event.button == 1:
            is_painting[0] = True; is_erasing[0] = False; handle_mouse(event)
        elif event.button == 3:
            is_painting[0] = True; is_erasing[0] = True; handle_mouse(event)

    def on_release(event):
        is_painting[0] = False
        is_erasing[0] = False

    def on_motion(event):
        if is_painting[0]:
            handle_mouse(event)

    def on_key(event):
        if event.key == 's': do_save()
        elif event.key == 'c': do_clear()
        elif event.key == 'right' and current_idx[0] < total - 1:
            sl_t.set_val(current_idx[0] + 1)
        elif event.key == 'left' and current_idx[0] > 0:
            sl_t.set_val(current_idx[0] - 1)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)

    refresh_volume()
    plt.show()

if __name__ == "__main__":
    main()
