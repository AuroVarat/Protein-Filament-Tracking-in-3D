#!/usr/bin/env python3
"""
Filament 3D Video Generator — Render 3D Inference to MP4

Usage:
    python scripts/filament_3d_mp4.py tifs3d/volume.tif [output.mp4]
"""

import sys
import os
import tifffile
import numpy as np
import torch
import imageio
from tqdm import tqdm

from unet3d import TinyUNet3D
from utils import best_device

def load_data(filepath):
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    elif img.ndim >= 4:
        if img.ndim == 5:
            img = img[:, :, 1, :, :]  # Second channel
            
    T, Z, H, W = img.shape
    
    norm = np.zeros_like(img)
    for t in range(T):
        for z in range(Z):
            mn, mx = img[t, z].min(), img[t, z].max()
            if mx > mn:
                norm[t, z] = (img[t, z] - mn) / (mx - mn)
    return norm

def render_2p5d(vol, mask_vol, shift_x=20, shift_y=-20):
    """
    Renders 5 Z-planes diagonally as a semi-transparent 3D stack.
    """
    Z, H, W = vol.shape
    out_H = H + abs(shift_y) * (Z - 1)
    out_W = W + abs(shift_x) * (Z - 1)
    
    canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
    
    for z in range(Z):
        # Draw from bottom (z=0) to top (z=Z-1)
        x_off = z * shift_x
        # If shift_y < 0, z=0 is at the lowest screen position (highest y index)
        y_off = (Z - 1 - z) * abs(shift_y) if shift_y < 0 else z * shift_y
        
        # Base grayscale slice
        slice_rgb = np.stack([vol[z]]*3, axis=-1)
        
        # Overlay green mask
        m = mask_vol[z] > 0.5
        slice_rgb[m] = [0.0, 1.0, 0.0]  # bright green
        
        # Calculate alpha. 
        # Ensure mask is highly opaque.
        # Ensure raw signal is semi-transparent, proportional to the signal intensity.
        # This makes the black background fully transparent.
        alpha = np.where(m, 0.85, vol[z] * 0.6)
        alpha = np.stack([alpha]*3, axis=-1)
        
        # Composite onto canvas
        target = canvas[y_off:y_off+H, x_off:x_off+W]
        canvas[y_off:y_off+H, x_off:x_off+W] = target * (1 - alpha) + slice_rgb * alpha
        
    return canvas

def make_projections(vol, mask_vol):
    """
    Creates Maximum Intensity Projections (MIP) for XY, XZ, YZ planes.
    Z-axis is upscaled by 5x so it's visibly thick instead of a tiny 5-px sliver.
    """
    Z, H, W = vol.shape
    
    # XY projection (Z-proj) - Shape (H, W)
    xy_vol = np.max(vol, axis=0)
    xy_mask = np.max(mask_vol, axis=0) > 0.5
    xy_rgb = np.stack([xy_vol]*3, axis=-1)
    xy_rgb[xy_mask] = [0.0, 1.0, 0.0]
    
    # XZ projection (Y-proj) - Shape (Z, W)
    xz_vol = np.max(vol, axis=1)
    xz_mask = np.max(mask_vol, axis=1) > 0.5
    xz_rgb = np.stack([xz_vol]*3, axis=-1)
    xz_rgb[xz_mask] = [0.0, 1.0, 0.0]
    xz_rgb = np.repeat(xz_rgb, 5, axis=0)  # Stretch Z axis by 5x
    
    # YZ projection (X-proj) - Shape (Z, H)
    yz_vol = np.max(vol, axis=2)
    yz_mask = np.max(mask_vol, axis=2) > 0.5
    yz_rgb = np.stack([yz_vol]*3, axis=-1)
    yz_rgb[yz_mask] = [0.0, 1.0, 0.0]
    yz_rgb = np.repeat(yz_rgb, 5, axis=0)  # Stretch Z axis by 5x
    
    # Combine vertically with 10px padding
    pad1 = np.zeros((10, W, 3), dtype=np.float32)
    pad2 = np.zeros((10, H, 3), dtype=np.float32)
    
    col = np.vstack([xy_rgb, pad1, xz_rgb, pad2, yz_rgb])
    return col

def make_frame(vol, mask_vol):
    """
    Creates a single composite frame for the MP4:
    Left Half: 2.5D pseudo-3D stack
    Right Half: XY, XZ, YZ Projections stacked vertically
    """
    # 2.5D stack
    stack_img = render_2p5d(vol, mask_vol, shift_x=25, shift_y=-25)
    
    # Projections
    proj_col = make_projections(vol, mask_vol)
    
    h_s, w_s, _ = stack_img.shape
    h_p, w_p, _ = proj_col.shape
    
    # Add borders and padding
    H_frame = max(h_s, h_p) + 40
    W_frame = w_s + w_p + 80
    
    frame = np.zeros((H_frame, W_frame, 3), dtype=np.float32)
    
    # Left alignment for 2.5D Stack
    y_s = (H_frame - h_s) // 2
    x_s = 20
    frame[y_s:y_s+h_s, x_s:x_s+w_s] = stack_img
    
    # Right alignment for Projections
    y_p = (H_frame - h_p) // 2
    x_p = x_s + w_s + 40
    frame[y_p:y_p+h_p, x_p:x_p+w_p] = proj_col
    
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_3d_mp4.py <volume.tif> [output.mp4]")
        sys.exit(1)

    filepath = sys.argv[1]
    
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = os.path.splitext(os.path.basename(filepath))[0] + "_inference.mp4"
        
    model_path = "models/filament_unet3d.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: 3D model not found at '{model_path}'. Please train first.")
        sys.exit(1)

    device = best_device()
    model = TinyUNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded 3D U-Net")

    # Load and normalize
    normd = load_data(filepath)
    T, Z, H, W = normd.shape

    print(f"Running inference and rendering {T} frames...")
    
    writer = imageio.get_writer(out_path, fps=10, macro_block_size=None)
    
    with torch.no_grad():
        for t in tqdm(range(T)):
            # Inference
            inp = torch.from_numpy(normd[t]).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.float32)
            
            # Render frame
            frame = make_frame(normd[t], pred_mask)
            writer.append_data(frame)
            
    writer.close()
    print(f"\n✅ Saved inference video to {out_path}")

if __name__ == "__main__":
    main()
