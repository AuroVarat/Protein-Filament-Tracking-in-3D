#!/usr/bin/env python3
"""
Filament 3D Video Generator — Render 3D Inference to MP4

Usage:
    python scripts/filament_3d_mp4.py tifs3d/volume.tif [--auto | --ridge] [output.mp4]

Flags:
    --auto  : Uses models/filament_unet3d_auto.pt
    --ridge : Uses models/filament_unet3d_ridge.pt
    (If no flag is provided, uses models/filament_unet3d.pt)
"""

import sys
import os
import tifffile
import numpy as np
import torch
import imageio
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw, ImageFont

from unet3d import TinyUNet3D, ridge_filter_3d
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
        mn, mx = img[t].min(), img[t].max()
        if mx > mn:
            norm[t] = (img[t] - mn) / (mx - mn)
    return norm

def render_2p5d(vol, mask_vol, shift_x=20, shift_y=-20):
    """
    Renders Z-planes diagonally as a semi-transparent 3D stack.
    """
    Z, H, W = vol.shape
    out_H = H + abs(shift_y) * (Z - 1)
    out_W = W + abs(shift_x) * (Z - 1)
    
    canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
    
    for z in range(Z):
        x_off = z * shift_x
        y_off = (Z - 1 - z) * abs(shift_y) if shift_y < 0 else z * shift_y
        
        slice_rgb = np.stack([vol[z]]*3, axis=-1)
        m = mask_vol[z] > 0.5
        slice_rgb[m] = [0.0, 1.0, 0.0]  # bright green
        
        alpha = np.where(m, 0.85, vol[z] * 0.6)
        alpha = np.stack([alpha]*3, axis=-1)
        
        target = canvas[y_off:y_off+H, x_off:x_off+W]
        canvas[y_off:y_off+H, x_off:x_off+W] = target * (1 - alpha) + slice_rgb * alpha
        
    return canvas

def make_projections(vol, mask_vol, z_scale=10):
    """
    Creates an ImageJ-style Orthogonal MIP view.
    """
    Z, H, W = vol.shape
    
    xy_vol = np.max(vol, axis=0)
    xy_mask = np.max(mask_vol, axis=0) > 0.5
    xy_rgb = np.stack([xy_vol]*3, axis=-1)
    xy_rgb[xy_mask] = [0.0, 1.0, 0.0]
    
    yz_vol = np.max(vol, axis=2).T
    yz_mask = np.max(mask_vol, axis=2).T > 0.5
    yz_rgb = np.stack([yz_vol]*3, axis=-1)
    yz_rgb[yz_mask] = [0.0, 1.0, 0.0]
    yz_rgb = np.repeat(yz_rgb, z_scale // Z if z_scale > Z else 1, axis=1)
    
    xz_vol = np.max(vol, axis=1)
    xz_mask = np.max(mask_vol, axis=1) > 0.5
    xz_rgb = np.stack([xz_vol]*3, axis=-1)
    xz_rgb[xz_mask] = [0.0, 1.0, 0.0]
    xz_rgb = np.repeat(xz_rgb, z_scale // Z if z_scale > Z else 1, axis=0)

    H_yz, W_yz, _ = yz_rgb.shape
    H_xz, W_xz, _ = xz_rgb.shape
    
    canvas_H = H + H_xz + 10
    canvas_W = W + W_yz + 10
    canvas = np.zeros((canvas_H, canvas_W, 3), dtype=np.float32)
    
    canvas[0:H, 0:W] = xy_rgb
    canvas[0:H, W+10:W+10+W_yz] = yz_rgb
    canvas[H+10:H+10+H_xz, 0:W] = xz_rgb
    
    return canvas

def make_frame(vol, mask_vol, frame_idx=0, total_frames=1, detection_history=None):
    """
    Creates a single composite 3-row frame with frame counter and detection bar.
    """
    Z, H, W = vol.shape
    
    pano_raw = np.hstack([vol[z] for z in range(Z)])
    pano_raw_rgb = np.stack([pano_raw]*3, axis=-1)
    
    pano_overlay = pano_raw_rgb.copy()
    mask_pano = np.hstack([mask_vol[z] for z in range(Z)])
    pano_overlay[mask_pano > 0.5] = [0.0, 1.0, 0.0]
    
    pano_W = pano_raw_rgb.shape[1]
    
    stack_img = render_2p5d(vol, mask_vol, shift_x=25, shift_y=-25)
    ortho_grid = make_projections(vol, mask_vol, z_scale=10)
    
    h_s, w_s, _ = stack_img.shape
    h_o, w_o, _ = ortho_grid.shape
    
    row3_H = max(h_s, h_o) + 40
    row3 = np.zeros((row3_H, pano_W, 3), dtype=np.float32)
    
    y_s = (row3_H - h_s) // 2
    x_s_start = max(10, (pano_W // 2 - w_s) // 2)
    row3[y_s:y_s+h_s, x_s_start:x_s_start+w_s] = stack_img
    
    y_o = (row3_H - h_o) // 2
    x_o_start = pano_W // 2 + max(10, (pano_W // 2 - w_o) // 2)
    w_o_clip = min(w_o, pano_W - x_o_start)
    row3[y_o:y_o+h_o, x_o_start:x_o_start+w_o_clip] = ortho_grid[:, 0:w_o_clip]
    
    pad = np.zeros((10, pano_W, 3), dtype=np.float32)
    frame = np.vstack([pano_raw_rgb, pad, pano_overlay, pad, row3])
    
    cur_H, cur_W, _ = frame.shape
    
    # Add Padding at the top for UI
    header_h = 80
    new_frame = np.zeros((cur_H + header_h, cur_W, 3), dtype=np.float32)
    new_frame[header_h:, :, :] = frame
    frame = new_frame
    cur_H += header_h

    # Add Frame Number and Detection Bar using PIL
    # Convert to PIL Image
    pil_img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    
    # --- Detection Bar at Top ---
    bar_h = 24
    if detection_history is not None:
        # Draw a bar at the top
        bar_x, bar_y = 10, 10
        bar_w = cur_W - 20
        # Background
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(40, 40, 40))
        
        # Segments
        n = len(detection_history)
        seg_w = bar_w / n
        for i, detected in enumerate(detection_history):
            if detected:
                draw.rectangle([bar_x + i*seg_w, bar_y, bar_x + (i+1)*seg_w, bar_y + bar_h], fill=(0, 255, 0))
        
        # Cursor
        cursor_x = bar_x + frame_idx * seg_w
        draw.rectangle([cursor_x, bar_y - 2, cursor_x + max(2, seg_w), bar_y + bar_h + 2], outline=(255, 255, 255), width=2)

    # --- Frame Number ---
    try:
        # Try to load a font, fallback to default
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    txt = f"Frame: {frame_idx + 1} / {total_frames}"
    draw.text((cur_W - 200, 45), txt, fill=(255, 255, 0), font=font)

    # Convert back to numpy
    frame = np.array(pil_img).astype(np.float32) / 255.0

    cur_H, cur_W, _ = frame.shape
    new_H = cur_H + (cur_H % 2)
    new_W = cur_W + (cur_W % 2)
    if new_H != cur_H or new_W != cur_W:
        even_frame = np.zeros((new_H, new_W, 3), dtype=np.float32)
        even_frame[:cur_H, :cur_W, :] = frame
        frame = even_frame

    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Filament 3D Video Generator")
    parser.add_argument("input", help="Input TIFF volume")
    parser.add_argument("--auto", action="store_true", help="Use auto-threshold model")
    parser.add_argument("--ridge", action="store_true", help="Use ridge-filter model")
    parser.add_argument("output", nargs="?", help="Output MP4 path (optional)")
    
    args = parser.parse_args()

    filepath = args.input
    
    if args.ridge:
        model_type = "ridge"
        model_path = "models/filament_unet3d_ridge.pt"
        in_ch = 2
    elif args.auto:
        model_type = "auto"
        model_path = "models/filament_unet3d_auto.pt"
        in_ch = 1
    else:
        model_type = "default"
        model_path = "models/filament_unet3d.pt"
        in_ch = 1

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = f"{base}_inference_{model_type}.mp4"

    if not os.path.exists(model_path):
        print(f"Error: 3D model not found at '{model_path}'. Please train first using the corresponding script.")
        sys.exit(1)

    device = best_device()
    model = TinyUNet3D(in_ch=in_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded 3D U-Net ({model_type}) from {model_path}")

    # Load and normalize
    normd = load_data(filepath)
    T, Z, H, W = normd.shape

    print(f"Running inference and rendering {T} frames...")
    
    writer = imageio.get_writer(out_path, fps=10, macro_block_size=None)
    
    history = []
    masks = []
    
    with torch.no_grad():
        print("Phase 1: Running Inference...")
        for t in tqdm(range(T)):
            vol = normd[t]
            if in_ch == 1:
                inp = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
            else:
                ridge = ridge_filter_3d(vol)
                two_ch = np.stack([vol, ridge], axis=0)
                inp = torch.from_numpy(two_ch).float().unsqueeze(0).to(device)
                
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.float32)
            
            masks.append(pred_mask)
            history.append(pred_mask.max() > 0.5)
            
        print("Phase 2: Rendering Frames...")
        for t in tqdm(range(T)):
            frame = make_frame(normd[t], masks[t], frame_idx=t, total_frames=T, detection_history=history)
            writer.append_data(frame)
            
    writer.close()
    print(f"\n✅ Saved inference video to {out_path}")

if __name__ == "__main__":
    main()
