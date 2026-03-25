#!/usr/bin/env python3
"""
Filament 3D Tracker MP4 Generator (Multi-Channel Edition)

Layout:
- Row 1: Raw Brightfield planes (Channel 0)
- Row 2: Raw Filament planes (Channel 1)
- Row 3: Tracked Filament Overlays
- Row 4: 2.5D Volumetric View
"""

import sys
import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw, ImageFont
import imageio

from unet3d import TinyUNet3D
from utils import best_device

# --- Parameters ---
PIXEL_SIZE_UM = 0.183
MIN_AREA_PX = 5        
PROB_THRESH = 0.5      

def segment_cells(bf_vol, cellpose_model):
    z_idx = bf_vol.shape[0] // 2
    ref_2d = bf_vol[z_idx]
    masks, _, _ = cellpose_model.eval(ref_2d, diameter=25, channels=[0,0])
    return masks

def z_localize(mask_3d, raw_3d):
    Z, H, W = mask_3d.shape
    localized = np.zeros_like(mask_3d)
    active_xy = np.sum(mask_3d, axis=0) > 0
    y_idx, x_idx = np.where(active_xy)
    for y, x in zip(y_idx, x_idx):
        z_candidates = np.where(mask_3d[:, y, x] > 0)[0]
        if len(z_candidates) > 0:
            best_z = z_candidates[np.argmax(raw_3d[z_candidates, y, x])]
            localized[best_z, y, x] = 1.0
    return localized

def get_vibrant_color(idx):
    if idx == 0: return np.array([0, 0, 0])
    phi = (1 + 5**0.5) / 2
    hue = (idx * phi) % 1.0
    cmap = plt.get_cmap('hsv')
    return np.array(cmap(hue)[:3])

def render_tracked_2p5d(vol, label_vol, shift_x=25, shift_y=-25):
    Z, H, W = vol.shape
    out_H, out_W = H + abs(shift_y)*(Z-1), W + abs(shift_x)*(Z-1)
    canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
    for z in range(Z):
        x_off, y_off = z * shift_x, (Z - 1 - z) * abs(shift_y) if shift_y < 0 else z * shift_y
        slice_rgb = np.stack([vol[z]]*3, axis=-1)
        z_labels = label_vol[z]
        for val in np.unique(z_labels[z_labels > 0]):
            slice_rgb[z_labels == val] = get_vibrant_color(val)
        
        alpha = np.zeros((H, W), dtype=np.float32)
        alpha[z_labels > 0] = 1.0 
        alpha[~(z_labels > 0)] = vol[z][~(z_labels > 0)] * 0.5
        alpha = np.stack([alpha]*3, axis=-1)
        target = canvas[y_off:y_off+H, x_off:x_off+W]
        canvas[y_off:y_off+H, x_off:x_off+W] = target * (1 - alpha) + slice_rgb * alpha
    return canvas

def make_tracked_frame(vol_bf, vol_fil, label_vol, cell_mask, frame_idx=0, total_frames=1):
    Z, H, W = vol_fil.shape
    
    # Normalizations
    def norm(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-6)

    # Panorama Rows
    p_bf = np.hstack([norm(vol_bf[z]) for z in range(Z)])
    p_bf_rgb = np.stack([p_bf]*3, axis=-1)
    
    p_fil = np.hstack([norm(vol_fil[z]) for z in range(Z)])
    p_fil_rgb = np.stack([p_fil]*3, axis=-1)
    
    p_overlay = p_fil_rgb.copy()
    cell_edges = ndimage.morphological_gradient(cell_mask, size=(3,3)) > 0
    for z in range(Z):
        x_s, x_e = z*W, (z+1)*W
        slice_ov = p_overlay[:, x_s:x_e]
        slice_ov[cell_edges] = [1.0, 1.0, 1.0] # White cell boundaries still useful in panorama
        z_labels = label_vol[z]
        for val in np.unique(z_labels[z_labels>0]):
            slice_ov[z_labels == val] = get_vibrant_color(val)
    
    # 2.5D View
    stack_img = render_tracked_2p5d(norm(vol_fil), label_vol)
    pano_W = p_fil_rgb.shape[1]
    
    # Adjust stack_img to fit width
    stack_row_H = stack_img.shape[0] + 40
    stack_row = np.zeros((stack_row_H, pano_W, 3), dtype=np.float32)
    # Center it
    start_x = (pano_W - stack_img.shape[1]) // 2
    stack_row[20:20+stack_img.shape[0], start_x:start_x+stack_img.shape[1]] = stack_img
    
    # Combine everything
    sep = np.zeros((10, pano_W, 3))
    frame = np.vstack([
        p_bf_rgb, sep,
        p_fil_rgb, sep,
        p_overlay, sep,
        stack_row
    ])
    
    pil_img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 15), f"Frame: {frame_idx + 1} / {total_frames} (Multi-Channel 4-Row View)", fill=(255, 255, 0))
    return np.array(pil_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input 5D TIFF volume")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    
    model_path = "models/filament_unet3d_temporal_auto.pt" if args.auto else "models/filament_unet3d_temporal.pt"
    device = best_device()
    
    model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    img = tifffile.imread(args.input).astype(np.float32)
    T, Z, C, H, W = img.shape
    vol_bf, vol_fil = img[:, :, 0, :, :], img[:, :, 1, :, :]
    
    # Global normalization for entire volume (more stable)
    norm_fil = (vol_fil - vol_fil.min()) / (vol_fil.max() - vol_fil.min() + 1e-6)

    from cellpose import models
    cellpose_model = models.CellposeModel(model_type='cyto', gpu=True)
    
    out_filament_labels = np.zeros((T, Z, H, W), dtype=np.int32)
    out_cell_masks = np.zeros((T, H, W), dtype=np.int32)
    
    print("Inference, Segmentation & Association...")
    with torch.no_grad():
        for t in tqdm(range(T)):
            # 1. Inference
            t_p, t_c, t_n = max(0, t-1), t, min(T-1, t+1)
            inp = torch.from_numpy(np.stack([norm_fil[t_p], norm_fil[t_c], norm_fil[t_n]], axis=0)).float().unsqueeze(0).to(device)
            logits = model(inp).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits[1])) 
            pred_mask = (probs > PROB_THRESH).astype(np.float32)
            
            # 2. Localize
            loc_mask = z_localize(pred_mask, vol_fil[t])
            
            # 3. Cell Segmentation
            cell_mask = segment_cells(vol_bf[t], cellpose_model)
            out_cell_masks[t] = cell_mask
            
            # 4. Association
            z_i, y_i, x_i = np.where(loc_mask > 0)
            if len(z_i) > 0:
                cids = cell_mask[y_i, x_i]
                mask = cids > 0
                out_filament_labels[t, z_i[mask], y_i[mask], x_i[mask]] = cids[mask]
                
    # Rendering
    out_path = f"results/{os.path.splitext(os.path.basename(args.input))[0]}_multi_channel.mp4"
    writer = imageio.get_writer(out_path, fps=10, macro_block_size=None)
    print("Rendering...")
    for t in tqdm(range(T)):
        writer.append_data(make_tracked_frame(vol_bf[t], vol_fil[t], out_filament_labels[t], out_cell_masks[t], t, T))
    writer.close()
    print(f"✅ Saved multi-channel tracked video to {out_path}")

if __name__ == "__main__": main()
