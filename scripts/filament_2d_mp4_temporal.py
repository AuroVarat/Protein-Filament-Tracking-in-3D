#!/usr/bin/env python3
"""
Filament 2D Temporal Video Generator

Usage:
    python scripts/filament_2d_mp4_temporal.py tifs2d/video.tif [--auto] [output.mp4]
"""

import sys
import os
import tifffile
import numpy as np
import torch
import imageio
from tqdm import tqdm
import argparse

from unet2d import TinyUNet2D
from utils import best_device

def load_data(filepath):
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    # Normalize per full video sequence
    mn, mx = img.min(), img.max()
    if mx > mn:
        norm = (img - mn) / (mx - mn)
    else:
        norm = np.zeros_like(img)
    return norm

def make_frame(img, mask):
    # img, mask are (H, W)
    rgb = np.stack([img]*3, axis=-1)
    rgba = np.concatenate([rgb, np.ones((*img.shape, 1))], axis=-1).astype(np.float32)
    m = mask > 0.5
    rgba[m] = [0, 1, 0, 1] # Green mask
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input 2D TIFF volume")
    parser.add_argument("--auto", action="store_true", help="Use auto-threshold model")
    parser.add_argument("output", nargs="?", help="Output MP4 path")
    args = parser.parse_args()

    model_type = "auto" if args.auto else "default"
    model_path = f"models/filament_unet2d_temporal{'_auto' if args.auto else ''}.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    device = best_device()
    model = TinyUNet2D(in_ch=3, out_ch=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded 2D Temporal Model ({model_type})")

    normd = load_data(args.input)
    T, H, W = normd.shape

    acc_logits = np.zeros((T, H, W), dtype=np.float32)
    acc_counts = np.zeros(T, dtype=np.float32)

    print(f"Running temporal sliding window inference for {T} frames...")
    with torch.no_grad():
        for t in tqdm(range(T)):
            t_prev = max(0, t - 1)
            t_next = min(T - 1, t + 1)
            window = np.stack([normd[t_prev], normd[t], normd[t_next]], axis=0) # (3, H, W)
            inp = torch.from_numpy(window).float().unsqueeze(0).to(device) # (1, 3, H, W)
            
            logits = model(inp).squeeze(0).cpu().numpy() # (3, H, W)
            
            acc_logits[t_prev] += logits[0]
            acc_counts[t_prev] += 1
            acc_logits[t] += logits[1]
            acc_counts[t] += 1
            acc_logits[t_next] += logits[2]
            acc_counts[t_next] += 1

    avg_logits = acc_logits / np.maximum(acc_counts[:, None, None], 1)
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    pred_masks = (probs > 0.5).astype(np.float32)

    out_path = args.output if args.output else f"{os.path.splitext(os.path.basename(args.input))[0]}_2d_temporal_{model_type}.mp4"
    
    print("Rendering video...")
    writer = imageio.get_writer(out_path, fps=10, macro_block_size=None)
    for t in tqdm(range(T)):
        frame = make_frame(normd[t], pred_masks[t])
        writer.append_data(frame)
    writer.close()
    print(f"✅ Saved 2D temporal video to {out_path}")

if __name__ == "__main__":
    main()
