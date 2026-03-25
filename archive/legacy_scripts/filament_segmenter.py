#!/usr/bin/env python3
"""
Filament Segmenter — Run Trained U-Net on New Videos

Usage:
    python filament_segmenter.py tifs/video.tif              # 1-ch model (raw only)
    python filament_segmenter.py tifs/video.tif --ridge      # 2-ch model (raw+ridge)
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
from scipy import ndimage
import os
import sys
from utils import best_device

# ─── Ridge Filter ─────────────────────────────────────────────────────────────

def ridge_filter_single(img, sigma=1.5, beta=0.5, c_thresh=0.02):
    Hxx = ndimage.gaussian_filter(img, sigma, order=[2, 0])
    Hyy = ndimage.gaussian_filter(img, sigma, order=[0, 2])
    Hxy = ndimage.gaussian_filter(img, sigma, order=[1, 1])
    trace = Hxx + Hyy
    det_diff = np.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)
    l1 = 0.5 * (trace + det_diff); l2 = 0.5 * (trace - det_diff)
    lambda1 = np.where(np.abs(l1) > np.abs(l2), l1, l2)
    lambda2 = np.where(np.abs(l1) > np.abs(l2), l2, l1)
    l1s = np.where(lambda1 == 0, 1e-10, lambda1)
    Rb = np.abs(lambda2) / np.abs(l1s)
    S  = np.sqrt(lambda1**2 + lambda2**2)
    c  = max(np.max(S) / 3.0, c_thresh)
    V  = np.exp(-(Rb**2) / (2*beta**2)) * (1 - np.exp(-(S**2) / (2*c**2)))
    V[lambda1 > 0] = 0
    return V.astype(np.float32)

# ─── U-Net (supports 1 or 2 input channels) ──────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

def make_unet(in_ch):
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(in_ch, 16); self.enc2 = ConvBlock(16, 32)
            self.enc3 = ConvBlock(32, 64);    self.pool = nn.MaxPool2d(2)
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
    return UNet()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args      = sys.argv[1:]
    use_ridge = '--ridge' in args
    tif_args  = [a for a in args if not a.startswith('--')]

    if not tif_args:
        print("Usage: python filament_segmenter.py <video.tif> [--ridge]")
        sys.exit(1)

    filepath   = tif_args[0]
    model_path = "models/filament_unet_ridge.pt" if use_ridge else "models/filament_unet.pt"
    n_ch       = 2 if use_ridge else 1
    label      = "2-ch (raw+ridge)" if use_ridge else "1-ch (raw only)"

    if not os.path.exists(model_path):
        alt = "models/filament_unet.pt" if use_ridge else "models/filament_unet_ridge.pt"
        print(f"Error: No model at '{model_path}'")
        if os.path.exists(alt):
            print(f"Tip: Found '{alt}' — try {'without' if use_ridge else 'with'} --ridge")
        sys.exit(1)

    device = best_device()
    model  = make_unet(n_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {label} model: {model_path}  ({sum(p.numel() for p in model.parameters()):,} params, {device})")

    # Load + per-frame normalise
    print(f"Loading {filepath}...")
    raw   = tifffile.imread(filepath).astype(np.float32)
    nf    = raw.shape[0]
    normd = np.zeros_like(raw)
    for i in range(nf):
        mn, mx = raw[i].min(), raw[i].max()
        if mx > mn: normd[i] = (raw[i] - mn) / (mx - mn)

    ridge_maps = None
    if use_ridge:
        print("Computing ridge filter maps...")
        ridge_maps = np.zeros_like(normd)
        for i in range(nf):
            ridge_maps[i] = ridge_filter_single(normd[i])

    # Inference
    print("Running pixel segmentation...")
    pred_probs = np.zeros_like(normd)
    pred_masks = np.zeros_like(normd)
    with torch.no_grad():
        for i in range(nf):
            if use_ridge:
                inp = torch.from_numpy(np.stack([normd[i], ridge_maps[i]], axis=0)).float().unsqueeze(0).to(device)
            else:
                inp = torch.from_numpy(normd[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred_probs[i] = prob
            pred_masks[i] = (prob > 0.5).astype(np.float32)
            if (i+1) % 25 == 0: print(f"  {i+1}/{nf} frames segmented")

    pxcounts = pred_masks.sum(axis=(1, 2))
    fil_frames = np.where(pxcounts > 10)[0]
    print(f"\nFilament detected in {len(fil_frames)}/{nf} frames [{label}]")
    if len(fil_frames) > 0: print(f"Frames: {fil_frames.tolist()}")

    # ── Viewer ──
    if use_ridge:
        fig, (ax_raw, ax_ridge, ax_prob, ax_ov) = plt.subplots(1, 4, figsize=(22, 7))
    else:
        fig, (ax_raw, ax_prob, ax_ov) = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle(f"U-Net [{label}]: {os.path.basename(filepath)}", fontsize=13, fontweight='bold')

    ir = ax_raw.imshow(normd[0], cmap='gray', vmin=0, vmax=1); ax_raw.axis('off'); ax_raw.set_title("Normalized")
    if use_ridge:
        irr = ax_ridge.imshow(ridge_maps[0], cmap='inferno'); ax_ridge.axis('off'); ax_ridge.set_title("Ridge (Ch.2)")
    ip = ax_prob.imshow(pred_probs[0], cmap='hot', vmin=0, vmax=1); ax_prob.axis('off'); ax_prob.set_title("Probability")
    ib = ax_ov.imshow(normd[0], cmap='gray', vmin=0, vmax=1)
    def make_ov(t):
        rgba = np.zeros((*normd[t].shape, 4), dtype=np.float32)
        rgba[pred_masks[t] > 0.5] = [0, 1, 0, 0.6]
        return rgba
    iovl = ax_ov.imshow(make_ov(0)); ax_ov.axis('off')
    px0 = int(pxcounts[0]); h0 = px0 > 10
    ot = ax_ov.set_title(f"Frame 0 — {'FILAMENT' if h0 else 'No filament'} ({px0} px)",
                         fontsize=11, color='#44ff44' if h0 else '#ff4444')

    import matplotlib.animation as animation
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

    def update(t):
        ir.set_data(normd[t])
        if use_ridge: irr.set_data(ridge_maps[t])
        ip.set_data(pred_probs[t])
        ib.set_data(normd[t]); iovl.set_data(make_ov(t))
        px = int(pxcounts[t]); h = px > 10
        ot.set_text(f"Frame {t} — {'FILAMENT' if h else 'No filament'} ({px} px)")
        ot.set_color('#44ff44' if h else '#ff4444')
        return [ir, ip, ib, iovl] + ([irr] if use_ridge else [])

    print(f"\nGenerating and saving animation ({nf} frames) to MP4...")
    ani = animation.FuncAnimation(fig, update, frames=nf, blit=False)
    
    os.makedirs('output', exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join('output', f"{base}_{'ridge' if use_ridge else 'raw'}_segmented.mp4")
    
    ani.save(out_path, writer='ffmpeg', fps=10)
    print(f"Animation saved successfully to: {out_path}")

if __name__ == "__main__":
    main()
