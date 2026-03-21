#!/usr/bin/env python3
"""
Filament Analyser — Batch quantification of filament morphology and dynamics

Runs the trained U-Net on one or more TIFF videos, extracts per-frame
morphological measurements, and exports a CSV + summary figure.

Usage:
    python filament_analyser.py tifs/video1.tif tifs/video2.tif [--ridge]

Output:
    results/filament_analysis.csv    — per-frame measurements
    results/filament_summary.pdf     — summary figure
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from scipy import ndimage
import os, sys, glob
import csv
from utils import best_device

# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL PARAMETERS — modify these if imaging conditions change
# ─────────────────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM   = 0.183   # µm per pixel
TIME_INTERVAL_MIN = 15.0  # minutes between frames
MIN_AREA_PX     = 10      # minimum pixels to count as a real filament (noise filter)
PROB_THRESHOLD  = 0.5     # U-Net probability threshold for binary mask

# ─────────────────────────────────────────────────────────────────────────────
#  Ridge Filter
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
#  U-Net (supports 1 or 2 channels)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
#  Morphological Measurement
# ─────────────────────────────────────────────────────────────────────────────
def measure_frame(mask):
    """
    Extract morphological properties from a binary mask.
    Uses covariance-matrix eigenvalues to estimate length, width, orientation.
    Returns list of dicts — one per detected filament blob.
    """
    labeled, n = ndimage.label(mask)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < MIN_AREA_PX:
            continue
        area_px = len(xs)
        area_um2 = area_px * PIXEL_SIZE_UM ** 2

        # Covariance matrix of pixel coordinates → eigenvalue-based dimensions
        coords = np.vstack([xs - xs.mean(), ys - ys.mean()])
        if coords.shape[1] < 2:
            continue
        cov = np.cov(coords)
        if cov.ndim < 2:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]
        # Length and width (4σ rule: covers ~95% of a Gaussian distribution)
        length_px = 4 * np.sqrt(max(eigenvalues[0], 0))
        width_px  = 4 * np.sqrt(max(eigenvalues[1], 0))
        length_um = length_px * PIXEL_SIZE_UM
        width_um  = width_px  * PIXEL_SIZE_UM
        # Orientation of major axis (angle from +x axis, in degrees)
        major_vec = eigenvectors[:, 0]
        angle_deg = np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180
        # Centroid
        cx_um = xs.mean() * PIXEL_SIZE_UM
        cy_um = ys.mean() * PIXEL_SIZE_UM

        blobs.append({
            'area_px':   area_px,
            'area_um2':  round(area_um2,  4),
            'length_um': round(length_um, 4),
            'width_um':  round(width_um,  4),
            'aspect_ratio': round(length_um / max(width_um, 0.001), 2),
            'angle_deg': round(angle_deg,  1),
            'cx_um':     round(cx_um, 3),
            'cy_um':     round(cy_um, 3),
        })
    return blobs

# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Event Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_events(per_frame_detected):
    """
    Detect contiguous runs of frames where filament is present.
    Returns list of (start_frame, end_frame, duration_min).
    """
    events = []
    in_event = False
    start = 0
    for i, detected in enumerate(per_frame_detected):
        if detected and not in_event:
            in_event = True; start = i
        elif not detected and in_event:
            in_event = False
            events.append((start, i - 1, (i - start) * TIME_INTERVAL_MIN))
    if in_event:
        events.append((start, len(per_frame_detected) - 1,
                       (len(per_frame_detected) - start) * TIME_INTERVAL_MIN))
    return events

# ─────────────────────────────────────────────────────────────────────────────
#  Main Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_video(filepath, model, device, use_ridge, verbose=True):
    """Run U-Net + morphological analysis on a single video. Returns list of row dicts."""
    video_name = os.path.basename(filepath)
    raw = tifffile.imread(filepath).astype(np.float32)
    nf = raw.shape[0]

    # Per-frame normalise
    normd = np.zeros_like(raw)
    for i in range(nf):
        mn, mx = raw[i].min(), raw[i].max()
        if mx > mn: normd[i] = (raw[i] - mn) / (mx - mn)

    ridge_maps = None
    if use_ridge:
        ridge_maps = np.array([ridge_filter_single(normd[i]) for i in range(nf)])

    rows = []
    with torch.no_grad():
        for i in range(nf):
            time_min = i * TIME_INTERVAL_MIN
            if use_ridge:
                inp = torch.from_numpy(np.stack([normd[i], ridge_maps[i]], axis=0)).float().unsqueeze(0).to(device)
            else:
                inp = torch.from_numpy(normd[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            mask = (prob > PROB_THRESHOLD).astype(np.float32)

            blobs = measure_frame(mask)
            if blobs:
                for b in blobs:
                    rows.append({'video': video_name, 'frame': i, 'time_min': time_min,
                                 'n_filaments_frame': len(blobs), **b})
            else:
                # Empty frame — record zeros so every frame appears in CSV
                rows.append({'video': video_name, 'frame': i, 'time_min': time_min,
                             'n_filaments_frame': 0, 'area_px': 0, 'area_um2': 0,
                             'length_um': 0, 'width_um': 0, 'aspect_ratio': 0,
                             'angle_deg': np.nan, 'cx_um': np.nan, 'cy_um': np.nan})

    if verbose:
        n_detected = sum(1 for r in rows if r['n_filaments_frame'] > 0)
        print(f"  {video_name}: {n_detected}/{nf} frames with filament")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_summary_figure(all_rows, output_path):
    # Group by video
    videos = sorted(set(r['video'] for r in all_rows))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(videos), 1)))

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Filament Analysis Summary", fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    ax_timeline = fig.add_subplot(gs[0, :])   # full width: detection timeline
    ax_length   = fig.add_subplot(gs[1, 0])
    ax_area     = fig.add_subplot(gs[1, 1])
    ax_aspect   = fig.add_subplot(gs[1, 2])
    ax_angle    = fig.add_subplot(gs[2, 0])
    ax_lifetime = fig.add_subplot(gs[2, 1])
    ax_freq     = fig.add_subplot(gs[2, 2])

    # --- Timeline ---
    for vi, v in enumerate(videos):
        rows_v = [r for r in all_rows if r['video'] == v]
        times  = [r['time_min'] for r in rows_v]
        detect = [1 if r['n_filaments_frame'] > 0 else 0 for r in rows_v]
        # One row per video, shaded where filament detected
        y_base = vi
        for ri, (t, d) in enumerate(zip(times, detect)):
            if d:
                ax_timeline.barh(y_base, TIME_INTERVAL_MIN, left=t, height=0.7,
                                 color=colors[vi], alpha=0.8)
    ax_timeline.set_yticks(range(len(videos)))
    ax_timeline.set_yticklabels([v[:30] for v in videos], fontsize=8)
    ax_timeline.set_xlabel("Time (min)")
    ax_timeline.set_title("Detection Timeline (shaded = filament present)")
    ax_timeline.set_xlim(0, max(r['time_min'] for r in all_rows) + TIME_INTERVAL_MIN)

    # Helper: get metric for detected frames only
    def get_metric(row_list, key):
        return [r[key] for r in row_list if r['n_filaments_frame'] > 0 and r[key] > 0]

    # --- Length histogram ---
    for vi, v in enumerate(videos):
        vals = get_metric([r for r in all_rows if r['video'] == v], 'length_um')
        if vals: ax_length.hist(vals, bins=15, alpha=0.6, color=colors[vi], label=v[:20])
    ax_length.set_xlabel("Length (µm)"); ax_length.set_ylabel("Count")
    ax_length.set_title("Filament Length Distribution"); ax_length.legend(fontsize=7)

    # --- Area histogram ---
    for vi, v in enumerate(videos):
        vals = get_metric([r for r in all_rows if r['video'] == v], 'area_um2')
        if vals: ax_area.hist(vals, bins=15, alpha=0.6, color=colors[vi])
    ax_area.set_xlabel("Area (µm²)"); ax_area.set_title("Filament Area Distribution")

    # --- Aspect ratio ---
    for vi, v in enumerate(videos):
        vals = get_metric([r for r in all_rows if r['video'] == v], 'aspect_ratio')
        if vals: ax_aspect.hist(vals, bins=15, alpha=0.6, color=colors[vi])
    ax_aspect.set_xlabel("Aspect Ratio (L/W)"); ax_aspect.set_title("Aspect Ratio Distribution")

    # --- Orientation (polar-like histogram with 180° symmetry) ---
    for vi, v in enumerate(videos):
        rows_v = [r for r in all_rows if r['video'] == v and r['n_filaments_frame'] > 0]
        angles = [r['angle_deg'] for r in rows_v if not np.isnan(r.get('angle_deg', np.nan))]
        if angles: ax_angle.hist(angles, bins=18, range=(0, 180), alpha=0.6, color=colors[vi])
    ax_angle.set_xlabel("Orientation (°, 0–180)"); ax_angle.set_title("Filament Orientation")
    ax_angle.set_xlim(0, 180)

    # --- Lifetime bar chart per video ---
    lifetime_means = []
    for v in videos:
        rows_v = [r for r in all_rows if r['video'] == v]
        detected = [r['n_filaments_frame'] > 0 for r in rows_v]
        events = detect_events(detected)
        mean_life = np.mean([e[2] for e in events]) if events else 0
        lifetime_means.append(mean_life)
    ax_lifetime.bar(range(len(videos)), lifetime_means, color=colors[:len(videos)], alpha=0.8)
    ax_lifetime.set_xticks(range(len(videos)))
    ax_lifetime.set_xticklabels([v[:15] for v in videos], rotation=30, ha='right', fontsize=7)
    ax_lifetime.set_ylabel("Mean event duration (min)")
    ax_lifetime.set_title("Mean Filament Lifetime per Video")

    # --- Detection frequency per video ---
    freqs = []
    for v in videos:
        rows_v = [r for r in all_rows if r['video'] == v]
        nf = len(rows_v)
        nd = sum(1 for r in rows_v if r['n_filaments_frame'] > 0)
        freqs.append(100 * nd / nf if nf > 0 else 0)
    ax_freq.bar(range(len(videos)), freqs, color=colors[:len(videos)], alpha=0.8)
    ax_freq.set_xticks(range(len(videos)))
    ax_freq.set_xticklabels([v[:15] for v in videos], rotation=30, ha='right', fontsize=7)
    ax_freq.set_ylabel("% frames with filament")
    ax_freq.set_title("Detection Frequency per Video")
    ax_freq.set_ylim(0, 100)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"  Saved figure: {output_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args      = sys.argv[1:]
    use_ridge = '--ridge' in args
    tif_files = [a for a in args if not a.startswith('--')]

    if not tif_files:
        print("Usage: python filament_analyser.py tifs/video1.tif [video2.tif ...] [--ridge]")
        sys.exit(1)

    model_path = "models/filament_unet_ridge.pt" if use_ridge else "models/filament_unet.pt"
    n_ch       = 2 if use_ridge else 1
    label      = "2-ch (raw+ridge)" if use_ridge else "1-ch (raw only)"

    if not os.path.exists(model_path):
        print(f"Error: No model found at '{model_path}'")
        sys.exit(1)

    device = best_device()
    model  = make_unet(n_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {label} model [{sum(p.numel() for p in model.parameters()):,} params, {device}]")
    print(f"Physical scale: {PIXEL_SIZE_UM} µm/px | {TIME_INTERVAL_MIN} min/frame\n")

    all_rows = []
    for fp in tif_files:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skipping."); continue
        print(f"Analysing {fp}...")
        rows = analyse_video(fp, model, device, use_ridge)
        all_rows.extend(rows)

    if not all_rows:
        print("No data to save."); sys.exit(1)

    # ── Print per-video summary ──
    print("\n" + "="*60)
    print(f"{'Video':<35} {'Frames':>6} {'Detected':>8} {'%':>5} {'MeanLen(µm)':>12} {'MeanLifetime(min)':>18}")
    print("─"*60)
    for v in sorted(set(r['video'] for r in all_rows)):
        rows_v = [r for r in all_rows if r['video'] == v]
        nf = len(rows_v)
        nd = sum(1 for r in rows_v if r['n_filaments_frame'] > 0)
        lengths = [r['length_um'] for r in rows_v if r['length_um'] > 0]
        mean_len = np.mean(lengths) if lengths else 0
        events = detect_events([r['n_filaments_frame'] > 0 for r in rows_v])
        mean_life = np.mean([e[2] for e in events]) if events else 0
        print(f"  {v:<33} {nf:>6} {nd:>8} {100*nd/nf:>4.0f}% {mean_len:>11.2f} {mean_life:>17.1f}")
    print("="*60)

    # ── CSV ──
    os.makedirs("results", exist_ok=True)
    csv_path = "results/filament_analysis.csv"
    fieldnames = ['video','frame','time_min','n_filaments_frame',
                  'area_px','area_um2','length_um','width_um','aspect_ratio',
                  'angle_deg','cx_um','cy_um']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV: {csv_path}  ({len(all_rows)} rows)")

    # ── Figure ──
    fig_path = "results/filament_summary.pdf"
    print("Generating summary figure...")
    make_summary_figure(all_rows, fig_path)
    print("\nDone!")

if __name__ == "__main__":
    main()
