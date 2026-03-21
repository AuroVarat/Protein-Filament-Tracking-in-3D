#!/usr/bin/env python3
"""
Filament Predictor — Hybrid Ridge Filter + CNN with Grad-CAM Tracking

Usage:
    python filament_predictor.py tifs/video.tif
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import os
import sys

# ─── Ridge Filter ────────────────────────────────────────────────────────────

def ridge_filter_single(img, sigma=1.5, beta=0.5, c_thresh=0.02):
    """Frangi vesselness filter — highlights straight-line structures."""
    Hxx = ndimage.gaussian_filter(img, sigma, order=[2, 0])
    Hyy = ndimage.gaussian_filter(img, sigma, order=[0, 2])
    Hxy = ndimage.gaussian_filter(img, sigma, order=[1, 1])
    trace = Hxx + Hyy
    det_diff = np.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)
    l1 = 0.5 * (trace + det_diff)
    l2 = 0.5 * (trace - det_diff)
    mag_l1, mag_l2 = np.abs(l1), np.abs(l2)
    lambda1 = np.where(mag_l1 > mag_l2, l1, l2)
    lambda2 = np.where(mag_l1 > mag_l2, l2, l1)
    l1_safe = np.where(lambda1 == 0, 1e-10, lambda1)
    Rb = np.abs(lambda2) / np.abs(l1_safe)
    S = np.sqrt(lambda1**2 + lambda2**2)
    c = max(np.max(S) / 3.0, c_thresh)
    vesselness = np.exp(-(Rb**2) / (2 * beta**2)) * (1 - np.exp(-(S**2) / (2 * c**2)))
    vesselness[lambda1 > 0] = 0
    return vesselness.astype(np.float32)

# ─── Model Architecture (2-channel: raw + ridge) ────────────────────────────

class FilamentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))
        
        self._activations = None
        self._gradients = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._activations = x
        if x.requires_grad:
            x.register_hook(self._save_gradients)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    def _save_gradients(self, grad):
        self._gradients = grad
    
    def grad_cam(self, input_tensor):
        """Generate Grad-CAM heatmap for a 2-channel input."""
        self.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self(input_tensor)
        prob = torch.sigmoid(output).item()
        self.zero_grad()
        output.backward()
        
        if self._gradients is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w)), prob
        
        gradients = self._gradients.detach()
        activations = self._activations.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                           mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam, prob

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_predictor.py <video.tif>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    model_path = "models/filament_model.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: No trained model found at '{model_path}'")
        print("Train one first: python filament_labeler.py tifs/your_video.tif")
        sys.exit(1)
    
    # Load model
    device = torch.device("cpu")  # CPU for grad-cam backward hooks
    model = FilamentCNN().to(device)
    
    # Load saved weights, adapting key names from the Sequential format
    saved_state = torch.load(model_path, map_location=device, weights_only=True)
    
    old_to_new = {
        'features.0': 'conv1.0', 'features.1': 'conv1.1', 'features.2': 'conv1.2',
        'features.3': 'conv2.0', 'features.4': 'conv2.1', 'features.5': 'conv2.2',
        'features.6': 'conv3.0', 'features.7': 'conv3.1',
        'classifier.1': 'classifier.1',
    }
    
    new_state = {}
    for old_key, value in saved_state.items():
        mapped = False
        for old_prefix, new_prefix in old_to_new.items():
            if old_key.startswith(old_prefix + '.'):
                suffix = old_key[len(old_prefix):]
                new_key = new_prefix + suffix
                new_state[new_key] = value
                mapped = True
                break
        if not mapped:
            new_state[old_key] = value
    
    model.load_state_dict(new_state, strict=False)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load video
    print(f"Loading {filepath}...")
    image = tifffile.imread(filepath).astype(np.float32)
    num_frames = image.shape[0]
    
    # Per-frame normalization
    normalized = np.zeros_like(image)
    for i in range(num_frames):
        f_min, f_max = image[i].min(), image[i].max()
        if f_max > f_min:
            normalized[i] = (image[i] - f_min) / (f_max - f_min)
    
    # Precompute ridge filter maps
    print("Computing ridge filter maps...")
    ridge_maps = np.zeros_like(normalized)
    for i in range(num_frames):
        ridge_maps[i] = ridge_filter_single(normalized[i])
    
    # Compute Grad-CAM heatmaps
    print("Computing Grad-CAM heatmaps...")
    heatmaps = []
    confidences = []
    predictions = []
    
    for i in range(num_frames):
        # Stack 2 channels: raw + ridge
        two_ch = np.stack([normalized[i], ridge_maps[i]], axis=0)  # 2×H×W
        frame_tensor = torch.from_numpy(two_ch).float().unsqueeze(0).to(device)
        cam, prob = model.grad_cam(frame_tensor)
        heatmaps.append(cam)
        confidences.append(prob)
        predictions.append(1 if prob > 0.5 else 0)
        
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{num_frames} frames processed")
    
    heatmaps = np.array(heatmaps)
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    
    # Summary
    filament_frames = np.where(predictions == 1)[0]
    print(f"\nDetected filaments in {len(filament_frames)}/{num_frames} frames")
    if len(filament_frames) > 0:
        print(f"Filament frames: {filament_frames.tolist()}")
    
    # ── Build the Interactive Viewer ──
    fig, (ax_img, ax_ridge, ax_heat, ax_conf) = plt.subplots(1, 4, figsize=(22, 7),
                                                     gridspec_kw={'width_ratios': [2, 2, 2, 1]})
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle(f"Hybrid Ridge+CNN Tracker: {os.path.basename(filepath)}", fontsize=14, fontweight='bold')
    
    # Col 1: Raw frame
    img_display = ax_img.imshow(normalized[0], cmap='gray', vmin=0, vmax=1)
    ax_img.axis('off')
    ax_img.set_title("Normalized Frame")
    
    # Col 2: Ridge filter output
    ridge_display = ax_ridge.imshow(ridge_maps[0], cmap='inferno')
    r_max = np.max(ridge_maps[0])
    ridge_display.set_clim(0, r_max if r_max > 0 else 1)
    ax_ridge.axis('off')
    ax_ridge.set_title("Ridge Filter (CNN Input Ch.2)")
    
    # Col 3: Grad-CAM overlay
    img_base = ax_heat.imshow(normalized[0], cmap='gray', vmin=0, vmax=1)
    heatmap_overlay = ax_heat.imshow(heatmaps[0], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax_heat.axis('off')
    
    # Col 4: Confidence timeline
    bar_colors = ['#44ff44' if p == 1 else '#333333' for p in predictions]
    ax_conf.barh(range(num_frames), confidences, color=bar_colors, height=1.0)
    ax_conf.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax_conf.set_xlabel('Prob', fontsize=9)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(-0.5, num_frames - 0.5)
    ax_conf.invert_yaxis()
    ax_conf.legend(fontsize=8)
    ax_conf.set_title("Confidence", fontsize=10)
    playhead = ax_conf.axhline(y=0, color='cyan', linewidth=2, alpha=0.8)
    
    prob = confidences[0]
    pred = predictions[0]
    status = f"FILAMENT ({prob*100:.0f}%)" if pred == 1 else f"No Filament ({prob*100:.0f}%)"
    status_color = '#44ff44' if pred == 1 else '#ff4444'
    heat_title = ax_heat.set_title(f"Grad-CAM — {status}", fontsize=11, color=status_color)
    
    # Slider
    ax_slider = plt.axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1, initcolor='none')
    
    def update(val):
        t = int(val)
        img_display.set_data(normalized[t])
        
        ridge_display.set_data(ridge_maps[t])
        r_max = np.max(ridge_maps[t])
        ridge_display.set_clim(0, r_max if r_max > 0 else 1)
        
        img_base.set_data(normalized[t])
        heatmap_overlay.set_data(heatmaps[t])
        
        prob = confidences[t]
        pred = predictions[t]
        status = f"FILAMENT ({prob*100:.0f}%)" if pred == 1 else f"No Filament ({prob*100:.0f}%)"
        status_color = '#44ff44' if pred == 1 else '#ff4444'
        heat_title.set_text(f"Grad-CAM — {status}")
        heat_title.set_color(status_color)
        
        playhead.set_ydata([t, t])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    print("\nLaunching Hybrid Ridge+CNN prediction viewer...")
    plt.show()

if __name__ == "__main__":
    main()
