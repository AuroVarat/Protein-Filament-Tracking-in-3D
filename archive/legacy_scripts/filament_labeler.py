#!/usr/bin/env python3
"""
Filament Labeler — Human-in-the-Loop Interactive Labeling & Training UI

Usage:
    python filament_labeler.py tifs/video1.tif [tifs/video2.tif ...]
    
Keyboard:
    y / Right-click  = Label as FILAMENT
    n / Middle-click = Label as NO FILAMENT
    ← / →           = Navigate frames
    
Buttons:
    Train Model     = Train CNN on all labeled frames
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
import json
import os
import sys

# ─── Ridge Filter (Anisotropy Detector) ──────────────────────────────────────

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

# ─── Model Architecture ─────────────────────────────────────────────────────

class FilamentCNN(nn.Module):
    """Tiny CNN for binary classification — 2 channels: raw + ridge filter."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),   # 2 input channels!
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ─── Data Augmentation Dataset ──────────────────────────────────────────────

class AugmentedFrameDataset(Dataset):
    """Dataset that generates augmented copies of labeled 2-channel frames."""
    def __init__(self, frames, labels, augment_factor=10):
        self.frames = frames   # List of (H, W) numpy arrays
        self.ridges = [ridge_filter_single(f) for f in frames]  # Precompute ridge
        self.labels = labels
        self.augment_factor = augment_factor
        
    def __len__(self):
        return len(self.frames) * self.augment_factor
    
    def __getitem__(self, idx):
        real_idx = idx // self.augment_factor
        frame = self.frames[real_idx].copy()
        label = self.labels[real_idx]
        
        # Augmentation (skip for first copy = clean original)
        if idx % self.augment_factor != 0:
            noise_sigma = np.random.uniform(0.02, 0.10)
            frame = frame + np.random.randn(*frame.shape).astype(np.float32) * noise_sigma
            if np.random.rand() > 0.5:
                frame = frame[:, ::-1].copy()
            if np.random.rand() > 0.5:
                frame = frame[::-1, :].copy()
            brightness = np.random.uniform(0.8, 1.2)
            frame = frame * brightness
            angle = np.random.uniform(-15, 15)
            frame = ndimage.rotate(frame, angle, reshape=False, order=1)
            frame = np.clip(frame, 0, 1)
        
        # Compute ridge filter on the (possibly augmented) frame
        ridge = ridge_filter_single(frame)
        
        # Stack as 2 channels: [raw, ridge] → 2×H×W
        tensor = torch.from_numpy(np.stack([frame, ridge], axis=0)).float()
        return tensor, torch.tensor([label], dtype=torch.float32)

# ─── Labels Manager ─────────────────────────────────────────────────────────

class LabelsManager:
    """Handles saving and loading labels from a JSON file."""
    def __init__(self, path="models/labels.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {}
    
    def set_label(self, filepath, frame_idx, label):
        key = os.path.abspath(filepath)
        if key not in self.data:
            self.data[key] = {}
        self.data[key][str(frame_idx)] = label
        self._save()
        
    def get_label(self, filepath, frame_idx):
        key = os.path.abspath(filepath)
        if key in self.data and str(frame_idx) in self.data[key]:
            return self.data[key][str(frame_idx)]
        return None
    
    def get_all_labeled(self):
        """Returns list of (filepath, frame_idx, label) tuples."""
        results = []
        for filepath, frame_labels in self.data.items():
            for frame_idx_str, label in frame_labels.items():
                results.append((filepath, int(frame_idx_str), label))
        return results
    
    def count(self):
        return sum(len(v) for v in self.data.values())
    
    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

# ─── Training Function ──────────────────────────────────────────────────────

def train_model(labels_manager, model_save_path="models/filament_model.pt"):
    """Train the CNN on all labeled frames."""
    labeled = labels_manager.get_all_labeled()
    if len(labeled) < 10:
        print(f"⚠ Only {len(labeled)} labels. Need at least 10 to train meaningfully.")
        return None
    
    # Count class balance
    pos = sum(1 for _, _, l in labeled if l == 1)
    neg = len(labeled) - pos
    print(f"\n{'─'*50}")
    print(f"Training on {len(labeled)} labeled frames")
    print(f"  Filament: {pos}  |  No Filament: {neg}")
    if pos == 0 or neg == 0:
        print("⚠ Need at least one example of each class!")
        return None
    
    # Load all frames
    frames_cache = {}
    frames = []
    labels = []
    
    for filepath, frame_idx, label in labeled:
        if filepath not in frames_cache:
            print(f"  Loading {os.path.basename(filepath)}...")
            raw = tifffile.imread(filepath).astype(np.float32)
            # Per-frame normalization
            normalized = np.zeros_like(raw)
            for i in range(raw.shape[0]):
                f_min, f_max = raw[i].min(), raw[i].max()
                if f_max > f_min:
                    normalized[i] = (raw[i] - f_min) / (f_max - f_min)
            frames_cache[filepath] = normalized
        
        frames.append(frames_cache[filepath][frame_idx])
        labels.append(float(label))
    
    # Create dataset with augmentation
    dataset = AugmentedFrameDataset(frames, labels, augment_factor=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"  Augmented dataset: {len(dataset)} samples")
    
    # Setup model
    device = best_device()
    model = FilamentCNN().to(device)
    
    # Weighted loss to handle class imbalance
    pos_weight = torch.tensor([neg / pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters on {device}")
    print(f"  Training...")
    
    # Train for 20 epochs
    model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        acc = 100 * correct / total
        avg_loss = total_loss / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/20  Loss: {avg_loss:.4f}  Accuracy: {acc:.1f}%")
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"  ✓ Model saved to {model_save_path}")
    print(f"  ✓ Final accuracy: {acc:.1f}%")
    print(f"{'─'*50}\n")
    
    return model

# ─── Interactive Labeling UI ─────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python filament_labeler.py <video1.tif> [video2.tif ...]")
        sys.exit(1)
    
    tif_files = sys.argv[1:]
    
    # Load all videos (per-frame normalization)
    all_frames = []   # (filepath, frame_idx, frame_data)
    for filepath in tif_files:
        print(f"Loading {filepath}...")
        img = tifffile.imread(filepath).astype(np.float32)
        
        for i in range(img.shape[0]):
            f_min, f_max = img[i].min(), img[i].max()
            if f_max > f_min:
                norm_frame = (img[i] - f_min) / (f_max - f_min)
            else:
                norm_frame = np.zeros_like(img[i])
            all_frames.append((filepath, i, norm_frame))
    
    total_frames = len(all_frames)
    print(f"Loaded {total_frames} frames from {len(tif_files)} video(s)")
    
    labels_mgr = LabelsManager()
    print(f"Existing labels: {labels_mgr.count()}")
    
    # Find first unlabeled frame
    current_idx = [0]
    for i, (fp, fi, _) in enumerate(all_frames):
        if labels_mgr.get_label(fp, fi) is None:
            current_idx[0] = i
            break
    
    # ── Build the UI ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.30)
    fig.suptitle("Filament Labeler — Human in the Loop", fontsize=16, fontweight='bold')
    
    fp, fi, frame = all_frames[current_idx[0]]
    img_display = ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    
    label_val = labels_mgr.get_label(fp, fi)
    status_text = "UNLABELED" if label_val is None else ("✓ FILAMENT" if label_val == 1 else "✗ NO FILAMENT")
    title_text = ax.set_title(
        f"[{current_idx[0]+1}/{total_frames}] {os.path.basename(fp)} Frame {fi}\n{status_text}",
        fontsize=12
    )
    
    # Progress text
    labeled_count = labels_mgr.count()
    progress_text = fig.text(0.5, 0.22, f"Labeled: {labeled_count}/{total_frames}", 
                             ha='center', fontsize=11, fontweight='bold')
    
    # Navigation slider
    ax_slider = plt.axes([0.2, 0.17, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=current_idx[0], valstep=1, initcolor='none')
    
    # Buttons
    ax_yes     = plt.axes([0.15, 0.08, 0.20, 0.06])
    ax_no      = plt.axes([0.40, 0.08, 0.20, 0.06])
    ax_train   = plt.axes([0.65, 0.08, 0.20, 0.06])
    
    btn_yes   = Button(ax_yes,   '✓ Filament (Y)', color='#2d5f2d', hovercolor='#3a7a3a')
    btn_no    = Button(ax_no,    '✗ No Filament (N)', color='#5f2d2d', hovercolor='#7a3a3a')
    btn_train = Button(ax_train, '⚡ Train Model', color='#2d2d5f', hovercolor='#3a3a7a')
    
    # Style button text
    for btn in [btn_yes, btn_no, btn_train]:
        btn.label.set_color('white')
        btn.label.set_fontweight('bold')
    
    _updating = [False]  # Guard against slider ↔ refresh recursion
    
    def refresh_display():
        if _updating[0]:
            return
        _updating[0] = True
        
        idx = current_idx[0]
        fp, fi, frame = all_frames[idx]
        img_display.set_data(frame)
        
        label_val = labels_mgr.get_label(fp, fi)
        if label_val is None:
            status = "UNLABELED"
            color = 'white'
        elif label_val == 1:
            status = "✓ FILAMENT"
            color = '#44ff44'
        else:
            status = "✗ NO FILAMENT"
            color = '#ff4444'
        
        title_text.set_text(f"[{idx+1}/{total_frames}] {os.path.basename(fp)} Frame {fi}\n{status}")
        title_text.set_color(color)
        
        progress_text.set_text(f"Labeled: {labels_mgr.count()}/{total_frames}")
        slider.set_val(idx)
        fig.canvas.draw_idle()
        
        _updating[0] = False
    
    def label_and_advance(label):
        fp, fi, _ = all_frames[current_idx[0]]
        labels_mgr.set_label(fp, fi, label)
        
        # Auto-advance to next unlabeled frame
        if current_idx[0] < total_frames - 1:
            current_idx[0] += 1
        refresh_display()
    
    def on_yes(event):
        label_and_advance(1)
    
    def on_no(event):
        label_and_advance(0)
    
    def on_train(event):
        print("\n⚡ Starting training...")
        train_model(labels_mgr)
        print("Training complete! You can continue labeling or close the window.")
    
    def on_slider(val):
        current_idx[0] = int(val)
        refresh_display()
    
    def on_key(event):
        if event.key == 'y':
            label_and_advance(1)
        elif event.key == 'n':
            label_and_advance(0)
        elif event.key == 'right':
            if current_idx[0] < total_frames - 1:
                current_idx[0] += 1
                refresh_display()
        elif event.key == 'left':
            if current_idx[0] > 0:
                current_idx[0] -= 1
                refresh_display()
    
    btn_yes.on_clicked(on_yes)
    btn_no.on_clicked(on_no)
    btn_train.on_clicked(on_train)
    slider.on_changed(on_slider)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n" + "="*50)
    print("  LABELING UI READY")
    print("  Keys: Y = Filament, N = No Filament")
    print("  Keys: ← → = Navigate frames")
    print("  Click 'Train Model' when ready")
    print("="*50 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
