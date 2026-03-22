import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage

# ─── 2D U-Net Architecture ───────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet2D(nn.Module):
    """Small 2D U-Net for grayscale → binary mask."""
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        
        self.out_conv = nn.Conv2d(16, out_ch, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out_conv(d1)

# ─── 2D Temporal Dataset ────────────────────────────────────────────────────

class SegDataset2DTemporal(Dataset):
    """
    Dataset for 2D temporal sequences. 
    Yields 3 consecutive frames: (t-1, t, t+1) and their corresponding masks.
    """
    def __init__(self, sequences, masks, valid_masks, augment_factor=10):
        self.sequences = sequences
        self.masks = masks
        self.valid_masks = valid_masks
        self.aug = augment_factor
        self.indices = []
        for v_idx, seq in enumerate(sequences):
            T = seq.shape[0]
            for t in range(T):
                self.indices.append((v_idx, t))

    def __len__(self):
        return len(self.indices) * self.aug

    def __getitem__(self, idx):
        real_idx = idx // self.aug
        v_idx, t = self.indices[real_idx]
        seq = self.sequences[v_idx]; mask_seq = self.masks[v_idx]; val_seq = self.valid_masks[v_idx]
        T = seq.shape[0]
        t_prev = max(0, t - 1); t_next = min(T - 1, t + 1)
        
        img = np.stack([seq[t_prev], seq[t], seq[t_next]], axis=0).copy()
        mask = np.stack([mask_seq[t_prev], mask_seq[t], mask_seq[t_next]], axis=0).copy()
        valid = np.stack([val_seq[t_prev], val_seq[t], val_seq[t_next]], axis=0).copy()
        
        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            if np.random.rand() > 0.5: img, mask = img[:, :, ::-1].copy(), mask[:, :, ::-1].copy()
            if np.random.rand() > 0.5: img, mask = img[:, ::-1, :].copy(), mask[:, ::-1, :].copy()
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(1, 2), reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, axes=(1, 2), reshape=False, order=0)
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            mask = (mask > 0.5).astype(np.float32)
            
        return (torch.from_numpy(img).float(), torch.from_numpy(mask).float(), torch.from_numpy(valid).float().view(3, 1, 1))

class SegDataset2DTemporalAuto(Dataset):
    """
    Dataset for 2D temporal sequences with DILATED Auto-Thresholding.
    """
    def __init__(self, sequences, masks, valid_masks, augment_factor=10, intensity_thresh=0.5):
        self.sequences = sequences; self.masks = masks; self.valid_masks = valid_masks; self.aug = augment_factor; self.thresh = intensity_thresh
        self.indices = []
        for v_idx, seq in enumerate(sequences):
            T = seq.shape[0]
            for t in range(T): self.indices.append((v_idx, t))

    def __len__(self): return len(self.indices) * self.aug

    def __getitem__(self, idx):
        real_idx = idx // self.aug
        v_idx, t = self.indices[real_idx]
        seq = self.sequences[v_idx]; mask_seq = self.masks[v_idx]; val_seq = self.valid_masks[v_idx]
        T = seq.shape[0]
        t_prev = max(0, t - 1); t_next = min(T - 1, t + 1)
        
        img = np.stack([seq[t_prev], seq[t], seq[t_next]], axis=0).copy()
        raw_mask = np.stack([mask_seq[t_prev], mask_seq[t], mask_seq[t_next]], axis=0).copy()
        valid = np.stack([val_seq[t_prev], val_seq[t], val_seq[t_next]], axis=0).copy()
        
        structuring_element = ndimage.generate_binary_structure(2, 1)
        refined_mask = np.zeros_like(raw_mask)
        for i in range(3):
            dilated = ndimage.binary_dilation(raw_mask[i] > 0, structure=structuring_element)
            refined_mask[i] = (dilated & (img[i] > self.thresh)).astype(np.float32)

        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            if np.random.rand() > 0.5: img, refined_mask = img[:, :, ::-1].copy(), refined_mask[:, :, ::-1].copy()
            if np.random.rand() > 0.5: img, refined_mask = img[:, ::-1, :].copy(), refined_mask[:, ::-1, :].copy()
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(1, 2), reshape=False, order=1)
            refined_mask = ndimage.rotate(refined_mask, angle, axes=(1, 2), reshape=False, order=0)
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            refined_mask = (refined_mask > 0.5).astype(np.float32)
            
        return (torch.from_numpy(img).float(), torch.from_numpy(refined_mask).float(), torch.from_numpy(valid).float().view(3, 1, 1))
