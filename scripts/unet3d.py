import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage

# ─── 3D U-Net Architecture ───────────────────────────────────────────────────

class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet3D(nn.Module):
    """
    3D U-Net designed for shallow Z-stacks (e.g. 5 planes).
    Uses anisotropic pooling: pools in XY (2x2) but NOT in Z (1x).
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc1 = Conv3dBlock(in_ch, 16)
        self.enc2 = Conv3dBlock(16, 32)
        self.enc3 = Conv3dBlock(32, 64)
        
        # Pool only in H, W (Z is untouched)
        self.pool = nn.MaxPool3d((1, 2, 2))
        
        self.bottleneck = Conv3dBlock(64, 128)
        
        # Upsample only in H, W
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = Conv3dBlock(128, 64)
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = Conv3dBlock(64, 32)
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = Conv3dBlock(32, 16)
        
        self.out_conv = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        # x: (B, 1, Z, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out_conv(d1)

# ─── 3D Data Augmentation ────────────────────────────────────────────────────

class SegDataset3D(Dataset):
    """
    Dataset for 3D volumes with shape (Z, H, W).
    Applies 3D-aware augmentations dynamically.
    """
    def __init__(self, volumes, masks, augment_factor=10):
        self.volumes = volumes
        self.masks = masks
        self.aug = augment_factor

    def __len__(self):
        return len(self.volumes) * self.aug

    def __getitem__(self, idx):
        ri = idx // self.aug
        img = self.volumes[ri].copy()
        mask = self.masks[ri].copy()
        
        if idx % self.aug != 0:
            # 1. Additive Gaussian noise
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            
            # 2. 3D Flips
            if np.random.rand() > 0.5:
                img, mask = img[:, :, ::-1].copy(), mask[:, :, ::-1].copy()  # Flip X
            if np.random.rand() > 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1, :].copy()  # Flip Y
            if np.random.rand() > 0.5:
                img, mask = img[::-1, :, :].copy(), mask[::-1, :, :].copy()  # Flip Z
                
            # 3. XY Rotation (rotate around Z axis)
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(1, 2), reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, axes=(1, 2), reshape=False, order=0)
            
            # 4. Brightness jitter
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            
            # Hard threshold mask to keep binary
            mask = (mask > 0.5).astype(np.float32)
            
        # Add channel dim: (1, Z, H, W)
        return (torch.from_numpy(img).float().unsqueeze(0),
                torch.from_numpy(mask).float().unsqueeze(0))

# ─── Loss Functions ──────────────────────────────────────────────────────────

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    return 1 - (2 * inter + smooth) / (probs.sum() + targets.sum() + smooth)
