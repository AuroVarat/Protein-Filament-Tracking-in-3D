import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage

# ─── 3D Ridge Filter ─────────────────────────────────────────────────────────

def ridge_filter_3d(img, sigma=1.5, beta=0.5, c_thresh=0.02):
    """
    3D Frangi vesselness filter — highlights tubular/filament structures.
    Uses 3D Hessian eigenvalues.
    img: 3D numpy array (Z, H, W).
    """
    # Compute 3D Hessian matrix components
    Hxx = ndimage.gaussian_filter(img, sigma, order=[0, 0, 2])
    Hyy = ndimage.gaussian_filter(img, sigma, order=[0, 2, 0])
    Hzz = ndimage.gaussian_filter(img, sigma, order=[2, 0, 0])
    Hxy = ndimage.gaussian_filter(img, sigma, order=[0, 1, 1])
    Hxz = ndimage.gaussian_filter(img, sigma, order=[1, 0, 1])
    Hyz = ndimage.gaussian_filter(img, sigma, order=[1, 1, 0])

    # For each voxel, we need to compute eigenvalues of the 3x3 Hessian matrix:
    # [ Hxx  Hxy  Hxz ]
    # [ Hxy  Hyy  Hyz ]
    # [ Hxz  Hyz  Hzz ]
    
    # Flatten spatial dims to compute eigenvalues efficiently
    Z, H, W = img.shape
    N = Z * H * W
    
    H_mat = np.zeros((N, 3, 3))
    H_mat[:, 0, 0] = Hxx.ravel()
    H_mat[:, 1, 1] = Hyy.ravel()
    H_mat[:, 2, 2] = Hzz.ravel()
    H_mat[:, 0, 1] = Hxy.ravel()
    H_mat[:, 1, 0] = Hxy.ravel()
    H_mat[:, 0, 2] = Hxz.ravel()
    H_mat[:, 2, 0] = Hxz.ravel()
    H_mat[:, 1, 2] = Hyz.ravel()
    H_mat[:, 2, 1] = Hyz.ravel()

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(H_mat)  # returns sorted eigenvalues for each 3x3 matrix
    
    # Sort eigenvalues by magnitude
    # We want |lambda1| <= |lambda2| <= |lambda3|
    abs_eig = np.abs(eigenvalues)
    sort_idx = np.argsort(abs_eig, axis=-1)
    
    lambda1 = np.take_along_axis(eigenvalues, sort_idx[:, [0]], axis=-1).squeeze()
    lambda2 = np.take_along_axis(eigenvalues, sort_idx[:, [1]], axis=-1).squeeze()
    lambda3 = np.take_along_axis(eigenvalues, sort_idx[:, [2]], axis=-1).squeeze()

    # Frangi vesselness function
    lambda2_safe = np.where(lambda2 == 0, 1e-10, lambda2)
    lambda3_safe = np.where(lambda3 == 0, 1e-10, lambda3)
    
    Ra = np.abs(lambda2) / np.abs(lambda3_safe)
    Rb = np.abs(lambda1) / np.sqrt(np.abs(lambda2_safe * lambda3_safe))
    S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
    
    c = max(np.max(S) / 3.0, c_thresh)
    
    alpha = 0.5
    V = (1 - np.exp(-(Ra**2) / (2*alpha**2))) * np.exp(-(Rb**2) / (2*beta**2)) * (1 - np.exp(-(S**2) / (2*c**2)))
    
    # Filter out blobs (lambda2 > 0 or lambda3 > 0 means dark structures)
    # Since we want bright structures:
    V[lambda2 > 0] = 0
    V[lambda3 > 0] = 0
    
    return V.reshape(Z, H, W).astype(np.float32)

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
    def __init__(self, in_ch=1, out_ch=1):
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
        
        self.out_conv = nn.Conv3d(16, out_ch, 1)

    def forward(self, x):
        # x: (B, 1, Z, H, W) or (B, 2, Z, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        # Resizing decoder path to match skip connections (handles odd dimensions)
        u3 = self.up3(b)
        if u3.shape[3:] != e3.shape[3:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape[3:] != e2.shape[3:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[3:] != e1.shape[3:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
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

class SegDataset3DAuto(Dataset):
    """
    Dataset for 3D volumes.
    Applies Auto-Thresholding: The raw 3D blob mask is refined by 
    intersecting it with bright pixels from the image volume.
    """
    def __init__(self, volumes, masks, augment_factor=10, intensity_thresh=0.5):
        self.volumes = volumes
        self.masks = masks
        self.aug = augment_factor
        self.thresh = intensity_thresh

    def __len__(self):
        return len(self.volumes) * self.aug

    def __getitem__(self, idx):
        ri = idx // self.aug
        img = self.volumes[ri].copy()
        mask = self.masks[ri].copy()
        
        # --- Auto-Thresholding the blob mask ---
        # The annotated mask acts as a bounding region. 
        # Only pixels inside the mask AND brighter than threshold are kept.
        refined_mask = ((mask > 0) & (img > self.thresh)).astype(np.float32)
        
        if idx % self.aug != 0:
            # 1. Additive Gaussian noise
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            
            # 2. 3D Flips
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, :, ::-1].copy(), refined_mask[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, ::-1, :].copy(), refined_mask[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[::-1, :, :].copy(), refined_mask[::-1, :, :].copy()
                
            # 3. XY Rotation (rotate around Z axis)
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(1, 2), reshape=False, order=1)
            refined_mask = ndimage.rotate(refined_mask, angle, axes=(1, 2), reshape=False, order=0)
            
            # 4. Brightness jitter
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            
            refined_mask = (refined_mask > 0.5).astype(np.float32)
            
        return (torch.from_numpy(img).float().unsqueeze(0),
                torch.from_numpy(refined_mask).float().unsqueeze(0))

class SegDataset3D2ch(Dataset):
    """
    Dataset for 3D volumes returning 2 channels: (Raw Image, 3D Ridge Filter).
    Also applies Auto-Thresholding to the mask.
    """
    def __init__(self, volumes, masks, augment_factor=10, intensity_thresh=0.5):
        self.volumes = volumes
        self.masks = masks
        self.aug = augment_factor
        self.thresh = intensity_thresh

    def __len__(self):
        return len(self.volumes) * self.aug

    def __getitem__(self, idx):
        ri = idx // self.aug
        img = self.volumes[ri].copy()
        mask = self.masks[ri].copy()
        
        # Auto-Threshold
        refined_mask = ((mask > 0) & (img > self.thresh)).astype(np.float32)
        
        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, :, ::-1].copy(), refined_mask[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, ::-1, :].copy(), refined_mask[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[::-1, :, :].copy(), refined_mask[::-1, :, :].copy()
                
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(1, 2), reshape=False, order=1)
            refined_mask = ndimage.rotate(refined_mask, angle, axes=(1, 2), reshape=False, order=0)
            
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            refined_mask = (refined_mask > 0.5).astype(np.float32)
            
        # Compute 3D ridge filter on the (augmented) image
        ridge = ridge_filter_3d(img)
        
        # Stack channels: (2, Z, H, W)
        two_ch = np.stack([img, ridge], axis=0)
        
        return (torch.from_numpy(two_ch).float(),
                torch.from_numpy(refined_mask).float().unsqueeze(0))

class SegDataset3DTemporal(Dataset):
    """
    Dataset for 3D temporal sequences. 
    Yields 3 consecutive frames: (t-1, t, t+1) and their corresponding masks.
    Output shapes: (3, Z, H, W) for both images and masks.
    Missing masks should be handled outside or ignored via valid masks.
    """
    def __init__(self, sequences, masks, valid_masks, augment_factor=10):
        # sequences: list of volumes of shape (T, Z, H, W)
        # masks: list of volumes of shape (T, Z, H, W)
        # valid_masks: list of 1D arrays of shape (T,) indicating if mask is valid
        self.sequences = sequences
        self.masks = masks
        self.valid_masks = valid_masks
        self.aug = augment_factor
        
        # Build index mapping
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
        
        seq = self.sequences[v_idx]
        mask_seq = self.masks[v_idx]
        val_seq = self.valid_masks[v_idx]
        T = seq.shape[0]
        
        # Get t-1, t, t+1 with padding at edges
        t_prev = max(0, t - 1)
        t_next = min(T - 1, t + 1)
        
        img = np.stack([seq[t_prev], seq[t], seq[t_next]], axis=0).copy()  # (3, Z, H, W)
        mask = np.stack([mask_seq[t_prev], mask_seq[t], mask_seq[t_next]], axis=0).copy()  # (3, Z, H, W)
        valid = np.stack([val_seq[t_prev], val_seq[t], val_seq[t_next]], axis=0).copy()  # (3,)
        
        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            
            if np.random.rand() > 0.5:
                img, mask = img[:, :, :, ::-1].copy(), mask[:, :, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img, mask = img[:, :, ::-1, :].copy(), mask[:, :, ::-1, :].copy()
            if np.random.rand() > 0.5:
                img, mask = img[:, ::-1, :, :].copy(), mask[:, ::-1, :, :].copy()
                
            angle = np.random.uniform(-15, 15)
            # rotate axes 2, 3 (H, W)
            img  = ndimage.rotate(img,  angle, axes=(2, 3), reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, axes=(2, 3), reshape=False, order=0)
            
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            mask = (mask > 0.5).astype(np.float32)
            
        return (torch.from_numpy(img).float(),
                torch.from_numpy(mask).float(),
                torch.from_numpy(valid).float().view(3, 1, 1, 1))

class SegDataset3DTemporalAuto(Dataset):
    """
    Dataset for 3D temporal sequences with DILATED Auto-Thresholding.
    Expands the user's manual blob by 1 pixel in all directions (dilation), 
    then strictly intersects it with bright pixels from the raw image.
    """
    def __init__(self, sequences, masks, valid_masks, augment_factor=10, intensity_thresh=0.5):
        self.sequences = sequences
        self.masks = masks
        self.valid_masks = valid_masks
        self.aug = augment_factor
        self.thresh = intensity_thresh
        
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
        
        seq = self.sequences[v_idx]
        mask_seq = self.masks[v_idx]
        val_seq = self.valid_masks[v_idx]
        T = seq.shape[0]
        
        t_prev = max(0, t - 1)
        t_next = min(T - 1, t + 1)
        
        img = np.stack([seq[t_prev], seq[t], seq[t_next]], axis=0).copy()
        raw_mask = np.stack([mask_seq[t_prev], mask_seq[t], mask_seq[t_next]], axis=0).copy()
        valid = np.stack([val_seq[t_prev], val_seq[t], val_seq[t_next]], axis=0).copy()
        
        # --- Morphological Augmentation & Thresholding ---
        # Dilate the raw blob mask by 1 pixel in 3D (Z, H, W) to catch underestimated edges
        structuring_element = ndimage.generate_binary_structure(3, 1) # cross shape
        
        refined_mask = np.zeros_like(raw_mask)
        for i in range(3):
            # Dilate the 3D mask for this frame
            dilated = ndimage.binary_dilation(raw_mask[i] > 0, structure=structuring_element)
            # Auto-threshold: must be inside dilated region AND bright enough
            refined_mask[i] = (dilated & (img[i] > self.thresh)).astype(np.float32)

        if idx % self.aug != 0:
            img += np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.02, 0.08)
            
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, :, :, ::-1].copy(), refined_mask[:, :, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, :, ::-1, :].copy(), refined_mask[:, :, ::-1, :].copy()
            if np.random.rand() > 0.5:
                img, refined_mask = img[:, ::-1, :, :].copy(), refined_mask[:, ::-1, :, :].copy()
                
            angle = np.random.uniform(-15, 15)
            img  = ndimage.rotate(img,  angle, axes=(2, 3), reshape=False, order=1)
            refined_mask = ndimage.rotate(refined_mask, angle, axes=(2, 3), reshape=False, order=0)
            
            img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            refined_mask = (refined_mask > 0.5).astype(np.float32)
            
        return (torch.from_numpy(img).float(),
                torch.from_numpy(refined_mask).float(),
                torch.from_numpy(valid).float().view(3, 1, 1, 1))

# ─── Loss Functions ──────────────────────────────────────────────────────────

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    return 1 - (2 * inter + smooth) / (probs.sum() + targets.sum() + smooth)
