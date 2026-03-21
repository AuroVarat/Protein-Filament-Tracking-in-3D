import numpy as np
from skimage.io import imread
import sys
import os

try:
    print(f"Loading files from {os.getcwd()}")
    img = imread('example.png')
    labeled = imread('labeled example.png')

    print("Image shape:", img.shape)
    print("Labeled shape:", labeled.shape)

    img_f = img.astype(float)
    labeled_f = labeled.astype(float)

    # find where they differ
    if img.shape == labeled.shape:
        diff = np.abs(img_f - labeled_f)
        if diff.ndim == 3:
            diff_mask = diff.sum(axis=2) > 20
        else:
            diff_mask = diff > 20
            
        if diff.ndim == 3:
            img_gray = img.mean(axis=2)
        else:
            img_gray = img
            
        filament_pixels = img_gray[diff_mask]
        bg_pixels = img_gray[~diff_mask]
        
        print(f"Filament pixels count: {len(filament_pixels)}")
        if len(filament_pixels) > 0:
            print(f"Mean filament intensity (gray): {filament_pixels.mean():.2f}")
            print(f"Mean background intensity (gray): {bg_pixels.mean():.2f}")
            y, x = np.where(diff_mask)
            print(f"Filament rough length scale: {max(y.max()-y.min(), x.max()-x.min())} pixels")
            print(f"Filament rough branch length min: probably need at least 10px? y_span={y.max()-y.min()}, x_span={x.max()-x.min()}")
        else:
            print("No difference found between labeled and original image!")
    else:
        print("Shapes differ, adjusting diff strategy...")
        print(f"img shape: {img.shape}, labeled shape: {labeled.shape}")
except Exception as e:
    print("Error:", repr(e))
