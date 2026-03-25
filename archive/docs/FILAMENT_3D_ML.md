# 3D Filament Detection ML Pipeline

This pipeline extends the project's filament detection capabilities to **3D volumetric data** (Z-stacks over time). It is designed to handle multi-slice TIFFs (e.g., shapes like `T × Z × H × W`).

## Architecture Highlights

The core model is a **3D Tiny U-Net** (`TinyUNet3D` in `scripts/unet3d.py`). 

**Critical Design Decision: Anisotropic Pooling**
Because the Z-depth is physically very shallow (e.g., Z=5) while XY dimensions are large (e.g., 128x128), standard 3D pooling (`2x2x2`) would immediately collapse the depth dimension to 1, losing all Z-resolution in the first layer. 

Instead, the model uses **anisotropic pooling** `(1, 2, 2)`. It performs convolutions across all 3 dimensions (`kernel_size=3`), but only downsamples and upsamples in the spatial XY dimensions. This allows the network to build deep hierarchical features while preserving the delicate 5-slice depth through to the final output.

## Workflow & Scripts

The 3D pipeline consists of three completely independent scripts:

### 1. Annotation (`filament_3d_viewer.py`)
Provides an interactive **tri-planar viewer** showing simultaneous slices through the XY, XZ, and YZ planes.
- **Usage**: `python scripts/filament_3d_viewer.py tifs3d/volume1.tif`
- **Action**: Use the sliders to navigate the 3D volume. Left-click and drag on *any* of the three planes to paint a 3D spherical mask.
- **Output**: Press `S` to save. Masks are saved as `(Z, H, W)` numpy arrays in `models/masks3d/`.

### 2. Training (`train_3d.py`)
Trains the 3D U-Net using a combined BCE + Dice loss, utilizing dynamic 3D-aware augmentations.
- **Usage**: `python scripts/train_3d.py tifs3d/volume1.tif tifs3d/volume2.tif`
- **Data Augmentation**: Automatically applies 3D flips, XY-plane rotations, additive Gaussian noise, and brightness jitter during training to prevent overfitting on the small dataset.
- **Output**: Saves the trained weights to `models/filament_unet3d.pt`.

### 3. Inference (`filament_3d_segmenter.py`)
Runs the trained model on new, unseen videos and provides an interactive viewer to inspect the predicted 3D boundaries.
- **Usage**: `python scripts/filament_3d_segmenter.py tifs3d/new_volume.tif`
- **Action**: Sweeps through the timepoints and generates a pixel-level 3D probability map and binary mask. Displays the result as a green overlay on the raw data across all three orthogonal planes.

## Notes on Data Dimensions
- The 3D scripts automatically interpret TIFF ranks. 
- Input expected rank is primarily 4D: `(T, Z, H, W)`.
- If a 5D TIFF is provided `(T, Z, C, H, W)`, it will automatically take the second channel `C=1`.
- All frames are per-slice Min-Max normalized to `[0, 1]` for the neural network (each 2D Z-slice is independently normalized per timepoint).
