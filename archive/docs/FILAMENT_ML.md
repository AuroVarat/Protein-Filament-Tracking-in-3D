# Filament Detection ML Pipeline

## Overview

This pipeline detects and spatially localises filamentous structures in live-cell TIFF microscopy videos (128×128 px, grayscale). It combines a classical mathematical ridge filter with a deep learning U-Net trained on human-annotated frames.

---

## The Problem

Standard microscopy image analysis tools (thresholding, blob detectors) struggle with filaments because:

- They are very thin (~3 px wide, ~22 px long)
- Their intensity varies between cells, conditions, and microscopes
- They appear and disappear transiently across video frames
- They are easily confused with membrane ruffles and fluorescent debris

The goal is to produce a **pixel-level segmentation mask** — a binary image where every filament pixel is labelled 1 and all background is 0.

---

## Architecture

The pipeline has two stages that run in series:

```
Raw TIFF Video
      │
      ▼
[Per-Frame Normalisation]     ← Each frame scaled independently to [0, 1]
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
[Ridge Filter (Frangi)]            [U-Net Segmentation]
Classical structure detector       Learned pixel-level mask
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
              Output: binary mask per frame
```

---

## Stage 1 — Frangi Ridge Filter

The **Frangi vesselness filter** is a classical second-order image analysis filter designed to detect tube-like (filamentous) structures. It works by analysing the **Hessian matrix** of the image intensity field.

### How it works

At every pixel, the Hessian matrix captures local curvature:

$$H = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}$$

where $I_{xx}$, $I_{yy}$, $I_{xy}$ are second-order Gaussian derivatives. Eigendecomposition of $H$ yields two eigenvalues $\lambda_1$, $\lambda_2$ (sorted $|\lambda_1| \geq |\lambda_2|$).

For a bright filament on dark background:

- $\lambda_1 \ll 0$ (strong curvature along the tube's cross-section)
- $\lambda_2 \approx 0$ (flat along the tube's length)

Two measures are derived:

| Measure                    | Formula                            | Purpose                  |
| -------------------------- | ---------------------------------- | ------------------------ | --- | --------- | --- | ------------------------------- |
| **Blob rejection** $R_B$   | $                                  | \lambda_2                | /   | \lambda_1 | $   | High for circles, low for lines |
| **Structure strength** $S$ | $\sqrt{\lambda_1^2 + \lambda_2^2}$ | Rejects background noise |

The final vesselness score at each pixel:

$$V = e^{-R_B^2 / 2\beta^2} \cdot \left(1 - e^{-S^2 / 2c^2}\right)$$

with $V = 0$ wherever $\lambda_1 > 0$ (bright blobs are suppressed). The result is a floating-point **anisotropy map** where high values indicate filament-like geometry.

### Role in the pipeline

The ridge filter is _not_ used as a final detector here — it is used as a **structural prior** fed into the CNN as a second input channel. This biases the neural network to attend to geometrically linear structures rather than learning spurious correlations from background texture.

---

## Stage 2 — U-Net Pixel Segmentation

### Architecture

A **Tiny U-Net** (≈ 483K parameters) operates on 128×128 grayscale frames and outputs a 128×128 probability map (one value per pixel: probability of being a filament pixel).

```
Input: 1×128×128 (normalised frame)
│
├─ Encoder
│   ├─ Conv(1→16)  + BN + ReLU × 2  → MaxPool  [64×64]
│   ├─ Conv(16→32) + BN + ReLU × 2  → MaxPool  [32×32]
│   └─ Conv(32→64) + BN + ReLU × 2  → MaxPool  [16×16]
│
├─ Bottleneck
│   └─ Conv(64→128) + BN + ReLU × 2            [16×16]
│
└─ Decoder (with skip connections)
    ├─ UpConv(128→64) + Skip(enc3) → Conv(128→64)  [32×32]
    ├─ UpConv(64→32)  + Skip(enc2) → Conv(64→32)   [64×64]
    └─ UpConv(32→16)  + Skip(enc1) → Conv(32→16)   [128×128]
        └─ Conv(16→1) → Output logits
```

Skip connections between encoder and decoder allow the model to recover fine spatial detail that is lost during pooling.

### Loss Function

Binary Cross Entropy with Logits:

$$\mathcal{L} = -[y \log(\sigma(\hat{y})) + (1-y)\log(1 - \sigma(\hat{y}))]$$

Monitoring metric is the **Dice coefficient**:

$$\text{Dice} = \frac{2 \cdot |P \cap G|}{|P| + |G|}$$

where $P$ is the predicted binary mask and $G$ is the ground-truth mask. A Dice of 1.0 means perfect agreement; 0.0 means no overlap.

### Data Augmentation

Since only ~20–70 manually annotated frames are available, each frame is augmented ×10 with:

| Transform                                  | Applied to                       |
| ------------------------------------------ | -------------------------------- |
| Gaussian noise ($\sigma \in [0.02, 0.08]$) | Image only                       |
| Horizontal flip                            | Both image and mask              |
| Vertical flip                              | Both image and mask              |
| Rotation (±15°)                            | Both (nearest-neighbor for mask) |
| Brightness jitter (×0.8–1.2)               | Image only                       |

Spatial transforms are applied **identically** to both image and mask to preserve correspondence.

---

## Human-in-the-Loop Annotation

Two annotation strategies are provided:

### 1. Bounding Box (`filament_boxer.py`)

The user draws a rectangle around a filament. An automatic intensity threshold ($> 0.5$ on the normalised frame) isolates bright filament pixels within the box:

```
User draws box [x0,y0,x1,y1]
     └─► Extract region from normalised frame
              └─► mask = frame[y0:y1, x0:x1] > threshold (default 0.5)
                       └─► Save as ground-truth mask
```

This exploits the fact that filaments are typically the brightest structures in the frame after per-frame normalisation.

### 2. Brush Painting (`filament_painter.py`)

The user directly paints pixel-level masks with an adjustable circular brush. Provides finer control for ambiguous or low-contrast frames. Masks are saved as `.npy` arrays in `models/masks/`.

Both tools share the same mask directory and train the same U-Net, so annotations from both can be combined freely.

---

## Per-Frame Normalisation

**Critical design decision:** every frame is normalised independently:

$$\hat{I}[x,y] = \frac{I[x,y] - \min(I)}{\max(I) - \min(I)}$$

This is applied uniformly in all scripts. The motivation is that the model must learn to identify filament _structure_ (shape, geometry, relative contrast), not absolute intensity — which varies with laser power, cell depth, and acquisition conditions. Without per-frame normalisation the model tends to learn intensity-based shortcuts that generalise poorly.

---

## Inference

The trained U-Net is deployed in `filament_segmenter.py`:

1. Load video and normalise each frame
2. Run each frame through the U-Net (no gradients)
3. Apply sigmoid to convert logits → probabilities
4. Threshold at 0.5 → binary mask per frame
5. Count detected pixels; frames with >10 mask pixels are flagged as containing a filament

Output: an interactive 3-column viewer (raw frame | probability heatmap | green mask overlay) with a frame slider.

---

## Scripts Summary

| Script                     | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `visual_tracker_nodiff.py` | Interactive ridge filter viewer with sliders               |
| `filament_boxer.py`        | Draw bounding boxes → auto-threshold masks → train U-Net   |
| `filament_painter.py`      | Brush-paint pixel masks → train U-Net                      |
| `filament_segmenter.py`    | Run trained U-Net on any video                             |
| `filament_labeler.py`      | Frame-level Y/N binary labelling (for classification only) |
| `filament_predictor.py`    | Hybrid ridge+CNN binary predictor with Grad-CAM            |
| `tif_to_mp4.py`            | Convert TIFF stacks to MP4 for external viewing            |

---

## Workflow

```
1. Annotate
   ./env/bin/python scripts/filament_boxer.py tifs/video1.tif tifs/video2.tif
   # Draw boxes on ~20-50 frames across multiple videos, press S after each

2. Train
   # Click [T] Train U-Net in the boxer UI, or re-open and train:
   ./env/bin/python scripts/filament_boxer.py tifs/video1.tif tifs/video2.tif

3. Segment new videos
   ./env/bin/python scripts/filament_segmenter.py tifs/new_video.tif

4. Iterate
   # Add more annotations (boxer or painter), retrain to improve Dice score
```

### Tips for good results

- Annotate frames from **multiple videos** to help the model generalise across conditions
- Include a mix of frames **with and without** filaments — the boxer automatically creates zero-masks for unannotated frames treated as negative examples
- Adjust the **Pixel Thresh** slider when the filament has low contrast; lower threshold = more pixels selected
- A Dice score above **0.70** is typically sufficient for reliable detection; above **0.80** gives precise boundaries

---

## Environment

```bash
# Setup
conda create -p ./env python=3.10
conda activate ./env

# Dependencies
pip install tifffile numpy matplotlib scipy torch torchvision
```

All scripts must be run from the project root (`/University/biohack/`) so relative paths to `tifs/` and `models/` resolve correctly.
