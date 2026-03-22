# Biohack — Filament Analysis Pipeline

Clear, coding-focused documentation for a live-cell microscopy project that covers annotation, segmentation, tracking, visualisation, and downstream filament dynamics analysis across 2D and 3D TIFF data.

## Project Overview

This repository contains a Python-based research pipeline for filament analysis in microscopy videos. It includes:

- interactive annotation tools for 2D and 3D TIFF stacks
- PyTorch-based U-Net training and inference code for filament segmentation
- tracking and post-processing scripts that export CSV, PNG, PDF, MP4, and NPZ outputs
- browser and desktop viewers for inspecting masks, segmentations, and temporal behaviour
- analysis scripts for length, morphology, and tracking summaries

## Quick Usage

The shortest end-to-end workflow in this repository is:

1. Annotate filament masks on your TIFF data.
2. Train a segmentation model.
3. Run inference or tracking on unseen data.
4. Run batch analysis to export quantitative outputs.
5. Inspect masks and outputs in the local web viewer or use the analysis scripts for bulk statistics.

Typical commands:

```bash
# 1) Annotate 3D masks in the browser
uv run python scripts/filament_5z_painter_web.py tiffs3d/video.tif

# 2) Train the 3D model
uv run python scripts/train_3d.py tiffs3d/video1.tif tiffs3d/video2.tif

# 3) Run inference / interactive review
uv run python scripts/filament_3d_dashboard.py

# 4) View saved mask TIFFs in the standalone local web viewer
uv run python scripts/filament_mask_web_viewer.py

# 5) Analyse bulk filament statistics from saved masks
uv run python scripts/filament_dynamics_analysis.py --mask-dir results/masks --tif-dir tiffs3d
```

Use the web tools when you want rapid visual inspection of masks and segmentations. Use `filament_dynamics_analysis.py` when you want bulk quantitative outputs such as per-frame measurements, per-filament summaries, filtered tables, and aggregate plot figures in `output/`.

## Frameworks, Libraries, Datasets, and AI Tools Used

### Frameworks and Libraries

The codebase primarily uses:

- Python 3.10+
- PyTorch and Torchvision for model definition, training, and inference
- NumPy for array operations
- SciPy for filtering, morphology, and numerical utilities
- Matplotlib for desktop visualisation and annotation interfaces
- Gradio for browser-based dashboards and labeling tools
- tifffile for microscopy TIFF I/O
- imageio and imageio-ffmpeg for MP4/video export
- pandas for tabular tracking and summary analysis
- Plotly for interactive plotting in web dashboards
- Pillow for image overlays and rendering utilities
- tqdm for progress reporting

Additional libraries used in specific scripts:

- Cellpose for cell segmentation in tracking workflows
- OpenCV (`cv2`) for image-processing utilities in the Cellpose workflow
- scikit-learn for DBSCAN clustering in the Cellpose ROI extraction script

### Datasets and Project Data

This repository works on live-cell microscopy TIFF data stored locally in the project. The main data locations are:

- `tifs2d/`: 2D TIFF inputs for framewise and temporal segmentation workflows
- `tiffs3d/`: 3D or time-resolved Z-stack TIFF inputs used by the 3D filament pipeline
- `tiffs3d_mp4/`: rendered video-style exports derived from 3D TIFF inputs
- `results/`: generated tracking CSVs, analysis summaries, segmentation checks, and exported review artifacts
- `videos/`: output videos and rendered inspection assets
- `models/`: trained model weights and saved masks

The current tracked example data and outputs include files named like `ch20_URA7_URA8_001_*` and `ch20_URA7_URA8_002_*`, which are the microscopy series used throughout the repository’s analysis and tracking outputs.

### AI Tools Used

The AI-assisted development and documentation workflow for this project used:

- Gemini 3.1 Pro
- GPT-5.4

### Compute Hardware Used

This project was developed and tested on:

- NVIDIA GeForce RTX 5090 for CUDA-based training and inference on Linux
- Apple M2 Pro using PyTorch MPS acceleration on macOS

All training and inference scripts use automatic device selection with the priority:

- CUDA
- MPS
- CPU

## Environment Installation

### Requirements

- Python 3.10 or newer
- `uv` for dependency and environment management

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and sync the project environment

From the repository root:

```bash
uv sync
```

This installs the project dependencies declared in [`pyproject.toml`](/home/auro/biohack/biohack/pyproject.toml) and creates the local virtual environment in `.venv/`.

### Run commands inside the environment

Use `uv run` to execute any project script without manually activating the virtual environment:

```bash
uv run python scripts/train_3d.py tiffs3d/video1.tif tiffs3d/video2.tif
```

If you prefer a shell with the environment activated:

```bash
source .venv/bin/activate
python scripts/train_3d.py tiffs3d/video1.tif tiffs3d/video2.tif
```

---

## 🦠 NEW: 3D Filament Pipeline (Z-Stack TIFFs)

The 3D pipeline targets $(T, Z, H, W)$ multi-page TIFFs using anisotropic Z-pooling to handle shallow (e.g., $Z=5$) depths. All 3D scripts automatically extract the 2nd channel ($C=1$) from 5D data and perform per-slice 2D normalizations. Detailed specifics are found in `FILAMENT_3D_ML.md`.

### 1. Annotate 3D Masks
**For Remote Headless Servers (Recommended):**
Generates a side-by-side panoramic web interface that stitches all 5 Z-planes horizontally for easy brush-painting.
```bash
uv run python scripts/filament_5z_painter_web.py tiffs3d/video.tif
```
(Access via browser at `http://localhost:7860`. Click **Save 3D Mask** to save)

**For Local Machines with GUI:**
Interactive 3-panel orthogonal viewer.
```bash
uv run python scripts/filament_3d_viewer.py tiffs3d/video.tif
```
(Use XYZ sliders, paint on any panel, press `[S]` to save)

### 2. Train 3D U-Net
Loads all explicitly annotated frames globally and drops empty ones, using 3D-aware augmentations (XYZ flips, XY rotations, etc).
```bash
uv run python scripts/train_3d.py tiffs3d/video1.tif tiffs3d/video2.tif
```
Saves the trained 3D architecture to `models/filament_unet3d.pt`.

### 3. Explore 3D Results
**Interactive Results Dashboard (Recommended):**
Explore results interactively with toggleable mask overlays, a 2.5D volumetric stack, and Fiji-style orthogonal MIPs across time.
```bash
uv run python scripts/filament_3d_dashboard.py
```

**Generate 3D Inference MP4:**
Renders a 3-row video: raw panorama, mask panorama, and [2.5D stack + Ortho-Grid Grid].
```bash
uv run python scripts/filament_3d_mp4.py tiffs3d/video.tif
```
(Saves as `tiffs3d/video_inference.mp4`)

**Interactive Local Viewer (Inference):**
Runs the model on a volume and opens the orthogonal viewer with model predictions overlaid.
```bash
uv run python scripts/filament_3d_segmenter.py tiffs3d/video.tif
```

---

## Setup

### macOS (Apple Silicon — MPS)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (PyTorch with MPS support)
uv sync
```

### Linux (NVIDIA GPU — RTX 5090 / Blackwell Support)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (PyTorch Nightly cu128 for Blackwell support)
uv sync
```

> **Note for Blackwell/sm_120**: The `pyproject.toml` is configured to use **PyTorch Nightly (cu128)** and allows pre-releases. This is required for the RTX 5090, as stable wheels currently lack the necessary binary kernels for this architecture. Standard `uv sync` will handle this automatically.

Device priority in all scripts: **CUDA → MPS → CPU** (auto-detected).

Hardware used during development and testing:

- **Linux CUDA machine**: NVIDIA GeForce RTX 5090
- **macOS machine**: Apple M2 Pro using MPS

---

## Annotation

### 1. Draw bounding boxes (recommended — fastest)

Click & drag on the right panel to surround each filament. The script auto-thresholds bright pixels inside the box to create the mask.

```bash
uv run python scripts/filament_boxer.py tifs/video1.tif tifs/video2.tif
```

Keys: `S` Save box · `C` Clear · `←→` Navigate frames · `T` Train (raw) · `R` Train +Ridge

### 2. Brush-paint pixels (finer control)

```bash
uv run python scripts/filament_painter.py tifs/video1.tif tifs/video2.tif
```

Keys: `S` Save · `C` Clear · `E` Toggle eraser · `←→` Navigate

All masks are saved to `models/masks/` and shared between both tools.

---

## Training

### Train the 1-channel model (raw frame only)

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from filament_boxer import train_unet
train_unet(['tifs/video1.tif', 'tifs/video2.tif'])
"
```

Saves to `models/filament_unet.pt`.

### Train the 2-channel model (raw + ridge filter)

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from filament_boxer import train_unet_ridge
train_unet_ridge(['tifs/video1.tif', 'tifs/video2.tif'])
"
```

Saves to `models/filament_unet_ridge.pt`.

### Train both in one go

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from filament_boxer import train_unet, train_unet_ridge
tifs = ['tifs/ch20_URA7_URA8_002-crop4.tif', 'tifs/ch20_URA7_URA8_002-crop5.tif']
train_unet(tifs)
train_unet_ridge(tifs)
"
```

### Train without negative (background) frames

By default the training includes **all** video frames — annotated frames use your drawn masks and unannotated frames use an all-zero mask as negative examples. To train on annotated frames only (old behaviour), open `filament_boxer.py` and change `_load_paired` to skip unannotated frames:

```python
# In _load_paired(), change:
mask = mask_lookup.get(key, np.zeros(frame.shape, dtype=np.float32))
# To:
if key not in mask_lookup: continue  # skip unannotated frames
mask = mask_lookup[key]
```

---

## Hyperparameter Search (pos_weight)

Finds the best class-imbalance weight for the BCE loss by splitting your annotated masks 80/20 train/val and trying `pos_weight ∈ {1, 2, 5, 10, 20, 50}`.

```bash
# 1-channel model
uv run python scripts/pos_weight_search.py tifs/video1.tif tifs/video2.tif

# 2-channel model
uv run python scripts/pos_weight_search.py tifs/video1.tif tifs/video2.tif --ridge
```

The script prints the best `pos_weight` value and tells you which line to edit in `filament_boxer.py`.

---

## Inference

### View segmentation on a video

```bash
# 1-channel model (raw only)
uv run python scripts/filament_segmenter.py tifs/ch20_URA7_URA8_002-crop6.tif

# 2-channel model (raw + ridge filter) — 4-column viewer
uv run python scripts/filament_segmenter.py tifs/ch20_URA7_URA8_002-crop6.tif --ridge
```

### Batch analysis + export CSV and summary figure

```bash
# Analyse all videos in one run
uv run python scripts/filament_analyser.py tifs/*.tif

# With ridge model
uv run python scripts/filament_analyser.py tifs/*.tif --ridge
```

Outputs:

- `results/filament_analysis.csv` — per-frame: `time_min`, `length_um`, `width_um`, `area_um2`, `aspect_ratio`, `angle_deg`
- `results/filament_summary.pdf` — detection timeline, length/area/angle distributions, per-video lifetime and frequency

Physical parameters (pixel size, frame interval) are at the top of `filament_analyser.py`.

---

## Other Scripts

| Script                     | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| `show_tiff.py`             | Simple frame-by-frame viewer                  |
| `frame_diff.py`            | Frame-to-frame absolute difference            |
| `frame_matrix.py`          | Pairwise N×N difference matrix                |
| `ddm_analysis.py`          | Differential Dynamic Microscopy               |
| `burst_tracker.py`         | Time-resolved spectrogram tracker             |
| `visual_tracker_nodiff.py` | Interactive ridge filter viewer with sliders  |
| `tif_to_mp4.py`            | Convert TIFF stack to MP4                     |
| `filament_labeler.py`      | Frame-level Y/N binary labelling (binary CNN) |
| `filament_predictor.py`    | Grad-CAM heatmap viewer (binary CNN)          |

---

## Headless / Server Use

The annotation and segmentation viewer scripts require a display. On a headless Linux server:

```bash
# Option 1: X11 forwarding over SSH
ssh -X user@server

# Option 2: Virtual display
sudo apt install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

The **analyser** (`filament_analyser.py`) is display-free — it only saves files and runs fine headlessly.
