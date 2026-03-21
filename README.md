# Biohack — Filament Analysis Pipeline

A suite of Python scripts for loading, visualising, and ML-based segmentation of filamentous structures in live-cell TIFF microscopy videos.

---

## Setup

### macOS (Apple Silicon — MPS)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (PyTorch with MPS support)
# First comment out the [tool.uv.sources] block in pyproject.toml, then:
uv sync
```

### Linux (NVIDIA GPU — CUDA 12.4)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Keep the [tool.uv.sources] block in pyproject.toml enabled, then:
uv sync
```

> The `[tool.uv.sources]` block in `pyproject.toml` points `torch`/`torchvision` to the PyTorch CUDA 12.4 index.
> Comment it out on macOS — the default PyPI wheel already includes MPS.

Device priority in all scripts: **CUDA → MPS → CPU** (auto-detected).

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
uv run python scripts/filament_segmenter.py tifs/new_video.tif

# 2-channel model (raw + ridge filter) — 4-column viewer
uv run python scripts/filament_segmenter.py tifs/new_video.tif --ridge
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
