# Biohack TIFF Viewer

This project contains a Python script to load and visualize multi-frame TIFF images.

## TDA Timeline Website

The repository now also includes a static website in `site/` that turns the frame-by-frame filament analysis from `tda.ipynb` into an interactive UI.

### Regenerate the web dataset

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/export_tda_web_data.py
```

This analyzes the default stack at `2D/ch20_URA7_URA8_001-crop1.tif`, exports `site/data/tda_timeline.json`, and writes normalized PNG frames to `site/assets/frames/`.

### Regenerate the 3D web dataset

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/export_tda_web_data_3d.py
```

By default this follows the notebook's 3D setup on `3D/ch20_URA7_URA8_002_hyperstack_crop_45.tif`, channel `1`, and exports `site/data/tda_3d_volumes.json` plus orthographic projection PNGs under `site/assets/3d/`.

You can change the source files or channels, for example:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/export_tda_web_data_3d.py \
  --input 3D/ch20_URA7_URA8_001_hyperstack_crop_01.tif \
  --channels 0 1
```

### View the site

Serve the `site/` directory with any static file server. For example:

```bash
python3 -m http.server 8123 --directory site
```

Then open:

- `http://127.0.0.1:8123/` for the 2D timeline view
- `http://127.0.0.1:8123/3d.html` for the 3D explorer

## Environment Setup

The project uses a local Conda environment (located in `./env`) containing all necessary dependencies (`tifffile`, `matplotlib`, `numpy`).

To create the environment from scratch, run from the project root:

```bash
conda create -y -p ./env python=3.10 tifffile matplotlib numpy
```

## Usage

You can run the visualization script using the environment's Python executable directly:

```bash
./env/bin/python show_tiff.py
```

Alternatively, you can activate the local environment and then run the script normally:

```bash
conda activate ./env
python show_tiff.py
```
