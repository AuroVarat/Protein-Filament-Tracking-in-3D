# Biohack TIFF Viewer

This project contains a Streamlit workflow for hyperstack TIFF review, crop extraction, painter-based mask editing,
and filament analysis with local napari launch support.

## Environment Setup

The project uses a local Conda environment (located in `./env`). For the current app you will want at least:

- `streamlit`
- `numpy`
- `pandas`
- `tifffile`
- `opencv-python`
- `gradio`
- `streamlit-image-coordinates`
- `plotly`
- `napari[all]`

To create the environment from scratch, run from the project root:

```bash
conda create -y -p ./env python=3.10
conda activate ./env
pip install streamlit numpy pandas tifffile opencv-python gradio streamlit-image-coordinates plotly "napari[all]"
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
