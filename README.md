# Biohack TIFF Viewer

This project contains a Python script to load and visualize multi-frame TIFF images.

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
