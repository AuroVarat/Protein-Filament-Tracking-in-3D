# Synthetic Filament Microscopy Pipeline

This repository now contains a modular Python pipeline for:

- indexing raw TIFF stacks and frame-level masks
- measuring empirical real-data statistics
- generating synthetic low-SNR 2D widefield microscopy images with cell and filament masks
- comparing real and synthetic distributions
- optionally training a small synthetic-only segmentation baseline

## Dependencies

Core:

```bash
pip install numpy scipy pandas matplotlib scikit-image tifffile
```

Interactive annotation:

```bash
pip install ultralytics pillow
```

Optional training:

```bash
pip install torch
```

## Expected data layout

Put TIFF files under `tifs/`.

Example raw stack:

```text
tifs/ch20_URA7_URA8_001-crop1.tif
```

Example frame-level masks for frame `f0005`:

```text
tifs/ch20_URA7_URA8_001-crop1_f0005_mask.tif
tifs/ch20_URA7_URA8_001-crop1_f0005_cellmask_1.tif
```

The indexer treats masks as annotations for a specific frame inside the raw stack.

## Run the pipeline

Create or edit masks interactively with SAM2 point prompts:

```bash
python scripts/annotate_sam2_masks.py --data-dir tifs --model sam2_b.pt --device cuda:0
```

Notes:

- Left click adds a positive point.
- Right click adds a negative point.
- Use the `Low %` and `High %` sliders to adjust display contrast normalization interactively.
- Switch between `Filament` and `Cell` to save the two mask types separately.
- `Segment` runs SAM2 on the current normalized frame preview.
- `Save Mask` writes:
  - `image_name_f0005_mask_1.tif`, `image_name_f0005_mask_2.tif`, ... for filament objects
  - `image_name_f0005_cellmask_1.tif`, `image_name_f0005_cellmask_2.tif`, ... for cell objects
- Use `Object #`, `Prev Obj`, `Next Obj`, and `New Obj` to annotate multiple filaments or multiple cells in the same frame.
- Arrow keys move through time. `s` segments. `w` saves.
- The annotator assumes SAM2 runs on GPU and will fail fast if CUDA is unavailable.

Measure real-data statistics:

```bash
python scripts/measure_real_data.py --data-dir tifs --output-dir outputs/real_stats
```

Generate synthetic data from measured statistics:

```bash
python scripts/make_synthetic_dataset.py --stats-dir outputs/real_stats --output-dir outputs/synthetic --count 256 --split
```

Compare real and synthetic measurements:

```bash
python scripts/compare_real_vs_synth.py --real-stats-dir outputs/real_stats --synth-dir outputs/synthetic --output-dir outputs/comparison
```

Optional baseline training on synthetic data:

```bash
python scripts/train_synth_baseline.py --synth-dir outputs/synthetic --output-dir outputs/training --epochs 10
```

## Outputs

- `outputs/real_stats/`
  - `dataset_index.csv`
  - `sample_measurements.csv`
  - `stack_measurements.csv`
  - `stats_summary.json`
  - histogram plots
- `outputs/synthetic/`
  - synthetic images
  - synthetic cell masks
  - synthetic filament masks
  - per-sample metadata JSON
  - `manifest.csv`
- `outputs/comparison/`
  - overlay histograms
  - Wasserstein-based comparison tables

## Notes

- If mask pairs are not present yet, the measurement phase still records stack-level intensity statistics and indexing metadata.
- Meaningful morphology calibration requires frame-level filament and cell masks.
- Distribution choices and v1 simplifications are documented in `MODELLING_DECISIONS.md`.
