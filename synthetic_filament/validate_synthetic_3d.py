from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import wasserstein_distance

from .measure_stats_3d import measure_timepoint
from .utils import ensure_dir


def measure_synthetic_dataset_3d(synth_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(synth_dir / "manifest.csv")
    rows = []
    for row in manifest.itertuples(index=False):
        image = tifffile.imread(Path(row.image_path)).astype(np.float32)
        filament_mask = tifffile.imread(Path(row.filament_mask_path)).astype(np.float32)
        rows.append(measure_timepoint(image, row.sample_id, filament_mask))
    df = pd.DataFrame(rows)
    df.to_csv(synth_dir / "synthetic_measurements_3d.csv", index=False)
    return df


def _overlay_plot(real: np.ndarray, synth: np.ndarray, title: str, path: Path) -> None:
    real = real[np.isfinite(real)]
    synth = synth[np.isfinite(synth)]
    if real.size == 0 or synth.size == 0:
        return
    lo = min(real.min(), synth.min())
    hi = max(real.max(), synth.max())
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 30)
    plt.figure(figsize=(5, 4))
    plt.hist(real, bins=bins, density=True, alpha=0.55, label="real", color="#1f77b4")
    plt.hist(synth, bins=bins, density=True, alpha=0.55, label="synthetic", color="#ff7f0e")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compare_real_and_synthetic_3d(real_stats_dir: Path, synth_dir: Path, output_dir: Path) -> pd.DataFrame:
    ensure_dir(output_dir)
    real_df = pd.read_csv(real_stats_dir / "timepoint_measurements_3d.csv")
    synth_df = measure_synthetic_dataset_3d(synth_dir)

    metrics = [
        "background_mean",
        "background_std",
        "cell_count_auto",
        "cell_volume_um3_mean",
        "cell_diffuse_mean",
        "filament_volume_um3",
        "filament_length_um",
        "filament_width_um_mean",
        "filament_intensity_mean",
        "filament_minus_host_diffuse",
        "filament_budget_fraction",
        "filament_z_occupancy",
    ]

    rows = []
    for metric in metrics:
        if metric not in real_df or metric not in synth_df:
            continue
        real = real_df[metric].dropna().to_numpy(dtype=float)
        synth = synth_df[metric].dropna().to_numpy(dtype=float)
        if len(real) == 0 or len(synth) == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "real_mean": float(np.mean(real)),
                "synth_mean": float(np.mean(synth)),
                "real_median": float(np.median(real)),
                "synth_median": float(np.median(synth)),
                "wasserstein": float(wasserstein_distance(real, synth)),
            }
        )
        _overlay_plot(real, synth, metric, output_dir / f"{metric}_overlay.png")

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "comparison_metrics_3d.csv", index=False)
    with (output_dir / "comparison_metrics_3d.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": report.to_dict(orient="records")}, handle, indent=2)
    return report
