from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .io_utils import read_tiff
from .measure_stats import _measure_sample
from .utils import ensure_dir, save_json


def measure_synthetic_dataset(synth_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(synth_dir / "manifest.csv")
    rows = []
    for row in manifest.itertuples(index=False):
        image = read_tiff(Path(row.image_path))
        cell_mask = read_tiff(Path(row.cell_mask_path))
        filament_mask = read_tiff(Path(row.filament_mask_path))
        rows.append(_measure_sample(image, cell_mask, filament_mask, row.sample_id))
    df = pd.DataFrame(rows)
    df.to_csv(synth_dir / "synthetic_measurements.csv", index=False)
    return df


def _overlay_plot(real: np.ndarray, synth: np.ndarray, title: str, path: Path) -> None:
    real = real[np.isfinite(real)]
    synth = synth[np.isfinite(synth)]
    if real.size == 0 or synth.size == 0:
        return
    lo = min(real.min(), synth.min())
    hi = max(real.max(), synth.max())
    bins = np.linspace(lo, hi, 30)
    plt.figure(figsize=(5, 4))
    plt.hist(real, bins=bins, density=True, alpha=0.55, label="real", color="#1f77b4")
    plt.hist(synth, bins=bins, density=True, alpha=0.55, label="synthetic", color="#ff7f0e")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compare_real_and_synthetic(real_stats_dir: Path, synth_dir: Path, output_dir: Path) -> pd.DataFrame:
    ensure_dir(output_dir)
    real_df = pd.read_csv(real_stats_dir / "sample_measurements.csv")
    synth_df = measure_synthetic_dataset(synth_dir)

    metrics = [
        "bg_mean",
        "bg_std",
        "cell_area",
        "filament_length",
        "filament_width_mean",
        "filament_intensity_mean",
        "filament_minus_background",
        "filament_minus_diffuse",
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
                "real_median": float(np.median(real)),
                "synth_median": float(np.median(synth)),
                "real_p95": float(np.percentile(real, 95)),
                "synth_p95": float(np.percentile(synth, 95)),
                "wasserstein": float(wasserstein_distance(real, synth)),
            }
        )
        _overlay_plot(real, synth, metric, output_dir / f"{metric}_overlay.png")

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "comparison_metrics.csv", index=False)
    save_json({"rows": report.to_dict(orient="records")}, output_dir / "comparison_metrics.json")
    return report
