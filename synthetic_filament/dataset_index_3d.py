from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import tifffile


MASK_3D_RE = re.compile(r"(?P<base>.+)_t(?P<time>\d{4})$", re.IGNORECASE)


def parse_3d_mask_name(path: Path) -> tuple[str, int] | None:
    match = MASK_3D_RE.match(path.stem)
    if match is None:
        return None
    return match.group("base"), int(match.group("time"))


def build_dataset_index_3d(data_dir: Path, mask_dir: Path, fluorescence_channel: int = 1) -> pd.DataFrame:
    raw_paths = sorted(data_dir.glob("*.tif"))
    mask_lookup: dict[tuple[str, int], Path] = {}
    if mask_dir.exists():
        for path in sorted(mask_dir.glob("*.npy")):
            parsed = parse_3d_mask_name(path)
            if parsed is not None:
                mask_lookup[parsed] = path

    rows = []
    for raw_path in raw_paths:
        with tifffile.TiffFile(raw_path) as tf:
            series_shape = tf.series[0].shape
            dtype = tf.series[0].dtype

        if len(series_shape) != 5:
            continue
        t_dim, z_dim, c_dim, y_dim, x_dim = series_shape
        for time_index in range(t_dim):
            mask_path = mask_lookup.get((raw_path.stem, time_index))
            rows.append(
                {
                    "sample_id": f"{raw_path.stem}_t{time_index:04d}",
                    "raw_path": str(raw_path),
                    "raw_stem": raw_path.stem,
                    "time_index": int(time_index),
                    "mask_path": str(mask_path) if mask_path else None,
                    "has_filament_mask": mask_path is not None,
                    "z_dim": int(z_dim),
                    "c_dim": int(c_dim),
                    "y_dim": int(y_dim),
                    "x_dim": int(x_dim),
                    "dtype": str(dtype),
                    "fluorescence_channel": int(fluorescence_channel),
                }
            )
    return pd.DataFrame(rows)


def save_dataset_index_3d(index: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
