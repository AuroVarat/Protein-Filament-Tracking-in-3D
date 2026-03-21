from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .utils import ensure_dir


def read_tiff(path: Path) -> np.ndarray:
    return tifffile.imread(path)


def read_stack_frame(path: Path, frame_index: int) -> np.ndarray:
    data = read_tiff(path)
    if data.ndim == 2:
        if frame_index not in (0, 1):
            raise IndexError(f"Requested frame {frame_index} from 2D TIFF: {path}")
        return data
    if frame_index >= data.shape[0]:
        raise IndexError(f"Frame {frame_index} out of bounds for {path} with shape {data.shape}")
    return data[frame_index]


def write_tiff(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    tifffile.imwrite(path, image)
