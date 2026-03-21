from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.generate_synthetic_3d import generate_dataset_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 3D microscopy crops from measured stats.")
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/real_stats_3d_labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_3d"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filament-prob-per-cell", type=float, default=0.3)
    parser.add_argument("--cell-size-scale", type=float, default=1.2)
    parser.add_argument("--cell-spacing-scale", type=float, default=0.7)
    args = parser.parse_args()
    generate_dataset_3d(
        stats_dir=args.stats_dir,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        filament_prob_per_cell=args.filament_prob_per_cell,
        cell_size_scale=args.cell_size_scale,
        cell_spacing_scale=args.cell_spacing_scale,
    )


if __name__ == "__main__":
    main()
