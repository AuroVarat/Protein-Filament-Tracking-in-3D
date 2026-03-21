from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.validate_synthetic_3d import compare_real_and_synthetic_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real and synthetic 3D microscopy distributions.")
    parser.add_argument("--real-stats-dir", type=Path, default=Path("outputs/real_stats_3d_labeled"))
    parser.add_argument("--synth-dir", type=Path, default=Path("outputs/synthetic_3d"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_3d"))
    args = parser.parse_args()
    compare_real_and_synthetic_3d(args.real_stats_dir, args.synth_dir, args.output_dir)


if __name__ == "__main__":
    main()
