from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.measure_stats import measure_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure empirical statistics from TIFF microscopy data.")
    parser.add_argument("--data-dir", type=Path, default=Path("tifs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_stats"))
    parser.add_argument("--filament-dilation-px", type=int, default=0)
    args = parser.parse_args()
    measure_dataset(args.data_dir, args.output_dir, filament_dilation_px=args.filament_dilation_px)


if __name__ == "__main__":
    main()
