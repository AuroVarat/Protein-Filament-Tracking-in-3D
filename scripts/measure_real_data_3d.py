from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.measure_stats_3d import measure_dataset_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure empirical statistics from 3D manual crop hyperstacks.")
    parser.add_argument("--data-dir", type=Path, default=Path("manual_crops"))
    parser.add_argument("--mask-dir", type=Path, default=Path("masks3d"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_stats_3d"))
    parser.add_argument("--fluorescence-channel", type=int, default=1)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--labeled-only", action="store_true")
    args = parser.parse_args()
    measure_dataset_3d(
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        fluorescence_channel=args.fluorescence_channel,
        max_timepoints=args.max_timepoints,
        labeled_only=args.labeled_only,
    )


if __name__ == "__main__":
    main()
