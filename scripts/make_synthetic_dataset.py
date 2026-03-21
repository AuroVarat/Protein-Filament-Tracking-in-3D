from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.generate_synthetic import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic microscopy filament dataset.")
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/real_stats"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    generate_dataset(args.stats_dir, args.output_dir, args.count, args.seed, args.split)


if __name__ == "__main__":
    main()
