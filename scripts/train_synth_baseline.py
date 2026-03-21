from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.train_baseline import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small U-Net baseline on synthetic data.")
    parser.add_argument("--synth-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_baseline(args.synth_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
