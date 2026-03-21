from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_filament.annotate_gui import launch_annotator


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SAM2 mask annotator for TIFF stacks.")
    parser.add_argument("--data-dir", type=Path, default=Path("tifs"))
    parser.add_argument("--model", type=str, default="sam2_b.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    launch_annotator(args.data_dir, args.model, device=args.device)


if __name__ == "__main__":
    main()
