#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from filament_pipeline import inspect_tiff, list_tiff_files, run_inference_many, serialize_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified temporal-auto filament inference pipeline.")
    parser.add_argument("inputs", nargs="+", help="TIFF files or directories.")
    parser.add_argument("--model-2d", default=None, help="Optional override for the 2D temporal-auto checkpoint.")
    parser.add_argument("--model-3d", default=None, help="Optional override for the 3D temporal-auto checkpoint.")
    parser.add_argument("--output-root", default="results", help="Root directory for masks, cell masks, and tracking CSVs.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list_tiff_files(args.inputs)
    if not files:
        raise SystemExit("No TIFF files found from the supplied inputs.")

    artifacts = run_inference_many(
        files,
        model_path_2d=args.model_2d,
        model_path_3d=args.model_3d,
        output_root=args.output_root,
    )
    payload = {
        "inputs": files,
        "modes": {filepath: inspect_tiff(filepath).mode for filepath in files},
        "artifacts": [serialize_artifact(artifact) for artifact in artifacts],
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    for artifact in payload["artifacts"]:
        print(f"[{artifact['mode']}] {artifact['source_path']}")
        print(f"  mask={artifact['mask_tiff']}")
        print(f"  cell_mask={artifact['cell_mask_tiff']}")
        print(f"  tracking_csv={artifact['tracking_csv']}")


if __name__ == "__main__":
    main()
