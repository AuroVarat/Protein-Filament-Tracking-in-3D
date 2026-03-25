#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from filament_pipeline import inspect_tiff, list_tiff_files, run_inference_many, serialize_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D-only compatibility wrapper over the unified temporal-auto inference pipeline.")
    parser.add_argument("--input_dir", default="tiffs3d", help="Directory containing 3D TIFF files.")
    parser.add_argument("--output_dir", default="results", help="Directory to save masks, cell masks, and tracking CSVs.")
    parser.add_argument("--model_path", default=None, help="Optional override for the 3D temporal-auto checkpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many files.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [f for f in list_tiff_files([args.input_dir]) if inspect_tiff(f).mode == "3d"]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No 3D TIFF files found in {args.input_dir}")
    artifacts = run_inference_many(files, model_path_3d=args.model_path, output_root=args.output_dir)
    payload = [serialize_artifact(item) for item in artifacts if item.mode == "3d"]
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    for item in payload:
        print(item["tracking_csv"])


if __name__ == "__main__":
    main()
