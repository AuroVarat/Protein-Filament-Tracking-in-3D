#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from filament_pipeline import (
    annotation_summary,
    inspect_tiff,
    list_tiff_files,
    serialize_train_result,
    train_temporal_auto,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified temporal-auto filament trainer.")
    parser.add_argument("inputs", nargs="+", help="TIFF files or directories.")
    parser.add_argument(
        "--mode",
        choices=["2d", "3d", "all"],
        default="all",
        help="Train only 2D, only 3D, or both from the discovered inputs.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per model.")
    parser.add_argument(
        "--promote-active",
        action="store_true",
        help="Copy the newly trained checkpoint to the default active checkpoint path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output instead of plain text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list_tiff_files(args.inputs)
    if not files:
        raise SystemExit("No TIFF files found from the supplied inputs.")

    files_by_mode: dict[str, list[str]] = {"2d": [], "3d": []}
    for filepath in files:
        info = inspect_tiff(filepath)
        files_by_mode[info.mode].append(filepath)

    requested_modes = ["2d", "3d"] if args.mode == "all" else [args.mode]
    results = []
    for mode in requested_modes:
        mode_files = files_by_mode[mode]
        if not mode_files:
            continue
        result = train_temporal_auto(
            mode=mode,
            files=mode_files,
            epochs=args.epochs,
            promote_to_active=args.promote_active,
        )
        results.append(serialize_train_result(result))

    if not results:
        raise SystemExit("No matching TIFFs were found for the requested mode.")

    if args.json:
        payload = {
            "inputs": files,
            "summary": annotation_summary(files),
            "results": results,
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    for result in results:
        print(f"[{result['mode']}] checkpoint={result['checkpoint_path']}")
        print(f"  active={result['active_checkpoint_path']}")
        print(f"  promoted={result['promoted_to_active']}")
        print(f"  sequences={result['num_sequences']} annotations={result['num_annotations']} epochs={result['epochs']}")
        print(f"  log={result['log_path']}")


if __name__ == "__main__":
    main()
