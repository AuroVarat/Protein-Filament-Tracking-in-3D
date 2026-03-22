#!/usr/bin/env python3
"""
Rank Filaments — Find the "best" filaments across all tracking results.

Ranking criteria:
1. Duration (Number of frames)
2. Size (Average voxels per frame)
3. Flatness (Standard deviation of Z-coordinate)
"""

import os
import pandas as pd

SUMMARY_CSV = "output/filament_dynamics_summary.csv"


def rank_filaments(summary_path: str = SUMMARY_CSV):
    if not os.path.exists(summary_path):
        print(f"Summary CSV not found ({summary_path}). Run filament_dynamics_analysis first.")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        print("Summary CSV contains no filaments.")
        return

    candidates = df[(df["observations"] >= 15) & (df["z_std_um"] < 0.3)].copy()
    if candidates.empty:
        candidates = df.copy()

    flat_long_ranked = candidates.sort_values(by="mean_length_um", ascending=False)

    print("\n📏 Top 5 'PHYSICALLY LONG & FLAT' Filaments 📏\n")
    print(flat_long_ranked.head(5).to_string(index=False))

    best = flat_long_ranked.iloc[0]
    print(f"\n🌟 Best Long & Flat Filament: ID {best['filament_id']} in video '{best['video']}'")
    print(f"   Duration: {best['observations']} frames")
    print(f"   Avg Physical Length: {best['mean_length_um']:.1f} µm")
    print(f"   Z-Stability (std): {best['z_std_um']:.3f} µm (lower is flatter)")

    return best


if __name__ == "__main__":
    rank_filaments()
