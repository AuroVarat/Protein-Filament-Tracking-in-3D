#!/usr/bin/env python3
"""
Quantify filament geometry directly from labeled mask TIFFs.

The filament mask TIFFs provide the voxel geometry for each tracked filament ID.
This script measures each frame-level filament observation from those voxels,
then summarizes per-track dynamics across time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile

from plot_filament_dynamics import run_plot as run_plot_filaments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify filament geometry and dynamics from mask TIFFs."
    )
    parser.add_argument(
        "--mask-dir",
        default="results/masks",
        help="Directory with labeled filament mask TIFFs.",
    )
    parser.add_argument(
        "--tif-dir",
        default="tiffs3d",
        help="Directory with original TIFF videos for true intensity measurements.",
    )
    parser.add_argument(
        "--summary-csv",
        default="output/filament_dynamics_summary.csv",
        help="Track-level summary output.",
    )
    parser.add_argument(
        "--processed-csv",
        default="output/processed_filaments.csv",
        help="Track-level summary for filaments that pass the filters.",
    )
    parser.add_argument(
        "--frame-measurements-csv",
        default="output/filament_frame_measurements.csv",
        help="Per-frame filament measurements derived from TIFF voxels.",
    )
    parser.add_argument(
        "--pixel-size-xy",
        "--pixel-size-um",
        dest="pixel_size_xy",
        type=float,
        default=0.3,
        help="Lateral pixel edge length in micrometers (XY).",
    )
    parser.add_argument(
        "--z-spacing-um",
        type=float,
        default=0.3,
        help="Separation between consecutive z-planes in micrometers.",
    )
    parser.add_argument(
        "--frame-interval-min",
        type=float,
        default=15.0,
        help="Time interval between frames in minutes.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=2,
        help="Minimum number of accepted frame observations per filament.",
    )
    parser.add_argument(
        "--size-tail-quantile",
        type=float,
        default=0.1,
        help="Lower-tail percentile used to reject short shallow detections.",
    )
    parser.add_argument(
        "--min-z-planes",
        type=int,
        default=3,
        help="Minimum distinct z-planes for a frame-level detection to count.",
    )
    parser.add_argument(
        "--length-estimate-csv",
        default="output/filament_length_estimate.csv",
        help="Per-video robust-length estimate with error bars.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="output",
        help="Directory where plot PNGs will be stored.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def collect_mask_files(mask_dir: Path) -> Iterable[Path]:
    yield from sorted(mask_dir.glob("*_mask.tif"))


def compute_length_metrics(coords_um: np.ndarray) -> tuple[float, float, float]:
    if coords_um.shape[0] < 2:
        return 0.0, 0.0, 0.0

    centered = coords_um - coords_um.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    principal_axis = evecs[:, int(np.argmax(evals))]
    axis_norm = np.linalg.norm(principal_axis)
    if axis_norm == 0.0:
        principal_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        principal_axis = principal_axis / axis_norm

    projections = centered @ principal_axis
    p5, p95 = np.percentile(projections, [5, 95])
    trimmed_span = max(float(p95 - p5), 0.0)
    robust_length_um = trimmed_span / 0.9 if trimmed_span > 0 else 0.0
    span_length_um = float(projections.max() - projections.min())

    unique_proj = np.unique(np.round(projections, 6))
    if unique_proj.size > 1:
        proj_steps = np.diff(unique_proj)
        positive_steps = proj_steps[proj_steps > 0]
        bin_width_um = float(np.median(positive_steps)) if positive_steps.size else 0.3
    else:
        bin_width_um = 0.3
    bin_width_um = max(bin_width_um, 0.15)

    start_proj = float(projections.min())
    end_proj = float(projections.max())
    if end_proj > start_proj:
        edges = np.arange(start_proj, end_proj + bin_width_um, bin_width_um)
    else:
        edges = np.array([start_proj, start_proj + bin_width_um], dtype=np.float64)

    centers: list[np.ndarray] = []
    for idx in range(len(edges) - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == len(edges) - 2:
            in_bin = (projections >= lo) & (projections <= hi)
        else:
            in_bin = (projections >= lo) & (projections < hi)
        if not np.any(in_bin):
            continue
        centers.append(coords_um[in_bin].mean(axis=0))

    if len(centers) >= 2:
        centerline = np.vstack(centers)
        deltas = np.diff(centerline, axis=0)
        centerline_length_um = float(np.linalg.norm(deltas, axis=1).sum())
    else:
        centerline_length_um = span_length_um

    arc_length_um = max(centerline_length_um, span_length_um)
    return robust_length_um, span_length_um, arc_length_um


def load_frame_measurements(
    mask_dir: Path,
    tif_dir: Path,
    pixel_size_xy: float,
    z_spacing_um: float,
) -> pd.DataFrame:
    records: list[dict] = []
    for mask_path in collect_mask_files(mask_dir):
        video_name = mask_path.stem.replace("_mask", "")
        tif_path = tif_dir / f"{video_name}.tif"
        if not tif_path.exists():
            print(f"Skipping {video_name}: missing original TIFF at {tif_path}")
            continue

        mask_stack = tifffile.imread(mask_path)
        img = tifffile.imread(tif_path).astype(np.float32)
        if np.count_nonzero(mask_stack) == 0:
            print(f"Skipping {video_name}: filament mask TIFF is empty.")
            continue
        if img.ndim == 4:
            img = img[np.newaxis]
        if img.ndim != 5:
            print(f"Skipping {video_name}: expected 5D TIFF, got {img.shape}")
            continue

        raw_fil = img[:, :, 1, :, :]
        if mask_stack.shape != raw_fil.shape:
            print(
                f"Skipping {video_name}: mask shape {mask_stack.shape} "
                f"does not match channel-1 TIFF shape {raw_fil.shape}"
            )
            continue

        for frame in range(mask_stack.shape[0]):
            frame_mask = mask_stack[frame]
            frame_raw = raw_fil[frame]
            filament_ids = [int(fid) for fid in np.unique(frame_mask) if fid != 0]
            for filament_id in filament_ids:
                coords_zyx = np.argwhere(frame_mask == filament_id)
                if coords_zyx.size == 0:
                    continue

                zyx_um = coords_zyx.astype(np.float64)
                zyx_um[:, 0] *= z_spacing_um
                zyx_um[:, 1] *= pixel_size_xy
                zyx_um[:, 2] *= pixel_size_xy
                coords_xyz_um = zyx_um[:, [2, 1, 0]]

                robust_length_um, span_length_um, arc_length_um = compute_length_metrics(
                    coords_xyz_um
                )
                intensities = frame_raw[frame_mask == filament_id]

                records.append(
                    {
                        "video": video_name,
                        "frame": int(frame),
                        "filament_id": filament_id,
                        "voxel_count": int(coords_zyx.shape[0]),
                        "volume_um3": float(
                            coords_zyx.shape[0] * pixel_size_xy * pixel_size_xy * z_spacing_um
                        ),
                        "z_plane_count": int(np.unique(coords_zyx[:, 0]).size),
                        "centroid_x_um": float(coords_xyz_um[:, 0].mean()),
                        "centroid_y_um": float(coords_xyz_um[:, 1].mean()),
                        "centroid_z_um": float(coords_xyz_um[:, 2].mean()),
                        "z_spread_um": float(coords_xyz_um[:, 2].max() - coords_xyz_um[:, 2].min()),
                        "robust_length_um": float(robust_length_um),
                        "span_length_um": float(span_length_um),
                        "arc_length_um": float(arc_length_um),
                        "mean_intensity": float(intensities.mean()),
                        "sum_intensity": float(intensities.sum()),
                        "max_intensity": float(intensities.max()),
                    }
                )

    return pd.DataFrame.from_records(records)


def filter_frame_measurements(
    frame_df: pd.DataFrame,
    size_tail_quantile: float,
    min_z_planes: int,
) -> tuple[pd.DataFrame, float]:
    if frame_df.empty:
        return frame_df, 0.0

    size_tail_quantile = float(np.clip(size_tail_quantile, 0.0, 1.0))
    size_threshold = float(frame_df["robust_length_um"].quantile(size_tail_quantile))
    keep = ~(
        (frame_df["robust_length_um"] <= size_threshold)
        & (frame_df["z_plane_count"] < min_z_planes)
    )
    return frame_df.loc[keep].copy(), size_threshold


def summarize_tracks(
    frame_df: pd.DataFrame,
    frame_interval_min: float,
    min_observations: int,
) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()

    records: list[dict] = []
    grouped = frame_df.groupby(["video", "filament_id"], sort=True)
    for (video, filament_id), group in grouped:
        group = group.sort_values("frame").copy()
        if len(group) < min_observations:
            continue

        frames = group["frame"].to_numpy()
        centroids = group[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy()
        if len(frames) > 1:
            delta_frames = np.diff(frames)
            delta_coords = np.diff(centroids, axis=0)
            distances = np.linalg.norm(delta_coords, axis=1)
            durations = delta_frames * frame_interval_min
            speeds = np.divide(
                distances,
                durations,
                out=np.zeros_like(distances),
                where=durations > 0,
            )
            path_length_um = float(distances.sum())
        else:
            speeds = np.array([], dtype=np.float64)
            path_length_um = 0.0

        lifetime_min = float((frames[-1] - frames[0]) * frame_interval_min) if len(frames) > 1 else 0.0
        net_displacement_um = float(np.linalg.norm(centroids[-1] - centroids[0])) if len(frames) > 1 else 0.0
        directionality = net_displacement_um / path_length_um if path_length_um > 0 else 0.0

        length_seq = group["robust_length_um"].to_numpy()
        length_change_rate = (
            float((length_seq[-1] - length_seq[0]) / lifetime_min) if lifetime_min > 0 else 0.0
        )

        records.append(
            {
                "video": video,
                "filament_id": int(filament_id),
                "observations": int(len(group)),
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "lifetime_min": lifetime_min,
                "mean_length_um": float(group["robust_length_um"].mean()),
                "max_length_um": float(group["robust_length_um"].max()),
                "mean_span_length_um": float(group["span_length_um"].mean()),
                "mean_arc_length_um": float(group["arc_length_um"].mean()),
                "length_change_rate_um_per_min": length_change_rate,
                "arc_trajectory_um": float(group["arc_length_um"].sum()),
                "net_displacement_um": net_displacement_um,
                "directionality": directionality,
                "mean_speed_um_per_min": float(speeds.mean()) if speeds.size else 0.0,
                "max_speed_um_per_min": float(speeds.max()) if speeds.size else 0.0,
                "speed_std_um_per_min": float(speeds.std(ddof=0)) if speeds.size else 0.0,
                "mean_z_plane_count": float(group["z_plane_count"].mean()),
                "max_z_plane_count": int(group["z_plane_count"].max()),
                "z_spread_um": float(group["z_spread_um"].max()),
                "z_center_um": float(group["centroid_z_um"].mean()),
                "mean_intensity": float(group["mean_intensity"].mean()),
                "sum_intensity": float(group["sum_intensity"].sum()),
                "max_intensity": float(group["max_intensity"].max()),
                "start_x_um": float(centroids[0, 0]),
                "start_y_um": float(centroids[0, 1]),
                "start_z_um": float(centroids[0, 2]),
                "end_x_um": float(centroids[-1, 0]),
                "end_y_um": float(centroids[-1, 1]),
                "end_z_um": float(centroids[-1, 2]),
            }
        )

    return pd.DataFrame.from_records(records)


def build_length_estimate(filtered_frames: pd.DataFrame, size_threshold: float) -> pd.DataFrame:
    if filtered_frames.empty:
        return pd.DataFrame()

    grouped = (
        filtered_frames.groupby("video")["arc_length_um"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_arc_length_um", "std": "std_arc_length_um"})
    )
    grouped["std_arc_length_um"] = grouped["std_arc_length_um"].fillna(0.0)
    grouped["sem"] = grouped["std_arc_length_um"] / np.sqrt(grouped["count"])
    grouped["ci95"] = 1.96 * grouped["sem"]
    grouped["length_threshold_um"] = size_threshold
    return grouped


def print_highlights(summary: pd.DataFrame, filtered: pd.DataFrame | None = None) -> None:
    if summary.empty:
        print("No filaments met the filtering criteria.")
        return

    display_df = filtered if filtered is not None and not filtered.empty else summary
    print("\nPhysical + dynamic summary")
    print("--------------------------")
    print(
        f"- {len(display_df)} filaments from {display_df['video'].nunique()} videos processed."
    )
    print(
        f"- Median lifetime: {display_df['lifetime_min'].median():.1f} min; "
        f"median length: {display_df['mean_length_um'].median():.2f} um; "
        f"median arc length: {display_df['mean_arc_length_um'].median():.2f} um."
    )
    print(
        f"- Mean directionality: {display_df['directionality'].mean():.2f}; "
        f"strongest max intensity: {display_df['max_intensity'].max():.1f}."
    )


def report_strongest_signal(frame_df: pd.DataFrame, top_n: int = 3) -> None:
    if frame_df.empty:
        print("No data to summarize signal strength.")
        return

    per_video = frame_df.groupby("video")["sum_intensity"].sum().sort_values(ascending=False)
    print("\nVideos with strongest filament signal (integrated raw intensity):")
    for idx, (video, value) in enumerate(per_video.head(top_n).items(), start=1):
        print(f"  {idx}. {video}: integrated intensity {value:.1f}")


def report_bendiest_filament(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    bendiness = summary["mean_arc_length_um"] / summary["mean_length_um"].clip(lower=1e-6)
    bendy = summary.assign(bendiness=bendiness).sort_values("bendiness", ascending=False).iloc[0]
    print("\nMost bendy filament:")
    print(
        f"  {bendy['video']}#{int(bendy['filament_id'])}: "
        f"bendiness {bendy['bendiness']:.2f} "
        f"(arc {bendy['mean_arc_length_um']:.2f} um / length {bendy['mean_length_um']:.2f} um)"
    )


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    tif_dir = Path(args.tif_dir)

    frame_df = load_frame_measurements(
        mask_dir=mask_dir,
        tif_dir=tif_dir,
        pixel_size_xy=args.pixel_size_xy,
        z_spacing_um=args.z_spacing_um,
    )
    if frame_df.empty:
        print("No frame-level filament measurements could be extracted.")
        return

    frame_path = Path(args.frame_measurements_csv)
    ensure_parent(frame_path)
    frame_df.to_csv(frame_path, index=False)
    print(f"Frame-level measurements written to {frame_path}")

    filtered_frames, size_threshold = filter_frame_measurements(
        frame_df,
        size_tail_quantile=args.size_tail_quantile,
        min_z_planes=args.min_z_planes,
    )
    summary = summarize_tracks(
        filtered_frames,
        frame_interval_min=args.frame_interval_min,
        min_observations=args.min_observations,
    )
    if summary.empty:
        print("No filaments survive filtering and minimum observation requirements.")
        return

    short_path = Path(args.summary_csv)
    processed_path = Path(args.processed_csv)
    length_path = Path(args.length_estimate_csv)
    for path in [short_path, processed_path, length_path]:
        ensure_parent(path)

    summary.to_csv(short_path, index=False)
    summary.to_csv(processed_path, index=False)
    print(f"Track-level summary written to {short_path}")
    print(f"Processed filaments written to {processed_path}")

    length_table = build_length_estimate(filtered_frames, size_threshold)
    if not length_table.empty:
        length_table.to_csv(length_path, index=False)
        print(f"Length estimate table written to {length_path}")

    run_plot_filaments(
        short_path,
        processed_path,
        length_path,
        Path(args.plot_output_dir),
        args.size_tail_quantile,
    )
    report_strongest_signal(filtered_frames)
    report_bendiest_filament(summary)
    print_highlights(summary, summary)


if __name__ == "__main__":
    main()
