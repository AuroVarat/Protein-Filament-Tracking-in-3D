#!/usr/bin/env python3
"""
Render a verification MP4 that combines the raw brightfield with the tracked mask TIFF.

The output replicates the multi-panel layout in `filament_3d_mp4.py` (panning of
individual Z-planes, a 2.5D stack, and orthogonal projections) but overlays the
tracked cells with outlines and color-coded IDs.

Usage:
  python scripts/cell_mask_video.py \
      --tif tiffs3d/ch20_URA7_URA8_002_hyperstack_crop_38.tif \
      --mask results/masks/ch20_URA7_URA8_002_hyperstack_crop_38/ch20_URA7_URA8_002_hyperstack_crop_38_mask.tif \
      --output output/cell_mask_video.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MPL_CONFIG_DIR = Path("output") / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

PALETTE = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.7],
        [1.0, 0.65, 0.0],
        [0.4, 0.1, 1.0],
        [0.0, 0.85, 0.85],
        [1.0, 0.3, 0.3],
        [0.7, 1.0, 0.4],
        [1.0, 0.0, 0.0],
        [0.8, 0.4, 1.0],
        [0.8, 0.8, 0.1],
    ],
    dtype=np.float32,
)
CELL_OUTLINE_RGB = np.array([1.0, 0.8, 0.4], dtype=np.float32)
PALETTE = np.clip(PALETTE, 0, 1)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    norm = np.zeros_like(vol, dtype=np.float32)
    for t in range(vol.shape[0]):
        frame = vol[t]
        mn, mx = frame.min(), frame.max()
        if mx > mn:
            norm[t] = (frame - mn) / (mx - mn)
    return norm


def label_color(label: int) -> np.ndarray:
    if label <= 0:
        return np.zeros(3, dtype=np.float32)
    idx = (label - 1) % PALETTE.shape[0]
    return PALETTE[idx]


def build_color_stack(mask_stack: np.ndarray) -> np.ndarray:
    T, Z, H, W = mask_stack.shape
    color_stack = np.zeros((T, Z, H, W, 3), dtype=np.float32)
    unique_labels = np.unique(mask_stack)
    for label in unique_labels:
        if label == 0:
            continue
        color_stack[mask_stack == label] = np.clip(label_color(int(label)) * 0.6 + 0.4, 0, 1)
    return color_stack


def build_boundary_stack(mask_stack: np.ndarray) -> np.ndarray:
    T, Z, H, W = mask_stack.shape
    boundary_stack = np.zeros((T, Z, H, W), dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    for t in range(T):
        for z in range(Z):
            plane = mask_stack[t, z]
            boundary = np.zeros_like(plane, dtype=bool)
            for label in np.unique(plane):
                if label == 0:
                    continue
                mask_label = plane == label
                if not mask_label.any():
                    continue
                eroded = ndimage.binary_erosion(mask_label, structure=structure, border_value=0)
                boundary |= mask_label & ~eroded
            boundary_stack[t, z] = boundary
    return boundary_stack


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1)


def overlay_boundaries(
    base_rgb: np.ndarray,
    color_plane: np.ndarray,
    boundary: np.ndarray,
    alpha: float = 0.6,
    dilate: int = 1,
    blend: bool = True,
) -> np.ndarray:
    overlay = base_rgb.copy()
    if not boundary.any():
        return overlay
    if dilate <= 0:
        outline = boundary
    else:
        outline = ndimage.binary_dilation(boundary, iterations=dilate)
    if blend:
        overlay[outline] = np.clip(
            overlay[outline] * (1 - alpha) + color_plane[outline] * alpha, 0, 1
        )
    else:
        overlay[outline] = np.clip(color_plane[outline], 0, 1)
    return overlay


def build_projection_boundary(mask2d: np.ndarray) -> np.ndarray:
    if not mask2d.any():
        return mask2d
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(mask2d, structure=structure, border_value=0)
    return mask2d & ~eroded


def render_2p5d(
    vol: np.ndarray,
    color_vol: np.ndarray,
    mask_binary: np.ndarray,
    boundaries: np.ndarray,
    cell_boundaries: np.ndarray,
    shift_x: int = 20,
    shift_y: int = -20,
) -> np.ndarray:
    Z, H, W = vol.shape
    out_H = H + abs(shift_y) * (Z - 1)
    out_W = W + abs(shift_x) * (Z - 1)
    canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)

    for z in range(Z):
        x_off = z * shift_x
        y_off = (Z - 1 - z) * abs(shift_y) if shift_y < 0 else z * shift_y

        base_rgb = gray_to_rgb(vol[z])
        slice_overlay = overlay_boundaries(
            base_rgb, color_vol[z], boundaries[z], alpha=0.9, dilate=1
        )
        cell_plane = np.broadcast_to(CELL_OUTLINE_RGB, slice_overlay.shape)
        slice_overlay = overlay_boundaries(
            slice_overlay, cell_plane, cell_boundaries[z], alpha=0.35, dilate=0
        )
        outline_mask = np.logical_or(boundaries[z], cell_boundaries[z])
        mask_rgb = np.stack([outline_mask] * 3, axis=-1)
        target = canvas[y_off : y_off + H, x_off : x_off + W]
        blend_alpha = 0.65
        target[mask_rgb] = np.clip(
            target[mask_rgb] * (1 - blend_alpha) + slice_overlay[mask_rgb] * blend_alpha,
            0,
            1,
        )
        canvas[y_off : y_off + H, x_off : x_off + W] = target

    return np.clip(canvas, 0, 1)


def make_projections(
    vol: np.ndarray,
    mask_binary: np.ndarray,
    color_vol: np.ndarray,
    boundaries: np.ndarray,
    cell_boundaries: np.ndarray,
    z_scale: int = 10,
) -> np.ndarray:
    Z, H, W = vol.shape
    xy_vol = np.max(vol, axis=0)
    xy_color = np.zeros((H, W, 3), dtype=np.float32)
    xy_mask_boundary = np.zeros((H, W), dtype=bool)
    xy_cell_boundary = np.zeros((H, W), dtype=bool)
    for z in range(Z):
        plane_mask = mask_binary[z]
        xy_color[plane_mask] = color_vol[z][plane_mask]
        xy_mask_boundary |= boundaries[z]
        xy_cell_boundary |= cell_boundaries[z]

    xy_overlay = gray_to_rgb(xy_vol)
    xy_overlay = overlay_boundaries(
        xy_overlay, xy_color, xy_mask_boundary, alpha=0.9, dilate=1, blend=False
    )
    cell_plane_xy = np.broadcast_to(CELL_OUTLINE_RGB, xy_overlay.shape)
    xy_overlay = overlay_boundaries(
        xy_overlay,
        cell_plane_xy,
        xy_cell_boundary,
        alpha=0.35,
        dilate=0,
        blend=False,
    )

    yz_vol = np.max(vol, axis=2).T
    yz_rgb = np.stack([yz_vol] * 3, axis=-1)
    yz_color = np.max(color_vol, axis=2).transpose(1, 0, 2)
    yz_mask = np.max(mask_binary, axis=2).T
    yz_cell_mask = np.max(cell_boundaries, axis=2).T
    yz_boundary = build_projection_boundary(yz_mask)
    yz_cell_boundary = build_projection_boundary(yz_cell_mask)
    yz_rgb = overlay_boundaries(
        yz_rgb,
        yz_color,
        yz_boundary,
        alpha=0.9,
        dilate=0,
        blend=False,
    )
    yz_rgb = overlay_boundaries(
        yz_rgb,
        np.broadcast_to(CELL_OUTLINE_RGB, yz_rgb.shape),
        yz_cell_boundary,
        alpha=0.35,
        dilate=0,
        blend=False,
    )
    yz_rgb = np.repeat(yz_rgb, z_scale // Z if z_scale > Z else 1, axis=1)

    xz_vol = np.max(vol, axis=1)
    xz_rgb = np.stack([xz_vol] * 3, axis=-1)
    xz_color = np.max(color_vol, axis=1)
    xz_mask = np.max(mask_binary, axis=1)
    xz_cell_mask = np.max(cell_boundaries, axis=1)
    xz_boundary = build_projection_boundary(xz_mask)
    xz_cell_boundary = build_projection_boundary(xz_cell_mask)
    xz_rgb = overlay_boundaries(
        xz_rgb,
        xz_color,
        xz_boundary,
        alpha=0.9,
        dilate=0,
        blend=False,
    )
    xz_rgb = overlay_boundaries(
        xz_rgb,
        np.broadcast_to(CELL_OUTLINE_RGB, xz_rgb.shape),
        xz_cell_boundary,
        alpha=0.35,
        dilate=0,
        blend=False,
    )
    xz_rgb = np.repeat(xz_rgb, z_scale // Z if z_scale > Z else 1, axis=0)

    H_yz, W_yz, _ = yz_rgb.shape
    H_xz, W_xz, _ = xz_rgb.shape

    canvas_H = H + H_xz + 10
    canvas_W = W + W_yz + 10
    canvas = np.zeros((canvas_H, canvas_W, 3), dtype=np.float32)
    canvas[0:H, 0:W] = xy_overlay
    canvas[0:H, W + 10 : W + 10 + W_yz] = yz_rgb
    canvas[H + 10 : H + 10 + H_xz, 0:W] = xz_rgb
    return np.clip(canvas, 0, 1)


def make_frame(
    vol: np.ndarray,
    mask_binary: np.ndarray,
    color_vol: np.ndarray,
    boundaries: np.ndarray,
    cell_boundaries: np.ndarray,
    frame_idx: int,
    total_frames: int,
    detection_history: list[bool],
) -> np.ndarray:
    Z, H, W = vol.shape
    pano_raw = np.hstack([vol[z] for z in range(Z)])
    pano_raw_rgb = np.stack([pano_raw] * 3, axis=-1)

    overlays = []
    for z in range(Z):
        base_rgb = gray_to_rgb(vol[z])
        overlay_rgb = overlay_boundaries(
            base_rgb, color_vol[z], boundaries[z], alpha=0.9, dilate=1, blend=False
        )
        cell_plane = np.broadcast_to(CELL_OUTLINE_RGB, overlay_rgb.shape)
        overlay_rgb = overlay_boundaries(
            overlay_rgb, cell_plane, cell_boundaries[z], alpha=0.35, dilate=0, blend=False
        )
        overlays.append(overlay_rgb)
    pano_overlay = np.hstack(overlays)
    pano_overlay = np.clip(pano_overlay, 0, 1)

    stack_img = render_2p5d(
        vol,
        color_vol,
        mask_binary,
        boundaries,
        cell_boundaries,
        shift_x=25,
        shift_y=-25,
    )
    ortho_grid = make_projections(
        vol,
        mask_binary,
        color_vol,
        boundaries,
        cell_boundaries,
        z_scale=10,
    )

    h_s, w_s, _ = stack_img.shape
    h_o, w_o, _ = ortho_grid.shape

    pano_W = pano_raw_rgb.shape[1]
    row3_H = max(h_s, h_o) + 40
    row3 = np.zeros((row3_H, pano_W, 3), dtype=np.float32)
    y_s = (row3_H - h_s) // 2
    x_s_start = max(10, (pano_W // 2 - w_s) // 2)
    row3[y_s : y_s + h_s, x_s_start : x_s_start + w_s] = stack_img
    y_o = (row3_H - h_o) // 2
    x_o_start = pano_W // 2 + max(10, (pano_W // 2 - w_o) // 2)
    w_o_clip = min(w_o, pano_W - x_o_start)
    row3[y_o : y_o + h_o, x_o_start : x_o_start + w_o_clip] = ortho_grid[:, 0:w_o_clip]

    pad = np.zeros((10, pano_W, 3), dtype=np.float32)
    frame = np.vstack([pano_raw_rgb, pad, pano_overlay, pad, row3])

    header_h = 80
    new_frame = np.zeros((frame.shape[0] + header_h, frame.shape[1], 3), dtype=np.float32)
    new_frame[header_h:, :, :] = frame
    frame = new_frame

    pil_img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    bar_h = 24
    if detection_history is not None:
        bar_x, bar_y = 10, 10
        bar_w = frame.shape[1] - 20
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(40, 40, 40))
        n = len(detection_history)
        seg_w = bar_w / n
        for i, detected in enumerate(detection_history):
            if detected:
                draw.rectangle(
                    [bar_x + i * seg_w, bar_y, bar_x + (i + 1) * seg_w, bar_y + bar_h],
                    fill=(0, 255, 0),
                )
        cursor_x = bar_x + frame_idx * seg_w
        draw.rectangle(
            [cursor_x, bar_y - 2, cursor_x + max(2, seg_w), bar_y + bar_h + 2],
            outline=(255, 255, 255),
            width=2,
        )

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24
        )
    except Exception:
        font = ImageFont.load_default()

    txt = f"Frame: {frame_idx + 1} / {total_frames}"
    draw.text((frame.shape[1] - 220, 45), txt, fill=(255, 255, 0), font=font)

    frame = np.array(pil_img).astype(np.float32) / 255.0
    frame = np.clip(frame, 0, 1)
    h, w = frame.shape[:2]
    new_h = h + (h % 2)
    new_w = w + (w % 2)
    if new_h != h or new_w != w:
        padded = np.zeros((new_h, new_w, 3), dtype=frame.dtype)
        padded[:h, :w] = frame
        frame = padded
    return (frame * 255).astype(np.uint8)


def process_pair(
    tif_path: Path,
    mask_path: Path,
    cell_mask_path: Path,
    output_dir: Path,
    fps: float | None,
) -> None:
    base = tif_path.stem
    output_path = output_dir / f"{base}_cell_masks.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(tif_path).astype(np.float32)
    if img.ndim == 4:
        img = img[np.newaxis]
    if img.ndim != 5:
        raise ValueError(f"Expected 5D TIFF, got {img.shape}")
    if img.shape[2] < 2:
        raise ValueError("Expected at least two channels; filament channel missing.")
    filament_vol = img[:, :, 1, :, :]
    norm_vol = normalize_volume(filament_vol)

    mask = tifffile.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[np.newaxis]
    if mask.ndim != 4:
        raise ValueError(f"Mask must be (T,Z,H,W) or (Z,H,W), got {mask.shape}")

    if mask.shape[:3] != norm_vol.shape[:3]:
        raise ValueError("Mask and BF TIFF must share the same (T,Z,H) dimensions.")

    color_stack = build_color_stack(mask)
    boundary_stack = build_boundary_stack(mask)
    mask_binary = mask > 0

    cell_mask = tifffile.imread(cell_mask_path)
    if cell_mask.ndim == 3:
        cell_mask = cell_mask[np.newaxis]
    if cell_mask.ndim != 4:
        raise ValueError(f"Cell mask must be (T,Z,H,W) or (Z,H,W), got {cell_mask.shape}")
    cell_boundaries = build_boundary_stack(cell_mask)
    history = [mask[t].max() > 0 for t in range(mask.shape[0])]

    resolved_fps = fps if fps is not None and fps > 0 else 1.0
    resolved_fps = max(1.0, resolved_fps)
    print(
        f"Rendering {output_path.name} ({norm_vol.shape[0]} frames) at {resolved_fps:.1f} fps ..."
    )
    writer = imageio.get_writer(
        str(output_path), fps=resolved_fps, macro_block_size=None
    )
    for t in range(norm_vol.shape[0]):
        frame = make_frame(
            norm_vol[t],
            mask_binary[t],
            color_stack[t],
            boundary_stack[t],
            cell_boundaries[t],
            frame_idx=t,
            total_frames=norm_vol.shape[0],
            detection_history=history,
        )
        writer.append_data(frame)
    writer.close()
    print(f"Saved colored cell-ID video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4s of masks overlayed on BF.")
    parser.add_argument("--input-dir", default="tiffs3d", help="Directory with TIFF videos.")
    parser.add_argument("--mask-dir", default="results/masks", help="Directory with stacked filament masks.")
    parser.add_argument(
        "--cell-mask-dir", default="results/cell_masks", help="Directory with stacked cell masks."
    )
    parser.add_argument("--output-dir", default="videos", help="Where MP4s go.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional playback speed for the MP4. All TIFF frames are always written; default is 1 fps.",
    )
    parser.add_argument(
        "--tif", help="Optional single TIFF path (use instead of input-dir)."
    )
    parser.add_argument(
        "--mask", help="Optional single filament mask path (use instead of mask-dir)."
    )
    parser.add_argument(
        "--cell-mask",
        help="Optional single cell mask path (use instead of cell-mask-dir).",
    )
    args = parser.parse_args()

    pairs: list[tuple[Path, Path, Path]] = []
    if args.tif and args.mask and args.cell_mask:
        pairs.append((Path(args.tif), Path(args.mask), Path(args.cell_mask)))
    else:
        tif_dir = Path(args.input_dir)
        mask_dir = Path(args.mask_dir)
        cell_mask_dir = Path(args.cell_mask_dir)
        if not tif_dir.exists() or not mask_dir.exists() or not cell_mask_dir.exists():
            raise FileNotFoundError(
                "input-dir, mask-dir, and cell-mask-dir must all exist. "
                "Run `uv run scripts/batch_3d_tracker.py --output-dir results` to regenerate the masks."
            )
        mask_map: dict[str, Path] = {}
        for mask_path in sorted(mask_dir.rglob("*_mask.tif")):
            video_base = mask_path.stem.replace("_mask", "")
            mask_map[video_base] = mask_path
        cell_mask_map: dict[str, Path] = {}
        for cell_path in sorted(cell_mask_dir.rglob("*_mask.tif")):
            video_base = cell_path.stem.replace("_mask", "")
            cell_mask_map[video_base] = cell_path
        tif_bases = []
        for tif_path in sorted(tif_dir.glob("*.tif")):
            base = tif_path.stem
            tif_bases.append(base)
            mask_path = mask_map.get(base)
            cell_mask_path = cell_mask_map.get(base)
            if mask_path is None or cell_mask_path is None:
                missing = []
                if mask_path is None:
                    missing.append("filament mask")
                if cell_mask_path is None:
                    missing.append("cell mask")
                print(
                    f"⚠️  Missing {', '.join(missing)} for '{base}'. "
                    "Run the tracker (`uv run scripts/batch_3d_tracker.py …`) to regenerate both masks."
                )
                continue
            pairs.append((tif_path, mask_path, cell_mask_path))

        if not pairs and tif_bases:
            raise FileNotFoundError(
                "No TIFF/mask pairs found. Please run the tracker (`uv run scripts/batch_3d_tracker.py …`) "
                "to produce both mask stacks before visualizing."
            )

    if not pairs:
        raise ValueError("No TIFF/mask pairs found to render.")

    output_dir = Path(args.output_dir)
    for tif_path, mask_path, cell_mask_path in pairs:
        process_pair(tif_path, mask_path, cell_mask_path, output_dir, args.fps)


if __name__ == "__main__":
    main()
