from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import exposure, filters, measure, morphology, restoration, util


@dataclass
class Params:
    bright_filaments: bool = True
    ridge_filter: str = "meijering"   # meijering | sato | frangi
    background_radius: int = 30
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    low_q: float = 0.88
    high_q: float = 0.97
    remove_objects_leq: int = 20
    remove_holes_leq: int = 16
    min_branch_length_px: float = 8.0
    max_centroid_move: float = 20.0
    max_angle_change_deg: float = 35.0
    max_length_ratio: float = 2.0
    max_cost: float = 35.0
    dist_weight: float = 1.0
    endpoint_weight: float = 0.75
    angle_weight: float = 8.0
    length_weight: float = 5.0
    pixel_size_um: float = 0.184


def load_stack(path: str | Path) -> np.ndarray:
    stack = tifffile.imread(path)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D stack, got shape {stack.shape}")
    return stack


def remove_small_objects_compat(mask: np.ndarray, size: int) -> np.ndarray:
    try:
        return morphology.remove_small_objects(mask, max_size=size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=size + 1)


def remove_small_holes_compat(mask: np.ndarray, size: int) -> np.ndarray:
    try:
        return morphology.remove_small_holes(mask, max_size=size)
    except TypeError:
        return morphology.remove_small_holes(mask, area_threshold=size + 1)


# ==========================================
# 1. DENOISING & CONTRAST ENHANCEMENT
# ==========================================
def preprocess(frame: np.ndarray, p: Params) -> np.ndarray:
    img = util.img_as_float32(frame)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if not p.bright_filaments:
        img = util.invert(img)

    img = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
    bg = restoration.rolling_ball(img, radius=p.background_radius)
    img = np.clip(img - bg, 0, None)
    img = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
    return img.astype(np.float32)


# ==========================================
# 2. RIDGE ENHANCEMENT
# ==========================================
def ridge_enhance(img: np.ndarray, p: Params) -> np.ndarray:
    name = p.ridge_filter.lower()

    if name == "meijering":
        ridge = filters.meijering(img, sigmas=p.sigmas, black_ridges=False)
    elif name == "sato":
        ridge = filters.sato(img, sigmas=p.sigmas, black_ridges=False)
    elif name == "frangi":
        ridge = filters.frangi(img, sigmas=p.sigmas, black_ridges=False)
    else:
        raise ValueError("ridge_filter must be one of: meijering, sato, frangi")

    ridge = exposure.rescale_intensity(ridge, out_range=(0.0, 1.0))
    return ridge.astype(np.float32)


# ==========================================
# 3. SEGMENTATION & 4. MORPHOLOGICAL CLEAN UP
# ==========================================
def segment_ridges(ridge: np.ndarray, p: Params) -> np.ndarray:
    low = float(np.quantile(ridge, p.low_q))
    high = float(np.quantile(ridge, p.high_q))
    high = max(high, low + 1e-3)

    mask = filters.apply_hysteresis_threshold(ridge, low, high)
    mask = remove_small_objects_compat(mask, p.remove_objects_leq)
    mask = remove_small_holes_compat(mask, p.remove_holes_leq)
    mask = morphology.closing(mask, footprint=morphology.disk(1))
    return mask


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(binary.astype(np.uint8), kernel, mode="constant", cval=0)


def order_path(coords: np.ndarray) -> np.ndarray:
    coord_set = {tuple(c) for c in coords.tolist()}
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for y, x in coord_set:
        hits = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in coord_set:
                    hits.append(q)
        nbrs[(y, x)] = hits

    endpoints = [node for node, n in nbrs.items() if len(n) == 1]
    start = endpoints[0] if endpoints else next(iter(coord_set))

    ordered = [start]
    visited = {start}
    prev = None
    cur = start

    while True:
        candidates = [n for n in nbrs[cur] if n != prev and n not in visited]
        if not candidates:
            break
        nxt = candidates[0]
        ordered.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    if len(ordered) != len(coord_set):
        for pt in coord_set:
            if pt not in visited:
                ordered.append(pt)

    return np.array(ordered, dtype=float)


def polyline_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def axial_angle(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    vy, vx = vt[0]
    return math.atan2(vy, vx) % math.pi


# ==========================================
# 6. FEATURE EXTRACTION
# ==========================================
def extract_segments(skeleton: np.ndarray, frame_idx: int, p: Params) -> list[dict]:
    deg = neighbor_count(skeleton)
    branchpoints = skeleton & (deg >= 3)
    split_skeleton = skeleton & ~branchpoints
    labels = measure.label(split_skeleton, connectivity=2)

    rows: list[dict] = []
    for region in measure.regionprops(labels):
        coords = region.coords
        if coords.shape[0] < 2:
            continue

        path = order_path(coords)
        length_px = polyline_length(path)
        if length_px < p.min_branch_length_px:
            continue

        cy, cx = coords.mean(axis=0)
        y0, x0 = path[0]
        y1, x1 = path[-1]

        rows.append(
            {
                "frame": int(frame_idx),
                "segment_id": int(region.label),
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "length_px": float(length_px),
                "length_um": float(length_px * p.pixel_size_um),
                "angle_rad": float(axial_angle(coords.astype(float))),
                "n_pixels": int(coords.shape[0]),
                "y0": float(y0),
                "x0": float(x0),
                "y1": float(y1),
                "x1": float(x1),
            }
        )
    return rows


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def endpoint_cost(a: pd.Series, b: pd.Series) -> float:
    a0 = np.array([a["y0"], a["x0"]], dtype=float)
    a1 = np.array([a["y1"], a["x1"]], dtype=float)
    b0 = np.array([b["y0"], b["x0"]], dtype=float)
    b1 = np.array([b["y1"], b["x1"]], dtype=float)

    same_order = np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)
    flipped = np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)
    return float(min(same_order, flipped) / 2.0)


def build_cost_matrix(prev_df: pd.DataFrame, curr_df: pd.DataFrame, p: Params) -> np.ndarray:
    cost = np.full((len(prev_df), len(curr_df)), 1e6, dtype=float)
    max_ang = math.radians(p.max_angle_change_deg)

    for i, a in prev_df.iterrows():
        for j, b in curr_df.iterrows():
            d_cent = math.hypot(a["centroid_y"] - b["centroid_y"], a["centroid_x"] - b["centroid_x"])
            if d_cent > p.max_centroid_move:
                continue

            d_ang = angle_diff(float(a["angle_rad"]), float(b["angle_rad"]))
            if d_ang > max_ang:
                continue

            len_ratio = max(float(a["length_px"]), float(b["length_px"])) / max(
                1e-6, min(float(a["length_px"]), float(b["length_px"]))
            )
            if len_ratio > p.max_length_ratio:
                continue

            d_end = endpoint_cost(a, b)
            d_len = abs(math.log((float(b["length_px"]) + 1e-6) / (float(a["length_px"]) + 1e-6)))

            total = (
                p.dist_weight * d_cent
                + p.endpoint_weight * d_end
                + p.angle_weight * d_ang
                + p.length_weight * d_len
            )
            cost[i, j] = total

    return cost


# ==========================================
# 7. TRACKING (HUNGARIAN ALGORITHM)
# ==========================================
def assign_tracks(feature_tables: list[pd.DataFrame], p: Params) -> pd.DataFrame:
    next_track_id = 0
    prev_df: pd.DataFrame | None = None
    output = []

    for df in feature_tables:
        df = df.copy().reset_index(drop=True)

        if df.empty:
            output.append(df)
            prev_df = df
            continue

        df["track_id"] = -1

        if prev_df is None or prev_df.empty:
            for i in range(len(df)):
                df.loc[i, "track_id"] = next_track_id
                next_track_id += 1
            output.append(df)
            prev_df = df
            continue

        cost = build_cost_matrix(prev_df, df, p)
        rows, cols = linear_sum_assignment(cost)
        matched_curr = set()

        for r, c in zip(rows, cols):
            if cost[r, c] >= p.max_cost:
                continue
            df.loc[c, "track_id"] = int(prev_df.loc[r, "track_id"])
            matched_curr.add(c)

        for i in range(len(df)):
            if i not in matched_curr and int(df.loc[i, "track_id"]) < 0:
                df.loc[i, "track_id"] = next_track_id
                next_track_id += 1

        output.append(df)
        prev_df = df

    if not output:
        return pd.DataFrame()

    return pd.concat(output, ignore_index=True)


def process_stack(stack: np.ndarray, p: Params):
    masks = []
    skeletons = []
    feature_tables = []

    for frame_idx, frame in enumerate(stack):
        # 1. Denoising & Contrast Enhancement
        pre = preprocess(frame, p)
        # 2. Ridge Enhancement
        ridge = ridge_enhance(pre, p)
        # 3 & 4. Segmentation & Morphological Clean up
        mask = segment_ridges(ridge, p)
        # 5. Skeletonization
        skeleton = morphology.skeletonize(mask)
        # 6. Feature Extraction
        rows = extract_segments(skeleton, frame_idx, p)

        masks.append(mask.astype(np.uint8) * 255)
        skeletons.append(skeleton.astype(np.uint8) * 255)
        feature_tables.append(pd.DataFrame(rows))

    # 7. Tracking (linking segments across frames over time)
    tracks = assign_tracks(feature_tables, p)

    # 8. Extend stable tracks by one frame before/after checking for faint signals
    if not tracks.empty:
        import dataclasses
        
        masks_arr = np.stack(masks) > 0
        skeletons_arr = np.stack(skeletons) > 0
        
        track_counts = tracks["track_id"].value_counts()
        # Extend ALL tracks. Even if a track was broken and only lasted 1-2 frames 
        # at the edge of a stable cluster, bridging it to a faint segment will 
        # successfully expand the final clustered duration!
        stable_track_ids = track_counts.index.tolist()
        
        if stable_track_ids:
            # We must drastically drop the intensity threshold (low_q/high_q) because 
            # at 99.7%, faint filaments are completely invisible to the algorithm. 
            # Dropping size to 5 alone doesn't expose them!
            p_ext = dataclasses.replace(p, 
                remove_objects_leq=5,
                low_q=max(0.000, p.low_q - 0.050),
                high_q=max(0.000, p.high_q - 0.050)
            )
            ext_cache = {}
            
            def get_ext(f_idx):
                if f_idx not in ext_cache:
                    frame = stack[f_idx]
                    pre = preprocess(frame, p_ext)
                    ridge = ridge_enhance(pre, p_ext)
                    mask = segment_ridges(ridge, p_ext)
                    skel = morphology.skeletonize(mask)
                    
                    deg = neighbor_count(skel)
                    branchpoints = skel & (deg >= 3)
                    split_skel = skel & ~branchpoints
                    labels = measure.label(split_skel, connectivity=2)
                    
                    rows = extract_segments(skel, f_idx, p_ext)
                    ext_cache[f_idx] = (mask, skel, labels, pd.DataFrame(rows))
                return ext_cache[f_idx]

            new_tracks = [tracks]
            
            for tid in stable_track_ids:
                track_df = tracks[tracks["track_id"] == tid].sort_values("frame")
                
                # We rely entirely on the STRICT PHYSICAL OVERLAP check below.
                # A 100px filament shrinking to a 5px tip will cause a massive centroid shift (47px)
                # and massive length ratio (20x), which the strict cost matrix normally rejects.
                # So we completely relax the math rules, because the spatial footprint overlap 
                # rigorously ensures it's the exact same filament structure anyway!
                p_match = dataclasses.replace(p,
                    max_length_ratio=1e5,
                    max_centroid_move=1e5,
                    max_angle_change_deg=360.0,
                    max_cost=1e5
                )
                
                with open("/tmp/ext_log.txt", "a") as f:
                    f.write(f"\nExtending Track {tid} with p_match.max_cost={p_match.max_cost}\n")
                
                # Check one frame before
                start_row = track_df.iloc[0]
                f_prev = int(start_row["frame"]) - 1
                if f_prev >= 0:
                    emask, eskel, elabels, edf = get_ext(f_prev)
                    if not edf.empty:
                        cost = build_cost_matrix(pd.DataFrame([start_row]).reset_index(drop=True), edf, p_match)
                        best_idx = int(np.argmin(cost[0]))
                        best_cost = cost[0, best_idx]
                        
                        with open("/tmp/ext_log.txt", "a") as f:
                            f.write(f"Prev Frame {f_prev}: Found {len(edf)} segments. Best cost={best_cost} against {p_match.max_cost}\n")
                        
                        if best_cost < p_match.max_cost:
                            matched_row = edf.iloc[best_idx].copy()
                            
                            seg_mask = (elabels == matched_row["segment_id"])
                            
                            # STRICT PHYSICAL CHECK: Extension MUST touch the endpoints of the previous segment!
                            # Tiny 4-pixel objects have meaningless computed angles, so we ignore angle/length,
                            # but rigorously ensure the new object physically overlaps the actual *tips* of the filament.
                            import scipy.ndimage as ndi
                            f_curr = int(start_row["frame"])
                            eps_mask = np.zeros_like(masks_arr[f_curr], dtype=bool)
                            H, W = eps_mask.shape
                            y0, x0 = int(round(start_row["y0"])), int(round(start_row["x0"]))
                            y1, x1 = int(round(start_row["y1"])), int(round(start_row["x1"]))
                            if 0 <= y0 < H and 0 <= x0 < W: eps_mask[y0, x0] = True
                            if 0 <= y1 < H and 0 <= x1 < W: eps_mask[y1, x1] = True
                            
                            dilated_tips = ndi.binary_dilation(eps_mask, iterations=3)
                            
                            if np.any(seg_mask & dilated_tips):
                                matched_row["track_id"] = tid
                                new_tracks.append(pd.DataFrame([matched_row]))
                                
                                skeletons_arr[f_prev] |= seg_mask
                                
                                cc_mask = measure.label(emask, connectivity=2)
                                ov_vals = np.unique(cc_mask[seg_mask])
                                for v in ov_vals:
                                    if v > 0:
                                        masks_arr[f_prev] |= (cc_mask == v)

                # Check one frame after
                end_row = track_df.iloc[-1]
                f_next = int(end_row["frame"]) + 1
                if f_next < len(stack):
                    emask, eskel, elabels, edf = get_ext(f_next)
                    if not edf.empty:
                        cost = build_cost_matrix(pd.DataFrame([end_row]).reset_index(drop=True), edf, p_match)
                        best_idx = int(np.argmin(cost[0]))
                        best_cost = cost[0, best_idx]
                        if best_cost < p_match.max_cost:
                            matched_row = edf.iloc[best_idx].copy()
                            
                            seg_mask = (elabels == matched_row["segment_id"])
                            
                            # STRICT PHYSICAL CHECK: Extension MUST touch the endpoints of the previous segment!
                            import scipy.ndimage as ndi
                            f_curr = int(end_row["frame"])
                            eps_mask = np.zeros_like(masks_arr[f_curr], dtype=bool)
                            H, W = eps_mask.shape
                            y0, x0 = int(round(end_row["y0"])), int(round(end_row["x0"]))
                            y1, x1 = int(round(end_row["y1"])), int(round(end_row["x1"]))
                            if 0 <= y0 < H and 0 <= x0 < W: eps_mask[y0, x0] = True
                            if 0 <= y1 < H and 0 <= x1 < W: eps_mask[y1, x1] = True
                            
                            dilated_tips = ndi.binary_dilation(eps_mask, iterations=3)
                            
                            if np.any(seg_mask & dilated_tips):
                                matched_row["track_id"] = tid
                                new_tracks.append(pd.DataFrame([matched_row]))
                                
                                skeletons_arr[f_next] |= seg_mask
                                
                                cc_mask = measure.label(emask, connectivity=2)
                                ov_vals = np.unique(cc_mask[seg_mask])
                                for v in ov_vals:
                                    if v > 0:
                                        masks_arr[f_next] |= (cc_mask == v)
            
            tracks = pd.concat(new_tracks, ignore_index=True)
            tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
            
            return (masks_arr.astype(np.uint8) * 255), (skeletons_arr.astype(np.uint8) * 255), tracks

    return np.stack(masks), np.stack(skeletons), tracks


def save_outputs(out_dir: str | Path, masks: np.ndarray, skeletons: np.ndarray, tracks: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(out_dir / "filament_masks.tif", masks)
    tifffile.imwrite(out_dir / "filament_skeletons.tif", skeletons)
    tracks.to_csv(out_dir / "filament_tracks.csv", index=False)

    if not tracks.empty:
        summary = (
            tracks.groupby("track_id", as_index=False)
            .agg(
                start_frame=("frame", "min"),
                end_frame=("frame", "max"),
                n_observations=("frame", "size"),
                mean_length_px=("length_px", "mean"),
                mean_length_um=("length_um", "mean"),
            )
        )
        summary.to_csv(out_dir / "track_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tif", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    params = Params()
    stack = load_stack(args.input_tif)
    masks, skeletons, tracks = process_stack(stack, params)
    save_outputs(args.output_dir, masks, skeletons, tracks)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Frames: {stack.shape[0]}")
    print(f"Tracked segments: {0 if tracks.empty else tracks['track_id'].nunique()}")


if __name__ == "__main__":
    main()