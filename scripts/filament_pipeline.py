#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage
from torch.utils.data import DataLoader

from unet2d import TinyUNet2D, SegDataset2DTemporalAuto
from unet3d import TinyUNet3D, SegDataset3DTemporalAuto
from utils import best_device


MODELS_DIR = Path("models")
ANNOTATION_ROOT = MODELS_DIR / "annotations"
RESULTS_ROOT = Path("results")
RESULT_MASK_DIR = RESULTS_ROOT / "masks"
RESULT_CELL_MASK_DIR = RESULTS_ROOT / "cell_masks"
RESULT_TRACKING_DIR = RESULTS_ROOT / "tracking_csvs"
ACTIVE_MODEL_PATHS = {
    "2d": MODELS_DIR / "filament_unet2d_temporal_auto.pt",
    "3d": MODELS_DIR / "filament_unet3d_temporal_auto.pt",
}


@dataclass
class DatasetInfo:
    filepath: str
    mode: str
    timepoints: int
    z_planes: int
    height: int
    width: int
    source_shape: tuple[int, ...]
    has_brightfield: bool
    filament_channel_index: int


@dataclass
class TrainResult:
    mode: str
    checkpoint_path: str
    active_checkpoint_path: str
    promoted_to_active: bool
    epochs: int
    num_sequences: int
    num_annotations: int
    log_path: str


@dataclass
class InferenceArtifact:
    source_path: str
    mode: str
    mask_tiff: str
    cell_mask_tiff: str
    tracking_csv: str
    raw_shape: tuple[int, ...]


def _ensure_dirs() -> None:
    for path in [
        ANNOTATION_ROOT / "2d",
        ANNOTATION_ROOT / "3d",
        MODELS_DIR,
        RESULT_MASK_DIR,
        RESULT_CELL_MASK_DIR,
        RESULT_TRACKING_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def list_tiff_files(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    files: list[str] = []
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(str(p.resolve()) for p in sorted(path.glob("*.tif")))
            files.extend(str(p.resolve()) for p in sorted(path.glob("*.tiff")))
        elif path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            files.append(str(path.resolve()))
    return sorted(dict.fromkeys(files))


def _annotation_key(filepath: str) -> str:
    resolved = str(Path(filepath).resolve())
    stem = Path(filepath).stem
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def annotation_dir_for(filepath: str, mode: str) -> Path:
    return ANNOTATION_ROOT / mode / _annotation_key(filepath)


def _legacy_mask_path(filepath: str, mode: str, frame_idx: int) -> Path:
    stem = Path(filepath).stem
    if mode == "2d":
        return MODELS_DIR / "masks" / f"{stem}_{frame_idx:04d}.npy"
    return MODELS_DIR / "masks3d" / f"{stem}_t{frame_idx:04d}.npy"


def inspect_tiff(filepath: str) -> DatasetInfo:
    arr = tifffile.imread(filepath)
    shape = tuple(int(v) for v in arr.shape)
    if arr.ndim == 3:
        t, h, w = arr.shape
        return DatasetInfo(
            filepath=str(Path(filepath).resolve()),
            mode="2d",
            timepoints=int(t),
            z_planes=1,
            height=int(h),
            width=int(w),
            source_shape=shape,
            has_brightfield=False,
            filament_channel_index=0,
        )
    if arr.ndim == 4:
        t, a, h, w = arr.shape
        if a <= 8:
            return DatasetInfo(
                filepath=str(Path(filepath).resolve()),
                mode="3d",
                timepoints=int(t),
                z_planes=int(a),
                height=int(h),
                width=int(w),
                source_shape=shape,
                has_brightfield=False,
                filament_channel_index=0,
            )
        raise ValueError(f"Unsupported 4D TIFF shape for {filepath}: {shape}")
    if arr.ndim == 5:
        t, z, c, h, w = arr.shape
        fil_ch = 1 if c > 1 else 0
        return DatasetInfo(
            filepath=str(Path(filepath).resolve()),
            mode="3d",
            timepoints=int(t),
            z_planes=int(z),
            height=int(h),
            width=int(w),
            source_shape=shape,
            has_brightfield=bool(c > 1),
            filament_channel_index=int(fil_ch),
        )
    raise ValueError(f"Unsupported TIFF shape for {filepath}: {shape}")


def load_dataset(filepath: str) -> tuple[DatasetInfo, np.ndarray, np.ndarray | None]:
    info = inspect_tiff(filepath)
    arr = tifffile.imread(filepath).astype(np.float32)
    brightfield = None
    if info.mode == "2d":
        norm = np.zeros_like(arr, dtype=np.float32)
        for t in range(arr.shape[0]):
            mn, mx = float(arr[t].min()), float(arr[t].max())
            if mx > mn:
                norm[t] = (arr[t] - mn) / (mx - mn)
        return info, norm, brightfield

    if arr.ndim == 4:
        fil = arr
    else:
        fil = arr[:, :, info.filament_channel_index, :, :]
        if info.has_brightfield:
            brightfield = arr[:, :, 0, :, :]

    norm = np.zeros_like(fil, dtype=np.float32)
    for t in range(fil.shape[0]):
        mn, mx = float(fil[t].min()), float(fil[t].max())
        if mx > mn:
            norm[t] = (fil[t] - mn) / (mx - mn)
    return info, norm, brightfield


def load_annotation(filepath: str, mode: str, frame_idx: int, expected_shape: tuple[int, ...]) -> np.ndarray:
    ann_dir = annotation_dir_for(filepath, mode)
    ann_path = ann_dir / f"frame_{frame_idx:04d}.npy"
    if ann_path.exists():
        arr = np.load(ann_path).astype(np.float32)
        return _coerce_annotation_shape(arr, expected_shape)
    legacy = _legacy_mask_path(filepath, mode, frame_idx)
    if legacy.exists():
        arr = np.load(legacy).astype(np.float32)
        return _coerce_annotation_shape(arr, expected_shape)
    return np.zeros(expected_shape, dtype=np.float32)


def save_annotation(filepath: str, mode: str, frame_idx: int, mask: np.ndarray, clear_legacy: bool = False) -> str:
    _ensure_dirs()
    ann_dir = annotation_dir_for(filepath, mode)
    ann_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ann_dir / "meta.json"
    if not meta_path.exists():
        meta = {"filepath": str(Path(filepath).resolve()), "mode": mode}
        meta_path.write_text(json.dumps(meta, indent=2))
    mask_path = ann_dir / f"frame_{frame_idx:04d}.npy"
    np.save(mask_path, mask.astype(np.float32))
    if clear_legacy:
        legacy = _legacy_mask_path(filepath, mode, frame_idx)
        if legacy.exists():
            legacy.unlink()
    return str(mask_path)


def count_annotations(filepath: str, mode: str) -> int:
    ann_dir = annotation_dir_for(filepath, mode)
    count = 0
    if ann_dir.exists():
        for p in sorted(ann_dir.glob("frame_*.npy")):
            try:
                if np.load(p).max() > 0:
                    count += 1
            except Exception:
                continue
    else:
        info = inspect_tiff(filepath)
        for t in range(info.timepoints):
            legacy = _legacy_mask_path(filepath, mode, t)
            if legacy.exists():
                try:
                    if np.load(legacy).max() > 0:
                        count += 1
                except Exception:
                    continue
    return count


def annotation_summary(files: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filepath in files:
        info = inspect_tiff(filepath)
        rows.append(
            {
                "file": str(Path(filepath).name),
                "mode": info.mode,
                "frames": info.timepoints,
                "z_planes": info.z_planes,
                "annotations": count_annotations(filepath, info.mode),
                "path": str(Path(filepath).resolve()),
            }
        )
    return rows


def _coerce_annotation_shape(arr: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if arr.shape == expected_shape:
        return (arr > 0.1).astype(np.float32)
    if arr.ndim == 2 and len(expected_shape) == 3 and expected_shape[0] == 1:
        return arr[np.newaxis, ...].astype(np.float32)
    if arr.ndim == 3 and len(expected_shape) == 2 and arr.shape[0] == 1:
        return arr[0].astype(np.float32)
    raise ValueError(f"Annotation shape {arr.shape} does not match expected {expected_shape}")


def _build_train_sequences_2d(files: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    ann_count = 0
    for filepath in files:
        info, data, _ = load_dataset(filepath)
        if info.mode != "2d":
            continue
        seq_mask = np.zeros_like(data, dtype=np.float32)
        seq_valid = np.zeros(info.timepoints, dtype=np.float32)
        for t in range(info.timepoints):
            mask = load_annotation(filepath, "2d", t, (info.height, info.width))
            if mask.max() > 0:
                seq_mask[t] = mask
                seq_valid[t] = 1.0
                ann_count += 1
        if seq_valid.sum() == 0:
            continue
        for t in range(min(3, info.timepoints)):
            if seq_valid[t] == 0:
                seq_valid[t] = 1.0
        sequences.append(data)
        masks.append(seq_mask)
        valid.append(seq_valid)
    return sequences, masks, valid, ann_count


def _build_train_sequences_3d(files: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    ann_count = 0
    for filepath in files:
        info, data, _ = load_dataset(filepath)
        if info.mode != "3d":
            continue
        seq_mask = np.zeros_like(data, dtype=np.float32)
        seq_valid = np.zeros(info.timepoints, dtype=np.float32)
        for t in range(info.timepoints):
            mask = load_annotation(filepath, "3d", t, (info.z_planes, info.height, info.width))
            if mask.max() > 0:
                seq_mask[t] = mask
                seq_valid[t] = 1.0
                ann_count += 1
        if seq_valid.sum() == 0:
            continue
        for t in range(min(3, info.timepoints)):
            if seq_valid[t] == 0:
                seq_valid[t] = 1.0
        sequences.append(data)
        masks.append(seq_mask)
        valid.append(seq_valid)
    return sequences, masks, valid, ann_count


def _temporal_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    bce = bce_loss_fn(logits, targets)
    bce = (bce * valid).sum() / (valid.sum() * logits[0, 0].numel() + 1e-8)

    probs = torch.sigmoid(logits)
    valid_probs = probs * valid
    valid_targets = targets * valid
    inter = (valid_probs * valid_targets).sum()
    dice = 1 - (2 * inter + 1.0) / (valid_probs.sum() + valid_targets.sum() + 1.0)
    return 0.7 * bce + 0.3 * dice


def _timestamped_checkpoint_path(mode: str) -> Path:
    stem = ACTIVE_MODEL_PATHS[mode].stem
    suffix = ACTIVE_MODEL_PATHS[mode].suffix
    import datetime as _dt

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / f"{stem}_{stamp}{suffix}"


def train_temporal_auto(mode: str, files: list[str], epochs: int = 30, promote_to_active: bool = False) -> TrainResult:
    _ensure_dirs()
    device = best_device()
    if mode == "2d":
        sequences, masks, valid_masks, ann_count = _build_train_sequences_2d(files)
        if not sequences:
            raise ValueError("No annotated 2D temporal-auto training data found.")
        model = TinyUNet2D(in_ch=3, out_ch=3).to(device)
        active_path = ACTIVE_MODEL_PATHS["2d"]
        base_init_path = MODELS_DIR / "filament_unet.pt"
        ds = SegDataset2DTemporalAuto(sequences, masks, valid_masks, augment_factor=10)
        batch_size = 8
        log_path = MODELS_DIR / "train_2d_temporal_auto_log.csv"
    elif mode == "3d":
        sequences, masks, valid_masks, ann_count = _build_train_sequences_3d(files)
        if not sequences:
            raise ValueError("No annotated 3D temporal-auto training data found.")
        model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
        active_path = ACTIVE_MODEL_PATHS["3d"]
        base_init_path = MODELS_DIR / "filament_unet3d.pt"
        ds = SegDataset3DTemporalAuto(sequences, masks, valid_masks, augment_factor=10, intensity_thresh=0.4)
        batch_size = 4
        log_path = MODELS_DIR / "train_3d_temporal_auto_log.csv"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if base_init_path.exists():
        state_dict = torch.load(base_init_path, map_location=device, weights_only=True)
        new_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if mode == "2d" and k == "enc1.block.0.weight":
                new_state_dict[k] = v.repeat(1, 3, 1, 1) / 3.0
            elif mode == "2d" and k == "out_conv.weight":
                new_state_dict[k] = v.repeat(3, 1, 1, 1)
            elif mode == "2d" and k == "out_conv.bias":
                new_state_dict[k] = v.repeat(3)
            elif mode == "3d" and k == "enc1.block.0.weight":
                new_state_dict[k] = v.repeat(1, 3, 1, 1, 1) / 3.0
            elif mode == "3d" and k == "out_conv.weight":
                new_state_dict[k] = v.repeat(3, 1, 1, 1, 1)
            elif mode == "3d" and k == "out_conv.bias":
                new_state_dict[k] = v.repeat(3)
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    if mode == "2d":
        pos_px = sum((m * v[:, None, None]).sum() for m, v in zip(masks, valid_masks))
        tot_px = sum(v.sum() * m[0].size for m, v in zip(masks, valid_masks))
    else:
        pos_px = sum((m * v[:, None, None, None]).sum() for m, v in zip(masks, valid_masks))
        tot_px = sum(v.sum() * m[0].size for m, v in zip(masks, valid_masks))
    pos_px = max(float(pos_px), 1.0)
    tot_px = float(tot_px)
    pos_weight = torch.tensor([min((tot_px - pos_px) / pos_px, 10.0)], device=device)

    with open(log_path, "w", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "loss", "dice"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        batches = 0
        for batch_x, batch_y, batch_valid in dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_valid = batch_valid.to(device)
            if float(batch_valid.sum().item()) == 0:
                continue
            opt.zero_grad()
            out = model(batch_x)
            loss = _temporal_loss(out, batch_y, batch_valid, pos_weight)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = (torch.sigmoid(out) > 0.5).float()
                valid_pred = pred * batch_valid
                valid_y = batch_y * batch_valid
                dice = (2 * (valid_pred * valid_y).sum()) / (valid_pred.sum() + valid_y.sum() + 1e-8)
            total_loss += float(loss.item())
            total_dice += float(dice.item())
            batches += 1
        with open(log_path, "a", newline="") as handle:
            csv.writer(handle).writerow([epoch + 1, total_loss / max(batches, 1), total_dice / max(batches, 1)])

    checkpoint_path = _timestamped_checkpoint_path(mode)
    torch.save(model.state_dict(), checkpoint_path)
    if promote_to_active:
        shutil.copy2(checkpoint_path, active_path)

    return TrainResult(
        mode=mode,
        checkpoint_path=str(checkpoint_path.resolve()),
        active_checkpoint_path=str(active_path.resolve()),
        promoted_to_active=promote_to_active,
        epochs=epochs,
        num_sequences=len(sequences),
        num_annotations=ann_count,
        log_path=str(log_path.resolve()),
    )


def _load_model(mode: str, model_path: str | None = None) -> tuple[torch.nn.Module, torch.device, str]:
    device = best_device()
    resolved = Path(model_path) if model_path else ACTIVE_MODEL_PATHS[mode]
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    if mode == "2d":
        model = TinyUNet2D(in_ch=3, out_ch=3).to(device)
    else:
        model = TinyUNet3D(in_ch=3, out_ch=3).to(device)
    model.load_state_dict(torch.load(resolved, map_location=device, weights_only=True))
    model.eval()
    return model, device, str(resolved.resolve())


def infer_temporal_auto_2d(filepath: str, model_path: str | None = None) -> dict[str, np.ndarray | DatasetInfo | str]:
    info, norm, _ = load_dataset(filepath)
    if info.mode != "2d":
        raise ValueError(f"{filepath} is not a 2D TIFF")
    model, device, resolved_model_path = _load_model("2d", model_path)
    t, h, w = norm.shape
    acc_logits = np.zeros((t, h, w), dtype=np.float32)
    acc_counts = np.zeros(t, dtype=np.float32)
    with torch.no_grad():
        for idx in range(t):
            t_prev, t_next = max(0, idx - 1), min(t - 1, idx + 1)
            window = np.stack([norm[t_prev], norm[idx], norm[t_next]], axis=0)
            inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
            logits = model(inp).squeeze(0).cpu().numpy()
            acc_logits[t_prev] += logits[0]
            acc_counts[t_prev] += 1
            acc_logits[idx] += logits[1]
            acc_counts[idx] += 1
            acc_logits[t_next] += logits[2]
            acc_counts[t_next] += 1
    avg_logits = acc_logits / np.maximum(acc_counts[:, None, None], 1)
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    pred = (probs > 0.5).astype(np.uint8)
    return {
        "info": info,
        "raw_norm": norm[:, np.newaxis, :, :],
        "probabilities": probs[:, np.newaxis, :, :],
        "pred_masks": pred[:, np.newaxis, :, :],
        "model_path": resolved_model_path,
    }


def infer_temporal_auto_3d(filepath: str, model_path: str | None = None) -> dict[str, np.ndarray | DatasetInfo | str]:
    info, norm, brightfield = load_dataset(filepath)
    if info.mode != "3d":
        raise ValueError(f"{filepath} is not a 3D TIFF")
    model, device, resolved_model_path = _load_model("3d", model_path)
    t, z, h, w = norm.shape
    acc_logits = np.zeros((t, z, h, w), dtype=np.float32)
    acc_counts = np.zeros(t, dtype=np.float32)
    with torch.no_grad():
        for idx in range(t):
            t_prev, t_next = max(0, idx - 1), min(t - 1, idx + 1)
            window = np.stack([norm[t_prev], norm[idx], norm[t_next]], axis=0)
            inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
            logits = model(inp).squeeze(0).cpu().numpy()
            acc_logits[t_prev] += logits[0]
            acc_counts[t_prev] += 1
            acc_logits[idx] += logits[1]
            acc_counts[idx] += 1
            acc_logits[t_next] += logits[2]
            acc_counts[t_next] += 1
    avg_logits = acc_logits / np.maximum(acc_counts[:, None, None, None], 1)
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    pred = (probs > 0.5).astype(np.uint8)
    return {
        "info": info,
        "raw_norm": norm,
        "probabilities": probs,
        "pred_masks": pred,
        "brightfield": brightfield,
        "model_path": resolved_model_path,
    }


class CellTracker:
    def __init__(self, max_distance: float = 40.0):
        self.max_distance = max_distance
        self.next_id = 1
        self.centroids: dict[int, tuple[float, float]] = {}

    def update(self, new_centroids: dict[int, tuple[float, float]]) -> dict[int, int]:
        assignment: dict[int, int] = {}
        used: set[int] = set()
        for label, center in new_centroids.items():
            best_id = None
            best_dist = float("inf")
            for cid, prev_center in self.centroids.items():
                if cid in used:
                    continue
                dist = float(np.linalg.norm(np.array(center) - np.array(prev_center)))
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = cid
            if best_id is not None:
                assignment[label] = best_id
                self.centroids[best_id] = center
                used.add(best_id)
            else:
                assignment[label] = self.next_id
                self.centroids[self.next_id] = center
                used.add(self.next_id)
                self.next_id += 1
        return assignment


def _compute_label_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    centroids: dict[int, tuple[float, float]] = {}
    for label in np.unique(mask):
        if int(label) == 0:
            continue
        centroid = ndimage.center_of_mass(mask == label)
        if np.isnan(centroid[0]):
            continue
        centroids[int(label)] = (float(centroid[0]), float(centroid[1]))
    return centroids


def _segment_cells_2d_frame(frame_2d: np.ndarray, cellpose_model=None) -> np.ndarray:
    if cellpose_model is not None:
        masks, _, _ = cellpose_model.eval(frame_2d, diameter=25, channels=[0, 0])
        return masks.astype(np.uint16)
    binary = frame_2d > float(np.percentile(frame_2d, 70))
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))
    binary = ndimage.binary_closing(binary, structure=np.ones((5, 5)))
    labels, _ = ndimage.label(binary)
    return labels.astype(np.uint16)


def _get_cellpose_model():
    try:
        from cellpose import models

        return models.CellposeModel(
            model_type="cyto",
            gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
        )
    except Exception:
        return None


def postprocess_2d_tracking(filepath: str, inference: dict[str, np.ndarray | DatasetInfo | str], output_root: str | Path = RESULTS_ROOT) -> InferenceArtifact:
    info = inference["info"]
    raw_norm = inference["raw_norm"][:, 0]
    probs = inference["probabilities"][:, 0]
    pred = inference["pred_masks"][:, 0]
    output_root = Path(output_root)
    tracking_dir = output_root / "tracking_csvs"
    mask_dir = output_root / "masks"
    cell_mask_dir = output_root / "cell_masks"
    for path in [tracking_dir, mask_dir, cell_mask_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base = Path(filepath).stem
    tracker = CellTracker()
    cellpose_model = _get_cellpose_model()
    frame_cell_masks: list[np.ndarray] = []
    frame_filament_masks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for t in range(info.timepoints):
        cell_mask = _segment_cells_2d_frame(raw_norm[t], cellpose_model)
        label_centroids = _compute_label_centroids(cell_mask)
        label_to_global = tracker.update(label_centroids)
        global_mask = np.zeros_like(cell_mask, dtype=np.uint16)
        for label, gid in label_to_global.items():
            global_mask[cell_mask == label] = gid
        filament_mask = global_mask.copy()
        filament_mask[pred[t] == 0] = 0
        frame_cell_masks.append(global_mask)
        frame_filament_masks.append(filament_mask)
        y_idx, x_idx = np.where(filament_mask > 0)
        for y, x in zip(y_idx, x_idx):
            rows.append(
                {
                    "frame": t,
                    "filament_id": int(filament_mask[y, x]),
                    "z": 0,
                    "y": int(y),
                    "x": int(x),
                    "raw_intensity": float(raw_norm[t, y, x]),
                    "probability": float(probs[t, y, x]),
                }
            )

    mask_stack = np.stack(frame_filament_masks, axis=0)[:, np.newaxis, :, :]
    cell_stack = np.stack(frame_cell_masks, axis=0)[:, np.newaxis, :, :]
    mask_path = mask_dir / f"{base}_mask.tif"
    cell_mask_path = cell_mask_dir / f"{base}_mask.tif"
    csv_path = tracking_dir / f"{base}_tracking.csv"
    tifffile.imwrite(mask_path, mask_stack.astype(np.uint16), imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(cell_mask_path, cell_stack.astype(np.uint16), imagej=True, metadata={"axes": "TZYX"})
    df = pd.DataFrame(rows, columns=["frame", "filament_id", "z", "y", "x", "raw_intensity", "probability"])
    df.to_csv(csv_path, index=False)
    return InferenceArtifact(
        source_path=str(Path(filepath).resolve()),
        mode="2d",
        mask_tiff=str(mask_path.resolve()),
        cell_mask_tiff=str(cell_mask_path.resolve()),
        tracking_csv=str(csv_path.resolve()),
        raw_shape=tuple(int(v) for v in raw_norm.shape),
    )


def _identify_pillar_ids(track_stats: dict[int, dict[str, list[float]]]) -> list[int]:
    stats: list[dict[str, float]] = []
    for global_id, values in track_stats.items():
        if not values["area"] or not values["mean_signal"]:
            continue
        stats.append(
            {
                "global_id": float(global_id),
                "mean_area": float(np.mean(values["area"])),
                "area_std": float(np.std(values["area"])),
                "mean_signal": float(np.mean(values["mean_signal"])),
                "centroid_motion": float(np.std(values["centroid_y"]) + np.std(values["centroid_x"])),
            }
        )
    if len(stats) < 2:
        return []
    best_pair = None
    best_score = None
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            a, b = stats[i], stats[j]
            pair_signal = a["mean_signal"] + b["mean_signal"]
            area_similarity = abs(a["mean_area"] - b["mean_area"])
            stability_penalty = a["area_std"] + b["area_std"] + a["centroid_motion"] + b["centroid_motion"]
            score = (pair_signal, area_similarity + stability_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (int(a["global_id"]), int(b["global_id"]))
    return list(best_pair) if best_pair is not None else []


def postprocess_3d_tracking(filepath: str, inference: dict[str, np.ndarray | DatasetInfo | str], output_root: str | Path = RESULTS_ROOT) -> InferenceArtifact:
    info = inference["info"]
    raw_norm = inference["raw_norm"]
    probs = inference["probabilities"]
    pred = inference["pred_masks"]
    brightfield = inference.get("brightfield")
    output_root = Path(output_root)
    tracking_dir = output_root / "tracking_csvs"
    mask_dir = output_root / "masks"
    cell_mask_dir = output_root / "cell_masks"
    for path in [tracking_dir, mask_dir, cell_mask_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base = Path(filepath).stem
    tracker = CellTracker()
    cellpose_model = _get_cellpose_model()
    frame_masks: list[np.ndarray] = []
    frame_filament_masks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    track_stats: dict[int, dict[str, list[float]]] = {}

    for t in range(info.timepoints):
        ref_2d = brightfield[t, info.z_planes // 2] if brightfield is not None else raw_norm[t, info.z_planes // 2]
        cell_mask = _segment_cells_2d_frame(ref_2d, cellpose_model)
        label_centroids = _compute_label_centroids(cell_mask)
        label_to_global = tracker.update(label_centroids)
        global_mask = np.zeros_like(cell_mask, dtype=np.uint16)
        for label, gid in label_to_global.items():
            global_mask[cell_mask == label] = gid
            region = cell_mask == label
            values = raw_norm[t][:, region].reshape(-1)
            centroid = ndimage.center_of_mass(region)
            stats = track_stats.setdefault(
                gid,
                {"area": [], "mean_signal": [], "centroid_y": [], "centroid_x": []},
            )
            stats["area"].append(float(region.sum()))
            stats["mean_signal"].append(float(values.mean()) if values.size else 0.0)
            stats["centroid_y"].append(float(centroid[0]))
            stats["centroid_x"].append(float(centroid[1]))
        frame_mask = np.repeat(global_mask[np.newaxis], info.z_planes, axis=0).astype(np.uint16)
        filament_frame_mask = frame_mask.copy()
        filament_frame_mask[pred[t] == 0] = 0
        frame_masks.append(frame_mask)
        frame_filament_masks.append(filament_frame_mask)
        z_idx, y_idx, x_idx = np.where(filament_frame_mask > 0)
        for z, y, x in zip(z_idx, y_idx, x_idx):
            rows.append(
                {
                    "frame": t,
                    "filament_id": int(filament_frame_mask[z, y, x]),
                    "z": int(z),
                    "y": int(y),
                    "x": int(x),
                    "raw_intensity": float(raw_norm[t, z, y, x]),
                    "probability": float(probs[t, z, y, x]),
                }
            )

    pillar_ids = _identify_pillar_ids(track_stats)
    mask_stack = np.stack(frame_filament_masks, axis=0).astype(np.uint16)
    cell_stack = np.stack(frame_masks, axis=0).astype(np.uint16)
    for pillar_id in pillar_ids:
        mask_stack[mask_stack == pillar_id] = 0
        cell_stack[cell_stack == pillar_id] = 0

    mask_path = mask_dir / f"{base}_mask.tif"
    cell_mask_path = cell_mask_dir / f"{base}_mask.tif"
    csv_path = tracking_dir / f"{base}_tracking.csv"
    tifffile.imwrite(mask_path, mask_stack, imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(cell_mask_path, cell_stack, imagej=True, metadata={"axes": "TZYX"})
    df = pd.DataFrame(rows, columns=["frame", "filament_id", "z", "y", "x", "raw_intensity", "probability"])
    if pillar_ids and not df.empty:
        df = df[~df["filament_id"].isin(pillar_ids)]
    df.to_csv(csv_path, index=False)
    return InferenceArtifact(
        source_path=str(Path(filepath).resolve()),
        mode="3d",
        mask_tiff=str(mask_path.resolve()),
        cell_mask_tiff=str(cell_mask_path.resolve()),
        tracking_csv=str(csv_path.resolve()),
        raw_shape=tuple(int(v) for v in raw_norm.shape),
    )


def run_inference(filepath: str, model_path: str | None = None, output_root: str | Path = RESULTS_ROOT) -> InferenceArtifact:
    info = inspect_tiff(filepath)
    if info.mode == "2d":
        inference = infer_temporal_auto_2d(filepath, model_path=model_path)
        return postprocess_2d_tracking(filepath, inference, output_root=output_root)
    inference = infer_temporal_auto_3d(filepath, model_path=model_path)
    return postprocess_3d_tracking(filepath, inference, output_root=output_root)


def run_inference_many(files: list[str], model_path_2d: str | None = None, model_path_3d: str | None = None, output_root: str | Path = RESULTS_ROOT) -> list[InferenceArtifact]:
    artifacts: list[InferenceArtifact] = []
    for filepath in files:
        info = inspect_tiff(filepath)
        model_path = model_path_2d if info.mode == "2d" else model_path_3d
        artifacts.append(run_inference(filepath, model_path=model_path, output_root=output_root))
    return artifacts


def serialize_train_result(result: TrainResult) -> dict[str, object]:
    return asdict(result)


def serialize_artifact(artifact: InferenceArtifact) -> dict[str, object]:
    return asdict(artifact)
