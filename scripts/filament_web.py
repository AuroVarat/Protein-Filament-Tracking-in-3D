#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tifffile
from scipy import ndimage

from filament_pipeline import (
    RESULT_CELL_MASK_DIR,
    RESULT_MASK_DIR,
    annotation_summary,
    count_annotations,
    infer_temporal_auto_2d,
    infer_temporal_auto_3d,
    inspect_tiff,
    list_tiff_files,
    load_annotation,
    load_dataset,
    run_inference_many,
    save_annotation,
    serialize_artifact,
    serialize_train_result,
    train_temporal_auto,
)

PAGE_SIZE_DEFAULT = 16
APP_THEME = gr.themes.Default()
APP_CSS = """
:root {
  --bg: #0d1316;
  --bg-2: #11191d;
  --panel: #141d21;
  --panel-2: #19252b;
  --panel-3: #213038;
  --line: rgba(255,255,255,0.08);
  --text: #ecf1f3;
  --muted: #8aa0ab;
  --accent: #31d0aa;
  --accent-2: #ff7a59;
  --shadow: 0 22px 46px rgba(0,0,0,0.22);
}

body, .gradio-container {
  background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1620px !important;
  margin: 0 auto !important;
  padding: 18px 20px 28px !important;
}

.filament-hero {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 24px 26px 18px;
  background:
    radial-gradient(circle at top right, rgba(49,208,170,0.16), transparent 28%),
    radial-gradient(circle at left center, rgba(255,122,89,0.14), transparent 20%),
    linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}

.filament-title {
  margin: 0 !important;
  font-size: 30px !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em;
  color: var(--text) !important;
}

.filament-subtitle {
  margin-top: 8px !important;
  color: var(--muted) !important;
  font-size: 14px !important;
  line-height: 1.5;
}

.filament-chip-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 16px;
}

.filament-chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  background: rgba(255,255,255,0.035);
  font-size: 12px;
  color: var(--muted);
}

.filament-shell {
  width: 100%;
  margin: 0 auto;
}

.filament-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(180deg, var(--panel) 0%, rgba(20,29,33,0.96) 100%);
  padding: 16px;
  box-shadow: var(--shadow);
}

.filament-tight .wrap {
  gap: 8px !important;
}

.filament-card h3,
.filament-card h4,
.filament-card p,
.filament-card span,
.filament-card label,
.filament-card .prose {
  color: var(--text) !important;
}

.filament-section-title {
  margin: 0 0 4px !important;
  font-size: 15px !important;
  font-weight: 650 !important;
  color: var(--text) !important;
  letter-spacing: 0.01em;
}

.filament-section-copy {
  margin: 0 0 14px !important;
  color: var(--muted) !important;
  font-size: 12px !important;
  line-height: 1.45;
}

.filament-card .gr-form,
.filament-card .gr-group,
.filament-card .block {
  border-color: var(--line) !important;
}

.filament-card input,
.filament-card textarea,
.filament-card select {
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}

.filament-card textarea::placeholder,
.filament-card input::placeholder {
  color: var(--muted) !important;
}

.filament-card button {
  border-radius: 12px !important;
  border: 1px solid var(--line) !important;
  background: var(--panel-2) !important;
  color: var(--text) !important;
  transition: transform 0.12s ease, background 0.18s ease, border-color 0.18s ease;
}

.filament-card button:hover {
  transform: translateY(-1px);
  border-color: rgba(49,208,170,0.45) !important;
  background: var(--panel-3) !important;
}

.filament-card button.primary,
.filament-card button[class*="primary"] {
  background: linear-gradient(135deg, var(--accent) 0%, #279f81 100%) !important;
  color: #081110 !important;
  border-color: rgba(49,208,170,0.15) !important;
  font-weight: 700 !important;
}

.filament-card .gradio-slider input[type="range"] {
  accent-color: var(--accent) !important;
}

.filament-source-row {
  align-items: end !important;
  gap: 10px !important;
}

.filament-source-row > * {
  min-width: 0 !important;
}

.filament-timeline-wrap {
  margin-top: 2px;
}

.filament-timeline-meta,
.filament-timeline-empty {
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 8px;
}

.filament-timeline-strip {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(4px, 1fr);
  gap: 2px;
  align-items: end;
  min-height: 14px;
}

.filament-tick {
  height: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
}

.filament-tick.masked {
  background: linear-gradient(180deg, rgba(49,208,170,0.95), rgba(39,159,129,0.95));
}

.filament-tick.current {
  outline: 1px solid rgba(255,255,255,0.9);
  outline-offset: 1px;
  height: 12px;
}

.filament-upload button {
  min-height: 44px !important;
  white-space: nowrap !important;
}

.filament-card .tabs,
.filament-card .tabitem,
.gradio-container .tabs,
.gradio-container .tabitem {
  border-color: transparent !important;
}

.gradio-container .tab-nav {
  gap: 10px !important;
  border-bottom: none !important;
  margin-bottom: 18px !important;
}

.gradio-container .tab-nav button {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  color: var(--muted) !important;
  min-height: 40px !important;
  padding: 0 16px !important;
}

.gradio-container .tab-nav button.selected {
  color: var(--text) !important;
  border-color: rgba(49,208,170,0.38) !important;
  background: linear-gradient(180deg, rgba(49,208,170,0.16), rgba(49,208,170,0.08)) !important;
  box-shadow: inset 0 0 0 1px rgba(49,208,170,0.12);
}

.filament-browser table {
  font-size: 12px;
  background: transparent !important;
}

.filament-browser thead th {
  background: rgba(255,255,255,0.03) !important;
  color: var(--muted) !important;
  border-bottom: 1px solid var(--line) !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 11px;
}

.filament-browser tbody td {
  background: rgba(255,255,255,0.015) !important;
  color: var(--text) !important;
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}

.filament-status textarea,
.filament-status input {
  font-size: 12px !important;
  background: rgba(255,255,255,0.03) !important;
}

.filament-status textarea {
  min-height: 88px !important;
}

.filament-muted {
  color: var(--muted) !important;
  font-size: 12px !important;
}

.filament-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.filament-painter .image-container,
.filament-painter canvas,
.filament-painter img {
  border-radius: 16px !important;
  background: #0f171a !important;
}

.filament-plot,
.filament-plot .js-plotly-plot {
  border-radius: 18px;
  overflow: hidden;
}

@media (max-width: 1100px) {
  .gradio-container {
    padding: 14px !important;
  }
  .filament-title {
    font-size: 25px !important;
  }
}
"""


def _normalize_rgb(image_2d: np.ndarray) -> np.ndarray:
    rgb = np.stack([image_2d] * 3, axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _outline(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(mask > 0, iterations=1) ^ (mask > 0)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _blend_overlay(background_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    background = background_rgb.astype(np.float32)
    alpha = (overlay_rgba[..., 3:4].astype(np.float32) / 255.0)
    overlay_rgb = overlay_rgba[..., :3].astype(np.float32)
    blended = background * (1.0 - alpha) + overlay_rgb * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _summary_rows(files: list[str]) -> list[list[object]]:
    return [
        [row["file"], row["mode"], row["frames"], row["z_planes"], row["annotations"], row["path"]]
        for row in annotation_summary(files)
    ]


def _summary_lookup(files: list[str]) -> dict[str, dict[str, object]]:
    rows = annotation_summary(files)
    return {str(row["path"]): row for row in rows}


def _sort_files_for_browser(files: list[str]) -> list[str]:
    lookup = _summary_lookup(files)
    return sorted(
        files,
        key=lambda filepath: (
            0 if int(lookup[str(filepath)]["annotations"]) > 0 else 1,
            -int(lookup[str(filepath)]["annotations"]),
            Path(filepath).name.lower(),
        ),
    )


def _page_count(total_items: int, page_size: int) -> int:
    if total_items <= 0:
        return 1
    return max(1, math.ceil(total_items / max(page_size, 1)))


def _clamp_page(page: int, total_items: int, page_size: int) -> int:
    return max(0, min(int(page), _page_count(total_items, page_size) - 1))


def _filter_files(files: list[str], search_text: str, mode_filter: str) -> list[str]:
    needle = (search_text or "").strip().lower()
    filtered: list[str] = []
    for filepath in files:
        info = inspect_tiff(filepath)
        if mode_filter and mode_filter != "all" and info.mode != mode_filter:
            continue
        label = Path(filepath).name.lower()
        if needle and needle not in label:
            continue
        filtered.append(filepath)
    return filtered


def _paged_files(files: list[str], page: int, page_size: int) -> list[str]:
    start = _clamp_page(page, len(files), page_size) * page_size
    end = start + page_size
    return files[start:end]


def _browser_rows(page_files: list[str], selected_files: list[str], summary_by_path: dict[str, dict[str, object]]) -> list[list[object]]:
    selected = set(selected_files)
    rows: list[list[object]] = []
    for filepath in page_files:
        summary = summary_by_path[str(filepath)]
        rows.append(
            [
                str(filepath) in selected,
                summary["file"],
                summary["mode"],
                summary["frames"],
                summary["z_planes"],
                summary["annotations"],
            ]
        )
    return rows


def _selection_status(selected_files: list[str], filtered_files: list[str], page: int, page_size: int) -> str:
    total_pages = _page_count(len(filtered_files), page_size)
    visible_count = len(_paged_files(filtered_files, page, page_size))
    if not filtered_files:
        return f"Selected {len(selected_files)} file(s) | No visible files"
    return f"Selected {len(selected_files)} file(s) | Showing {visible_count} on page {page + 1}/{total_pages} | Filtered total {len(filtered_files)}"


def _label_choice_update(filtered_files: list[str], current_label: str | None):
    if not filtered_files:
        return gr.update(choices=[], value=None)
    value = current_label if current_label in filtered_files else filtered_files[0]
    return gr.update(choices=filtered_files, value=value)


def _selected_choice_update(selected_files: list[str], current_label: str | None):
    if not selected_files:
        return gr.update(choices=[], value=None)
    value = current_label if current_label in selected_files else selected_files[0]
    return gr.update(choices=selected_files, value=value)


def _build_browser_view(all_files: list[str], selected_files: list[str], search_text: str, mode_filter: str, page: int, page_size: int, current_label: str | None):
    filtered = _filter_files(all_files, search_text, mode_filter)
    page = _clamp_page(page, len(filtered), page_size)
    page_files = _paged_files(filtered, page, page_size)
    summary_by_path = _summary_lookup(all_files)
    browser_rows = _browser_rows(page_files, selected_files, summary_by_path)
    page_text = f"Page {page + 1}/{_page_count(len(filtered), page_size)}"
    selected_summary_rows = _summary_rows(selected_files)
    selection_text = _selection_status(selected_files, filtered, page, page_size)
    return {
        "filtered_files": filtered,
        "browser_rows": browser_rows,
        "page": page,
        "page_text": page_text,
        "selected_summary_rows": selected_summary_rows,
        "selection_text": selection_text,
        "page_files": page_files,
        "label_update": _label_choice_update(filtered, current_label),
        "selected_label_update": _selected_choice_update(selected_files, current_label),
    }


def _painter_value(raw_norm: np.ndarray, annotation: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    if mode == "2d":
        background = _normalize_rgb(raw_norm)
        mask_plane = annotation
    else:
        background = _normalize_rgb(np.hstack([raw_norm[z] for z in range(raw_norm.shape[0])]))
        mask_plane = np.hstack([annotation[z] for z in range(annotation.shape[0])])
    rgba = np.zeros((*mask_plane.shape, 4), dtype=np.uint8)
    rgba[mask_plane > 0] = [0, 255, 0, 160]
    return {"background": background, "layers": [rgba], "composite": _blend_overlay(background, rgba)}


def _masked_frame_indices(filepath: str) -> list[int]:
    info = inspect_tiff(filepath)
    expected_shape = (info.height, info.width) if info.mode == "2d" else (info.z_planes, info.height, info.width)
    masked: list[int] = []
    for frame_idx in range(info.timepoints):
        try:
            if load_annotation(filepath, info.mode, frame_idx, expected_shape).max() > 0:
                masked.append(frame_idx)
        except Exception:
            continue
    return masked


def _mask_timeline_html(total_frames: int, masked_frames: list[int], current_frame: int) -> str:
    if total_frames <= 0:
        return '<div class="filament-timeline-empty">No frames loaded.</div>'
    masked = set(masked_frames)
    bars: list[str] = []
    for idx in range(total_frames):
        classes = ["filament-tick"]
        if idx in masked:
            classes.append("masked")
        if idx == current_frame:
            classes.append("current")
        bars.append(f'<div class="{" ".join(classes)}" title="Frame {idx}"></div>')
    return f"""
    <div class="filament-timeline-wrap">
      <div class="filament-timeline-meta">Frames with saved masks are highlighted in green.</div>
      <div class="filament-timeline-strip">{''.join(bars)}</div>
    </div>
    """


def _mask_from_editor(editor_value: dict | None, info, frame_idx: int) -> np.ndarray:
    expected_shape = (info.height, info.width) if info.mode == "2d" else (info.z_planes, info.height, info.width)
    if not editor_value:
        return np.zeros(expected_shape, dtype=np.float32)
    layers = editor_value.get("layers") or []
    if not layers:
        return np.zeros(expected_shape, dtype=np.float32)
    combined = np.zeros(layers[0].shape[:2], dtype=np.float32)
    for layer in layers:
        if layer.shape[-1] == 4:
            combined = np.maximum(combined, layer[:, :, 3] / 255.0)
    binary = (combined > 0.1).astype(np.float32)
    if info.mode == "2d":
        return binary
    volume = np.zeros((info.z_planes, info.height, info.width), dtype=np.float32)
    for z in range(info.z_planes):
        start = z * info.width
        end = (z + 1) * info.width
        volume[z] = binary[:, start:end]
    return volume


def _discover_files(source_path: str, upload_path: str | None) -> tuple[list[str], str]:
    inputs: list[str] = []
    if source_path:
        inputs.append(source_path)
    if upload_path:
        inputs.append(upload_path)
    files = _sort_files_for_browser(list_tiff_files(inputs))
    status = f"Found {len(files)} TIFF file(s)." if files else "No TIFF files found."
    return files, status


def _normalize_upload_path(upload_value) -> str | None:
    if upload_value is None:
        return None
    if isinstance(upload_value, str):
        return upload_value
    if isinstance(upload_value, list):
        if not upload_value:
            return None
        first = upload_value[0]
        if isinstance(first, str):
            return first
        return getattr(first, "name", None) or getattr(first, "path", None) or str(first)
    return getattr(upload_value, "name", None) or getattr(upload_value, "path", None) or str(upload_value)


def _on_load_source(source_path: str, upload_path: str | None):
    files, status = _discover_files(source_path, _normalize_upload_path(upload_path))
    summary_lookup = _summary_lookup(files)
    preselected = [filepath for filepath in files if int(summary_lookup[str(filepath)]["annotations"]) > 0]
    view = _build_browser_view(
        all_files=files,
        selected_files=preselected,
        search_text="",
        mode_filter="all",
        page=0,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=files[0] if files else None,
    )
    summary_rows = _summary_rows(files)
    return (
        files,
        view["filtered_files"],
        preselected,
        0,
        gr.update(visible=False),
        gr.update(visible=bool(files)),
        view["browser_rows"],
        view["page_text"],
        view["selection_text"],
        view["page_files"],
        view["label_update"],
        view["selected_label_update"],
        summary_rows,
        summary_rows,
        status,
    )


def _show_source_stage():
    return gr.update(visible=True), gr.update(visible=False)


def _refresh_browser(all_files: list[str], selected_files: list[str], search_text: str, mode_filter: str, page: int, current_label: str | None):
    view = _build_browser_view(
        all_files=all_files,
        selected_files=selected_files,
        search_text=search_text,
        mode_filter=mode_filter,
        page=page,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=current_label,
    )
    return (
        view["filtered_files"],
        view["page"],
        view["browser_rows"],
        view["page_files"],
        view["page_text"],
        view["selection_text"],
        view["label_update"],
        view["selected_label_update"],
        view["selected_summary_rows"],
        view["selected_summary_rows"],
    )


def _change_page(delta: int, all_files: list[str], selected_files: list[str], search_text: str, mode_filter: str, page: int, current_label: str | None):
    filtered = _filter_files(all_files, search_text, mode_filter)
    next_page = _clamp_page(page + delta, len(filtered), PAGE_SIZE_DEFAULT)
    return _refresh_browser(all_files, selected_files, search_text, mode_filter, next_page, current_label)


def _replace_visible_selection(visible_rows: list[list[object]], selected_files: list[str], filtered_files: list[str], page: int):
    selected = set(selected_files)
    page_files = _paged_files(filtered_files, page, PAGE_SIZE_DEFAULT)
    for filepath in page_files:
        selected.discard(filepath)
    if isinstance(visible_rows, pd.DataFrame):
        row_values = visible_rows.values.tolist()
    elif visible_rows is None:
        row_values = []
    else:
        row_values = visible_rows
    for row, filepath in zip(row_values, page_files):
        is_selected = bool(row[0]) if row else False
        if is_selected:
            selected.add(filepath)
    return sorted(selected)


def _apply_browser_selection(visible_rows: list[list[object]], all_files: list[str], selected_files: list[str], filtered_files: list[str], page: int, search_text: str, mode_filter: str, current_label: str | None):
    next_selected = _replace_visible_selection(visible_rows, selected_files, filtered_files, page)
    view = _build_browser_view(
        all_files=all_files,
        selected_files=next_selected,
        search_text=search_text,
        mode_filter=mode_filter,
        page=page,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=current_label,
    )
    return next_selected, view["browser_rows"], view["selection_text"], view["selected_label_update"], view["selected_summary_rows"], view["selected_summary_rows"]


def _select_visible(all_files: list[str], selected_files: list[str], filtered_files: list[str], page: int, search_text: str, mode_filter: str, current_label: str | None):
    page_files = _paged_files(filtered_files, page, PAGE_SIZE_DEFAULT)
    next_selected = sorted(set(selected_files).union(page_files))
    view = _build_browser_view(
        all_files=all_files,
        selected_files=next_selected,
        search_text=search_text,
        mode_filter=mode_filter,
        page=page,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=current_label,
    )
    return next_selected, view["browser_rows"], view["selection_text"], view["selected_label_update"], view["selected_summary_rows"], view["selected_summary_rows"]


def _clear_visible(all_files: list[str], selected_files: list[str], filtered_files: list[str], page: int, search_text: str, mode_filter: str, current_label: str | None):
    page_files = set(_paged_files(filtered_files, page, PAGE_SIZE_DEFAULT))
    next_selected = sorted([filepath for filepath in selected_files if filepath not in page_files])
    view = _build_browser_view(
        all_files=all_files,
        selected_files=next_selected,
        search_text=search_text,
        mode_filter=mode_filter,
        page=page,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=current_label,
    )
    return next_selected, view["browser_rows"], view["selection_text"], view["selected_label_update"], view["selected_summary_rows"], view["selected_summary_rows"]


def _clear_all_selection(all_files: list[str], search_text: str, mode_filter: str, page: int, current_label: str | None):
    view = _build_browser_view(
        all_files=all_files,
        selected_files=[],
        search_text=search_text,
        mode_filter=mode_filter,
        page=page,
        page_size=PAGE_SIZE_DEFAULT,
        current_label=current_label,
    )
    return [], view["browser_rows"], view["selection_text"], view["selected_label_update"], view["selected_summary_rows"], view["selected_summary_rows"]


def _browser_pick_file(page_files: list[str], evt: gr.SelectData):
    if not page_files:
        return gr.update(value=None)
    row_index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
    row_index = int(row_index)
    if row_index < 0 or row_index >= len(page_files):
        return gr.update()
    return gr.update(value=page_files[row_index])


def _load_label_file(filepath: str):
    if not filepath:
        return (
            gr.update(maximum=0, value=0),
            None,
            '<div class="filament-timeline-empty">Select a TIFF to see frame mask coverage.</div>',
            "Select a TIFF to begin labeling.",
            {},
        )
    info, data, _ = load_dataset(filepath)
    annotation = load_annotation(
        filepath,
        info.mode,
        0,
        (info.height, info.width) if info.mode == "2d" else (info.z_planes, info.height, info.width),
    )
    editor_value = _painter_value(data[0], annotation, info.mode)
    masked_frames = _masked_frame_indices(filepath)
    status = (
        f"Loaded {Path(filepath).name} | mode={info.mode} | "
        f"T={info.timepoints} | Z={info.z_planes} | annotations={count_annotations(filepath, info.mode)} | masked frames={len(masked_frames)}"
    )
    return (
        gr.update(maximum=max(info.timepoints - 1, 0), value=0),
        editor_value,
        _mask_timeline_html(info.timepoints, masked_frames, 0),
        status,
        {"filepath": filepath, "info": info.__dict__},
    )


def _change_label_frame(file_state: dict, frame_idx: int):
    if not file_state:
        return None, '<div class="filament-timeline-empty">No TIFF selected.</div>', "No TIFF selected."
    filepath = file_state["filepath"]
    info, data, _ = load_dataset(filepath)
    annotation = load_annotation(
        filepath,
        info.mode,
        int(frame_idx),
        (info.height, info.width) if info.mode == "2d" else (info.z_planes, info.height, info.width),
    )
    return _painter_value(data[int(frame_idx)], annotation, info.mode), _mask_timeline_html(info.timepoints, _masked_frame_indices(filepath), int(frame_idx)), f"Frame {int(frame_idx)} loaded."


def _save_label(editor_value: dict, file_state: dict, frame_idx: int, clear_legacy: bool):
    if not file_state:
        return "No TIFF selected.", '<div class="filament-timeline-empty">No TIFF selected.</div>'
    filepath = file_state["filepath"]
    info = inspect_tiff(filepath)
    mask = _mask_from_editor(editor_value, info, int(frame_idx))
    save_path = save_annotation(filepath, info.mode, int(frame_idx), mask, clear_legacy=clear_legacy)
    return f"Saved annotation: {save_path}", _mask_timeline_html(info.timepoints, _masked_frame_indices(filepath), int(frame_idx))


def _clear_label(file_state: dict, frame_idx: int):
    if not file_state:
        return None, "No TIFF selected."
    filepath = file_state["filepath"]
    info, data, _ = load_dataset(filepath)
    frame_idx = int(frame_idx)
    blank = np.zeros((info.height, info.width), dtype=np.float32) if info.mode == "2d" else np.zeros((info.z_planes, info.height, info.width), dtype=np.float32)
    return _painter_value(data[frame_idx], blank, info.mode), "Cleared current editor layer. Save to persist the blank mask."


def _train_selected(files: list[str], epochs: int, promote_to_active: bool):
    if not files:
        return [], "Select at least one TIFF for training.", gr.Tabs(selected="train-tab"), "", ""
    modes: dict[str, list[str]] = {"2d": [], "3d": []}
    for filepath in files:
        modes[inspect_tiff(filepath).mode].append(filepath)
    results = []
    for mode in ["2d", "3d"]:
        if not modes[mode]:
            continue
        result = train_temporal_auto(mode, modes[mode], epochs=epochs, promote_to_active=promote_to_active)
        results.append(serialize_train_result(result))
    if not results:
        return [], "No annotated files were eligible for training.", gr.Tabs(selected="train-tab"), "", ""
    lines = [f"{item['mode']}: {item['checkpoint_path']}" for item in results]
    model_2d = next((item["checkpoint_path"] for item in results if item["mode"] == "2d"), "")
    model_3d = next((item["checkpoint_path"] for item in results if item["mode"] == "3d"), "")
    return results, "\n".join(lines), gr.Tabs(selected="inference-tab"), model_2d, model_3d


def _find_raw_source_for_mask(mask_tiff: str) -> str | None:
    stem = Path(mask_tiff).stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    for root in [Path("tifs2d"), Path("tiffs3d")]:
        for suffix in [".tif", ".tiff"]:
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return str(candidate.resolve())
    return None


def _load_result_state(mask_tiff: str):
    if not mask_tiff:
        return {}, gr.update(maximum=0, value=0), [], "Select a result mask TIFF."
    mask_arr = tifffile.imread(mask_tiff)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, np.newaxis, :, :]
    if mask_arr.ndim != 4:
        raise ValueError(f"Expected TZYX mask TIFF, got {mask_arr.shape}")
    raw_source = _find_raw_source_for_mask(mask_tiff)
    raw_norm = None
    mode = "3d"
    if raw_source:
        info, raw_data, _ = load_dataset(raw_source)
        mode = info.mode
        raw_norm = raw_data[:, np.newaxis, :, :] if info.mode == "2d" else raw_data
    cell_mask_path = Path(RESULT_CELL_MASK_DIR) / Path(mask_tiff).name
    cell_mask = tifffile.imread(cell_mask_path) if cell_mask_path.exists() else None
    if cell_mask is not None and cell_mask.ndim == 3:
        cell_mask = cell_mask[:, np.newaxis, :, :]
    csv_path = Path("results/tracking_csvs") / Path(mask_tiff).name.replace("_mask.tif", "_tracking.csv")
    state = {
        "mask_tiff": str(Path(mask_tiff).resolve()),
        "raw_source": raw_source,
        "mode": mode if mask_arr.shape[1] > 1 else "2d",
        "mask": mask_arr,
        "cell_mask": cell_mask,
        "raw_norm": raw_norm,
        "csv_paths": [str(csv_path.resolve())] if csv_path.exists() else [],
    }
    frame_max = max(mask_arr.shape[0] - 1, 0)
    path_rows = [[p] for p in state["csv_paths"]]
    status = f"Loaded result {Path(mask_tiff).name} | mode={state['mode']} | frames={mask_arr.shape[0]}"
    return state, gr.update(maximum=frame_max, value=0), path_rows, status


def _render_result_view(result_state: dict, frame_idx: int, show_raw: bool, show_mask: bool, show_outline: bool, point_size: float):
    if not result_state:
        return None, None, None, None, "Load a result stack first."
    t = int(frame_idx)
    mask = result_state["mask"][t]
    cell_mask = result_state["cell_mask"][t] if result_state.get("cell_mask") is not None else None
    raw = result_state["raw_norm"][t] if result_state.get("raw_norm") is not None else np.zeros_like(mask, dtype=np.float32)
    mode = result_state["mode"]
    if mode == "2d":
        base = raw[0] if raw.ndim == 3 else raw
        image = np.zeros((base.shape[0], base.shape[1], 3), dtype=np.float32)
        if show_raw:
            image = np.stack([base] * 3, axis=-1)
        if show_mask:
            image[mask[0] > 0] = [0.0, 1.0, 0.0]
        if show_outline and cell_mask is not None:
            outline = _outline(cell_mask[0] > 0)
            image[outline] = [1.0, 0.55, 0.0]
        figure = go.Figure(go.Image(z=_to_uint8_rgb(image)))
        figure.update_layout(template="plotly_dark", title="2D view", height=420, margin={"l": 0, "r": 0, "t": 30, "b": 0})
        status = f"2D frame={t} labels={int(mask.max())}"
        return (_to_uint8_rgb(image), None, None, figure, status)

    xy = np.max(raw if show_raw else np.zeros_like(raw), axis=0)
    xz = np.max(np.transpose(raw, (1, 0, 2)) if show_raw else np.zeros((raw.shape[1], raw.shape[0], raw.shape[2]), dtype=np.float32), axis=2)
    yz = np.max(np.transpose(raw, (2, 0, 1)) if show_raw else np.zeros((raw.shape[2], raw.shape[0], raw.shape[1]), dtype=np.float32), axis=2)
    xy_rgb = np.stack([xy] * 3, axis=-1)
    xz_rgb = np.stack([xz] * 3, axis=-1)
    yz_rgb = np.stack([yz] * 3, axis=-1)
    if show_mask:
        # Fix: transpose boolean mask for xz_rgb to match (128, 5)
        xy_rgb[np.max(mask, axis=0) > 0] = [0.0, 1.0, 0.0]
        xz_rgb[(np.max(mask, axis=1) > 0).T] = [0.0, 1.0, 0.0]
        yz_rgb[np.max(mask, axis=2).T > 0] = [0.0, 1.0, 0.0]
    if show_outline and cell_mask is not None:
        xy_rgb[_outline(np.max(cell_mask, axis=0) > 0)] = [1.0, 0.55, 0.0]
        xz_rgb[_outline((np.max(cell_mask, axis=1) > 0).T)] = [1.0, 0.55, 0.0]
        yz_rgb[_outline(np.max(cell_mask, axis=2).T > 0)] = [1.0, 0.55, 0.0]

    figure = go.Figure()
    if show_raw:
        z, y, x = np.where(raw > np.percentile(raw, 95))
        if len(z) > 0:
            keep = slice(None, None, max(1, len(z) // 4000))
            figure.add_trace(
                go.Scatter3d(
                    x=x[keep],
                    y=y[keep],
                    z=z[keep],
                    mode="markers",
                    marker={"size": 1, "color": raw[z[keep], y[keep], x[keep]], "colorscale": "Greys", "opacity": 0.15},
                    name="raw",
                )
            )
    if show_mask:
        z, y, x = np.where(mask > 0)
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={"size": point_size, "color": "#31d0aa", "opacity": 0.75},
                name="mask",
            )
        )
    if show_outline and cell_mask is not None:
        edge = np.zeros_like(cell_mask, dtype=bool)
        for z_idx in range(cell_mask.shape[0]):
            edge[z_idx] = _outline(cell_mask[z_idx] > 0)
        z, y, x = np.where(edge)
        if len(z) > 0:
            figure.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={"size": max(1, point_size - 1), "color": "#ff7a59", "opacity": 0.6},
                    name="cell outline",
                )
            )
    figure.update_layout(template="plotly_dark", scene_aspectmode="data", height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0})
    status = f"3D frame={t} voxels={int((mask > 0).sum())}"
    return (_to_uint8_rgb(xy_rgb), _to_uint8_rgb(xz_rgb), _to_uint8_rgb(yz_rgb), figure, status)


def _run_inference(files: list[str], model_2d: str, model_3d: str):
    if not files:
        return [], gr.update(choices=[]), "Select TIFF files to run inference.", []
    artifacts = run_inference_many(
        files,
        model_path_2d=model_2d or None,
        model_path_3d=model_3d or None,
    )
    rows = [serialize_artifact(artifact) for artifact in artifacts]
    mask_choices = sorted(str(p.resolve()) for p in RESULT_MASK_DIR.glob("*.tif"))
    status = "\n".join(f"{item['mode']}: {Path(item['tracking_csv']).name}" for item in rows)
    csv_rows = [[item["tracking_csv"]] for item in rows]
    return rows, gr.update(choices=mask_choices, value=mask_choices[0] if mask_choices else None), status, csv_rows


def _result_choices(preferred: str | None = None) -> tuple[list[str], str | None]:
    choices = sorted(str(p.resolve()) for p in RESULT_MASK_DIR.glob("*.tif"))
    if preferred:
        preferred = str(Path(preferred).resolve())
    if preferred in choices:
        return choices, preferred
    return choices, (choices[0] if choices else None)


def _refresh_result_choices():
    choices, selected = _result_choices()
    return gr.update(choices=choices, value=selected)


def _sync_results_view(mask_tiff: str | None, show_raw: bool, show_mask: bool, show_outline: bool, point_size: float):
    choices, selected = _result_choices(mask_tiff)
    dropdown = gr.update(choices=choices, value=selected)
    if not selected:
        return dropdown, {}, gr.update(maximum=0, value=0), [], None, None, None, None, "No result mask TIFFs found in results/masks."
    state, frame_update, csv_rows, _ = _load_result_state(selected)
    xy, xz, yz, plot, status = _render_result_view(state, 0, show_raw, show_mask, show_outline, point_size)
    return dropdown, state, frame_update, csv_rows, xy, xz, yz, plot, status


KEYBINDINGS_SCRIPT = """
<script>
(() => {
  const bind = () => {
    if (window.__filamentKeysBound) return;
    window.__filamentKeysBound = true;
    const click = (id) => document.getElementById(id)?.click();
    const clickByText = (terms) => {
      const buttons = Array.from(document.querySelectorAll('button'));
      for (const button of buttons) {
        const txt = (button.innerText || button.getAttribute('aria-label') || button.title || '').toLowerCase();
        if (terms.some((term) => txt.includes(term))) {
          button.click();
          return true;
        }
      }
      return false;
    };
    window.addEventListener('keydown', (event) => {
      const tag = (event.target?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || event.target?.isContentEditable) return;
      if (event.key === 'ArrowLeft') { event.preventDefault(); click('label-prev-frame'); }
      if (event.key === 'ArrowRight') { event.preventDefault(); click('label-next-frame'); }
      if (event.key === 's' || event.key === 'S') { event.preventDefault(); click('label-save-mask'); }
      if (event.key === 'c' || event.key === 'C') { event.preventDefault(); click('label-clear-mask'); }
      if (event.key === 'e' || event.key === 'E') { event.preventDefault(); click('label-toggle-brush'); clickByText(['eraser', 'erase']); }
      if (event.key === 'b' || event.key === 'B') { event.preventDefault(); click('label-toggle-brush'); clickByText(['brush', 'draw']); }
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
</script>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Unified Temporal-Auto Filament Pipeline") as demo:
        gr.HTML(KEYBINDINGS_SCRIPT)
        gr.HTML(
            """
            <section class="filament-hero filament-shell">
              <h1 class="filament-title">Unified Temporal-Auto Filament Pipeline</h1>
              <p class="filament-subtitle">A darker, viewer-first workspace for browsing TIFFs, painting masks, training temporal models, running inference, and reviewing tracked results without drowning the page in controls.</p>
              <div class="filament-chip-row">
                <div class="filament-chip">Filter and page through datasets</div>
                <div class="filament-chip">Label 2D and multi-plane 3D in one app</div>
                <div class="filament-chip">Train and infer from the same selected set</div>
              </div>
            </section>
            """
        )
        train_result_state = gr.State(value=[])
        inference_records_state = gr.State(value=[])
        result_state = gr.State(value={})
        file_state = gr.State(value={})
        all_files_state = gr.State(value=[])
        filtered_files_state = gr.State(value=[])
        page_files_state = gr.State(value=[])
        selected_files_state = gr.State(value=[])
        browser_page_state = gr.State(value=0)

        with gr.Tabs(selected="label-tab") as app_tabs:
            with gr.Tab("Label", id="label-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=380, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Dataset Browser", elem_classes=["filament-section-title"])
                        gr.Markdown("Load a directory or TIFF, then curate a training set from a compact paged browser instead of an all-files wall.", elem_classes=["filament-section-copy"])
                        with gr.Column(visible=True) as source_stage:
                            with gr.Row(elem_classes=["filament-source-row"]):
                                source_path = gr.Textbox(
                                    label="TIFF directory or file path",
                                    placeholder="tifs2d or tiffs3d or /full/path/file.tif",
                                    scale=8,
                                )
                                upload_tiff = gr.UploadButton(
                                    "Upload",
                                    file_types=[".tif", ".tiff"],
                                    file_count="single",
                                    scale=2,
                                    elem_classes=["filament-upload"],
                                )
                                load_source_btn = gr.Button("Load", variant="primary", scale=2)
                        with gr.Column(visible=False) as browser_stage:
                            with gr.Row():
                                back_to_source_btn = gr.Button("Back", scale=1)
                                with gr.Column(scale=2, min_width=0):
                                    page_label = gr.Markdown("Page 1/1")
                            file_search = gr.Textbox(label="Search files", placeholder="Search the loaded dataset by filename")
                            mode_filter = gr.Radio(label="Filter", choices=["all", "2d", "3d"], value="all")
                            with gr.Row():
                                prev_page_btn = gr.Button("Prev Page")
                                next_page_btn = gr.Button("Next Page")
                            browser_selection_status = gr.Markdown("Selected 0 file(s)")
                            browser_table = gr.Dataframe(
                                label="Files on this page",
                                headers=["Train", "File", "Mode", "Frames", "Z", "Masks"],
                                datatype=["bool", "str", "str", "number", "number", "number"],
                                row_count=(PAGE_SIZE_DEFAULT, "fixed"),
                                column_count=(6, "fixed"),
                                interactive=True,
                                elem_classes=["filament-browser"],
                            )
                            gr.Markdown("Click a row to open that file on the painter. Tick the Train column to add or remove it from the selected training set.", elem_classes=["filament-section-copy"])
                            with gr.Row():
                                select_visible_btn = gr.Button("Select Visible")
                                clear_visible_btn = gr.Button("Clear Visible")
                                clear_all_btn = gr.Button("Clear All")
                            gr.Markdown("### Painter Target", elem_classes=["filament-section-title"])
                            gr.Markdown("Open a file from the current page or jump to one already in the selected training set.", elem_classes=["filament-section-copy"])
                            label_file = gr.Dropdown(label="From paged browser", choices=[])
                            selected_file_picker = gr.Dropdown(label="From selected training files", choices=[])
                            label_status = gr.Textbox(label="Label status", interactive=False, lines=3, elem_classes=["filament-status"])

                    with gr.Column(scale=1, min_width=360, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### File Selection", elem_classes=["filament-section-title"])
                        gr.Markdown("Select a TIFF file to annotate.", elem_classes=["filament-section-copy"])
                        label_file = gr.Dropdown(label="Select file to annotate", choices=[], interactive=True)
                        annotated_file_picker = gr.Dropdown(label="Files with Annotations", choices=[], interactive=True)
                        label_status = gr.Textbox(label="Label status", interactive=False, lines=3, elem_classes=["filament-status"])

                    with gr.Column(scale=2, min_width=820, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Annotation Stage", elem_classes=["filament-section-title"])
                        gr.Markdown("Use the painter as the main focus area. Keyboard navigation and save/clear shortcuts stay active.", elem_classes=["filament-section-copy"])
                        with gr.Row():
                            frame_slider = gr.Slider(label="Timepoint", minimum=0, maximum=0, step=1, value=0)
                            prev_frame = gr.Button("Prev", elem_id="label-prev-frame")
                            next_frame = gr.Button("Next", elem_id="label-next-frame")
                        frame_timeline = gr.HTML('<div class="filament-timeline-empty">Select a TIFF to see frame mask coverage.</div>')
                        with gr.Row():
                            brush_mode = gr.Radio(label="Painter mode", choices=["Brush", "Erase"], value="Brush", interactive=False)
                            toggle_brush = gr.Button("Toggle Brush/Erase", elem_id="label-toggle-brush")
                            clear_legacy = gr.Checkbox(label="Delete matching old legacy mask after save", value=False)
                        painter = gr.ImageEditor(
                            label="Temporal painter",
                            image_mode="RGBA",
                            type="numpy",
                            elem_id="filament-unified-editor",
                            elem_classes=["filament-painter"],
                            brush=gr.Brush(colors=["#00ff66"], color_mode="fixed"),
                        )
                        painter = gr.ImageEditor(
                            label="Temporal painter",
                            image_mode="RGBA",
                            type="numpy",
                            elem_id="filament-unified-editor",
                            elem_classes=["filament-painter"],
                            brush=gr.Brush(colors=["#00ff66"], color_mode="fixed"),
                        )
                        with gr.Row():
                            save_mask_btn = gr.Button("Save Mask", variant="primary", elem_id="label-save-mask")
                            clear_mask_btn = gr.Button("Clear Editor", elem_id="label-clear-mask")
                        label_summary = gr.Dataframe(
                            label="Selected training files",
                            headers=["file", "mode", "frames", "z_planes", "annotations", "path"],
                            datatype=["str"] * 6,
                            interactive=False,
                            max_height=220,
                        )

            with gr.Tab("Train", id="train-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=360, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Training Control", elem_classes=["filament-section-title"])
                        gr.Markdown("Train only the files in the curated selection set. The browser state from the Label tab is the source of truth.", elem_classes=["filament-section-copy"])
                        train_selection_status = gr.Markdown("Selected 0 file(s)")
                        epochs = gr.Slider(label="Epochs", minimum=1, maximum=100, step=1, value=30)
                        promote_active = gr.Checkbox(label="Promote trained checkpoint to active default path", value=False)
                        train_btn = gr.Button("Train Temporal-Auto Models", variant="primary")
                        train_status = gr.Textbox(label="Train status", interactive=False, lines=4, elem_classes=["filament-status"])
                    with gr.Column(scale=2, min_width=820, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Training Set and Outputs", elem_classes=["filament-section-title"])
                        gr.Markdown("Review which files will be used and where the resulting checkpoints are written.", elem_classes=["filament-section-copy"])
                        train_summary = gr.Dataframe(label="Training files", headers=["file", "mode", "frames", "z_planes", "annotations", "path"], datatype=["str"] * 6, interactive=False, max_height=320)
                        train_results_table = gr.Dataframe(
                            label="Training outputs",
                            headers=["mode", "checkpoint_path", "active_checkpoint_path", "promoted_to_active", "epochs", "num_sequences", "num_annotations", "log_path"],
                            datatype=["str", "str", "str", "bool", "number", "number", "number", "str"],
                            interactive=False,
                            max_height=260,
                        )

            with gr.Tab("Inference / Results", id="inference-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=360, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Inference Control", elem_classes=["filament-section-title"])
                        gr.Markdown("Run temporal-auto inference on the selected set, then inspect outputs from the canonical results directories.", elem_classes=["filament-section-copy"])
                        inference_selection_status = gr.Markdown("Selected 0 file(s)")
                        model_2d = gr.Textbox(label="2D checkpoint override", value="")
                        model_3d = gr.Textbox(label="3D checkpoint override", value="")
                        infer_btn = gr.Button("Run Inference + Tracker", variant="primary")
                        inference_status = gr.Textbox(label="Inference status", interactive=False, lines=4, elem_classes=["filament-status"])
                        csv_paths = gr.Dataframe(label="CSV outputs", headers=["csv_path"], datatype=["str"], interactive=False, max_height=220)
                        with gr.Row():
                            result_mask_choice = gr.Dropdown(label="Result mask TIFF", choices=_result_choices()[0], value=_result_choices()[1])
                            refresh_results = gr.Button("Refresh")
                        with gr.Row():
                            result_frame = gr.Slider(label="Result frame", minimum=0, maximum=0, step=1, value=0)
                            point_size = gr.Slider(label="3D point size", minimum=1, maximum=6, step=0.5, value=3)
                        with gr.Row():
                            show_raw = gr.Checkbox(label="Show normalized raw", value=True)
                            show_mask = gr.Checkbox(label="Show mask", value=True)
                            show_outline = gr.Checkbox(label="Show cell outline", value=True)
                        result_status = gr.Textbox(label="Viewer status", interactive=False, lines=3, elem_classes=["filament-status"])
                    with gr.Column(scale=2, min_width=820, elem_classes=["filament-card", "filament-tight"]):
                        gr.Markdown("### Viewer Stage", elem_classes=["filament-section-title"])
                        gr.Markdown("Result viewing follows the older mask viewer aesthetic: dark panels, strong projections, and a clear visual focus on the active frame.", elem_classes=["filament-section-copy"])
                        plot_3d = gr.Plot(label="3D / 2D result view")
                        with gr.Row():
                            xy = gr.Image(label="XY / 2D")
                            xz = gr.Image(label="XZ")
                            yz = gr.Image(label="YZ")

        load_source_btn.click(
            fn=_on_load_source,
            inputs=[source_path, upload_tiff],
            outputs=[
                all_files_state,
                filtered_files_state,
                selected_files_state,
                browser_page_state,
                source_stage,
                browser_stage,
                browser_table,
                page_label,
                browser_selection_status,
                page_files_state,
                label_file,
                selected_file_picker,
                train_summary,
                label_summary,
                label_status,
            ],
        )
        back_to_source_btn.click(_show_source_stage, outputs=[source_stage, browser_stage])
        file_search.change(
            fn=_refresh_browser,
            inputs=[all_files_state, selected_files_state, file_search, mode_filter, browser_page_state, label_file],
            outputs=[filtered_files_state, browser_page_state, browser_table, page_files_state, page_label, browser_selection_status, label_file, selected_file_picker, train_summary, label_summary],
        )
        mode_filter.change(
            fn=_refresh_browser,
            inputs=[all_files_state, selected_files_state, file_search, mode_filter, browser_page_state, label_file],
            outputs=[filtered_files_state, browser_page_state, browser_table, page_files_state, page_label, browser_selection_status, label_file, selected_file_picker, train_summary, label_summary],
        )
        prev_page_btn.click(
            fn=lambda all_files, selected_files, search_text, filter_mode, page, current_label: _change_page(-1, all_files, selected_files, search_text, filter_mode, page, current_label),
            inputs=[all_files_state, selected_files_state, file_search, mode_filter, browser_page_state, label_file],
            outputs=[filtered_files_state, browser_page_state, browser_table, page_files_state, page_label, browser_selection_status, label_file, selected_file_picker, train_summary, label_summary],
        )
        next_page_btn.click(
            fn=lambda all_files, selected_files, search_text, filter_mode, page, current_label: _change_page(1, all_files, selected_files, search_text, filter_mode, page, current_label),
            inputs=[all_files_state, selected_files_state, file_search, mode_filter, browser_page_state, label_file],
            outputs=[filtered_files_state, browser_page_state, browser_table, page_files_state, page_label, browser_selection_status, label_file, selected_file_picker, train_summary, label_summary],
        )
        browser_table.change(
            fn=_apply_browser_selection,
            inputs=[browser_table, all_files_state, selected_files_state, filtered_files_state, browser_page_state, file_search, mode_filter, label_file],
            outputs=[selected_files_state, browser_table, browser_selection_status, selected_file_picker, train_summary, label_summary],
        )
        select_visible_btn.click(
            fn=_select_visible,
            inputs=[all_files_state, selected_files_state, filtered_files_state, browser_page_state, file_search, mode_filter, label_file],
            outputs=[selected_files_state, browser_table, browser_selection_status, selected_file_picker, train_summary, label_summary],
        )
        clear_visible_btn.click(
            fn=_clear_visible,
            inputs=[all_files_state, selected_files_state, filtered_files_state, browser_page_state, file_search, mode_filter, label_file],
            outputs=[selected_files_state, browser_table, browser_selection_status, selected_file_picker, train_summary, label_summary],
        )
        clear_all_btn.click(
            fn=_clear_all_selection,
            inputs=[all_files_state, file_search, mode_filter, browser_page_state, label_file],
            outputs=[selected_files_state, browser_table, browser_selection_status, selected_file_picker, train_summary, label_summary],
        )
        browser_table.select(_browser_pick_file, inputs=page_files_state, outputs=label_file)
        selected_file_picker.change(lambda filepath: gr.update(value=filepath), inputs=selected_file_picker, outputs=label_file)
        label_file.change(_load_label_file, inputs=label_file, outputs=[frame_slider, painter, frame_timeline, label_status, file_state])
        frame_slider.change(_change_label_frame, inputs=[file_state, frame_slider], outputs=[painter, frame_timeline, label_status])
        prev_frame.click(lambda t: max(int(t) - 1, 0), inputs=frame_slider, outputs=frame_slider)
        next_frame.click(lambda t, state: min(int(t) + 1, inspect_tiff(state["filepath"]).timepoints - 1) if state else 0, inputs=[frame_slider, file_state], outputs=frame_slider)
        save_mask_btn.click(_save_label, inputs=[painter, file_state, frame_slider, clear_legacy], outputs=[label_status, frame_timeline])
        clear_mask_btn.click(_clear_label, inputs=[file_state, frame_slider], outputs=[painter, label_status])
        toggle_brush.click(lambda current: "Erase" if current == "Brush" else "Brush", inputs=brush_mode, outputs=brush_mode)
        selected_files_state.change(lambda files: f"Selected {len(files)} file(s) for training", inputs=selected_files_state, outputs=train_selection_status)
        selected_files_state.change(lambda files: f"Selected {len(files)} file(s) for inference", inputs=selected_files_state, outputs=inference_selection_status)

        train_btn.click(
            _train_selected,
            inputs=[selected_files_state, epochs, promote_active],
            outputs=[train_result_state, train_status, app_tabs, model_2d, model_3d],
        )
        train_result_state.change(
            lambda rows: [[
                r["mode"],
                r["checkpoint_path"],
                r["active_checkpoint_path"],
                r["promoted_to_active"],
                r["epochs"],
                r["num_sequences"],
                r["num_annotations"],
                r["log_path"],
            ] for r in rows],
            inputs=train_result_state,
            outputs=train_results_table,
        )
        infer_btn.click(_run_inference, inputs=[selected_files_state, model_2d, model_3d], outputs=[inference_records_state, result_mask_choice, inference_status, csv_paths]).then(
            _sync_results_view,
            inputs=[result_mask_choice, show_raw, show_mask, show_outline, point_size],
            outputs=[result_mask_choice, result_state, result_frame, csv_paths, xy, xz, yz, plot_3d, result_status],
        )
        refresh_results.click(
            _sync_results_view,
            inputs=[result_mask_choice, show_raw, show_mask, show_outline, point_size],
            outputs=[result_mask_choice, result_state, result_frame, csv_paths, xy, xz, yz, plot_3d, result_status],
        )
        result_mask_choice.change(_load_result_state, inputs=result_mask_choice, outputs=[result_state, result_frame, csv_paths, result_status]).then(
            _render_result_view,
            inputs=[result_state, result_frame, show_raw, show_mask, show_outline, point_size],
            outputs=[xy, xz, yz, plot_3d, result_status],
        )
        for trigger in [result_frame, show_raw, show_mask, show_outline, point_size]:
            trigger.change(
                _render_result_view,
                inputs=[result_state, result_frame, show_raw, show_mask, show_outline, point_size],
                outputs=[xy, xz, yz, plot_3d, result_status],
            )
        demo.load(
            _sync_results_view,
            inputs=[result_mask_choice, show_raw, show_mask, show_outline, point_size],
            outputs=[result_mask_choice, result_state, result_frame, csv_paths, xy, xz, yz, plot_3d, result_status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified temporal-auto filament web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app()
    launch_kwargs = {"server_name": args.host, "share": False, "theme": APP_THEME, "css": APP_CSS}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
