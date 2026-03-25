#!/usr/bin/env python3
"""
Interactive filament mask viewer for labeled 3D+time TIFF stacks.

This viewer reads the saved mask TIFFs directly. It does not rerun inference.
Each frame is shown as:
- a 3D voxel scatter colored by filament ID
- simple XY/XZ/YZ maximum projections for the selected timepoint
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import tifffile


DEFAULT_MASK_DIR = "results/masks"
BRIGHT_COLORS = [
    "#39ff14",  # neon green
    "#ff4fd8",  # pink
    "#ffe600",  # yellow
    "#00e5ff",  # cyan
    "#ff7a00",  # orange
    "#b266ff",  # purple
    "#ff1744",  # red
    "#00ffa6",  # mint
    "#8bc34a",  # lime
    "#00b0ff",  # blue
]

CACHE: dict[str, np.ndarray] = {}


def list_mask_files(mask_dir: str) -> list[str]:
    pattern = os.path.join(mask_dir, "*.tif")
    return sorted(glob.glob(pattern))


def display_name(path: str) -> str:
    return os.path.basename(path)


def resolve_mask_path(choice: str | None, uploaded_file) -> str | None:
    if uploaded_file is not None:
        file_path = getattr(uploaded_file, "name", None) or str(uploaded_file)
        return file_path
    return choice


def load_mask(path: str) -> np.ndarray:
    if path not in CACHE:
        arr = tifffile.imread(path)
        if arr.ndim != 4:
            raise ValueError(
                f"Expected mask TIFF with shape (T, Z, Y, X), got {arr.shape}."
            )
        CACHE[path] = arr
    return CACHE[path]


def color_for_label(label: int) -> str:
    return BRIGHT_COLORS[(label - 1) % len(BRIGHT_COLORS)]


def make_projection_rgb(frame_mask: np.ndarray, axis: int) -> np.ndarray:
    labels = np.unique(frame_mask)
    labels = labels[labels > 0]
    projection = np.max(frame_mask, axis=axis)
    rgb = np.zeros(projection.shape + (3,), dtype=np.uint8)
    for label in labels:
        color = color_for_label(int(label)).lstrip("#")
        rgb[projection == label] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


def build_3d_figure(frame_mask: np.ndarray, point_size: float, selected_ids: list[int]) -> go.Figure:
    fig = go.Figure()
    labels = np.unique(frame_mask)
    labels = labels[labels > 0]

    if selected_ids:
        allowed = set(selected_ids)
        labels = np.array([label for label in labels if int(label) in allowed], dtype=labels.dtype)

    for label in labels:
        z, y, x = np.where(frame_mask == label)
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name=f"Filament {int(label)}",
                marker=dict(
                    size=point_size,
                    color=color_for_label(int(label)),
                    opacity=0.9,
                ),
                hovertemplate="id=%{text}<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
                text=[int(label)] * len(x),
            )
        )

    z_dim, y_dim, x_dim = frame_mask.shape
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(range=[0, x_dim]),
            yaxis=dict(range=[0, y_dim]),
            zaxis=dict(range=[0, z_dim]),
            aspectmode="data",
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Filament Mask",
        legend=dict(itemsizing="constant"),
    )
    return fig


def summarize_mask(mask: np.ndarray, frame_idx: int) -> str:
    frame_mask = mask[frame_idx]
    labels = np.unique(frame_mask)
    labels = labels[labels > 0]
    active_voxels = int(np.count_nonzero(frame_mask))
    return (
        f"Loaded `{mask.shape[0]}` frames, `{mask.shape[1]}` z-planes, "
        f"shape `{mask.shape[2]}x{mask.shape[3]}`. "
        f"Frame `{frame_idx}` has `{len(labels)}` active filament IDs and `{active_voxels}` labeled voxels."
    )


def load_file_state(mask_choice: str | None, uploaded_file):
    path = resolve_mask_path(mask_choice, uploaded_file)
    if not path:
        return (
            gr.update(maximum=0, value=0),
            gr.update(choices=[], value=[]),
            None,
            None,
            None,
            go.Figure(),
            "Pick a mask TIFF to begin.",
            None,
        )

    try:
        mask = load_mask(path)
    except Exception as exc:  # pragma: no cover - UI path
        return (
            gr.update(maximum=0, value=0),
            gr.update(choices=[], value=[]),
            None,
            None,
            None,
            go.Figure(),
            f"Failed to load mask TIFF: {exc}",
            None,
        )

    frame0 = mask[0]
    ids = [int(v) for v in np.unique(frame0) if v > 0]
    xy = make_projection_rgb(frame0, axis=0)
    xz = make_projection_rgb(frame0, axis=1)
    yz = make_projection_rgb(frame0, axis=2)
    fig = build_3d_figure(frame0, point_size=3.0, selected_ids=ids)
    status = summarize_mask(mask, frame_idx=0)
    return (
        gr.update(maximum=max(0, mask.shape[0] - 1), value=0),
        gr.update(choices=ids, value=ids),
        xy,
        xz,
        yz,
        fig,
        status,
        path,
    )


def update_frame(
    stored_path: str | None,
    frame_idx: int,
    selected_ids: list[int] | None,
    point_size: float,
):
    if not stored_path:
        return None, None, None, go.Figure(), "Pick a mask TIFF to begin."

    mask = load_mask(stored_path)
    frame_idx = int(np.clip(frame_idx, 0, mask.shape[0] - 1))
    selected_ids = [int(v) for v in (selected_ids or [])]

    frame_mask = mask[frame_idx]
    xy = make_projection_rgb(frame_mask, axis=0)
    xz = make_projection_rgb(frame_mask, axis=1)
    yz = make_projection_rgb(frame_mask, axis=2)
    fig = build_3d_figure(frame_mask, point_size=float(point_size), selected_ids=selected_ids)
    status = summarize_mask(mask, frame_idx=frame_idx)
    return xy, xz, yz, fig, status


def build_app(mask_dir: str) -> gr.Blocks:
    default_masks = list_mask_files(mask_dir)
    default_choice = default_masks[0] if default_masks else None

    with gr.Blocks(title="Filament Mask Time Viewer") as demo:
        gr.Markdown("# Filament Mask Time Viewer")
        gr.Markdown(
            "Load a labeled mask TIFF and scrub through time. "
            "This uses the saved mask stack directly; it does not rerun inference."
        )

        stored_path = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                mask_choice = gr.Dropdown(
                    label="Mask TIFF from results/masks",
                    choices=default_masks,
                    value=default_choice,
                    type="value",
                )
                upload_mask = gr.File(
                    label="Or upload a mask TIFF",
                    file_types=[".tif", ".tiff"],
                    type="filepath",
                )
                frame_slider = gr.Slider(
                    label="Frame",
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                )
                filament_ids = gr.CheckboxGroup(
                    label="Visible filament IDs",
                    choices=[],
                    value=[],
                )
                point_size = gr.Slider(
                    label="3D point size",
                    minimum=1,
                    maximum=8,
                    step=0.5,
                    value=3,
                )
                refresh = gr.Button("Refresh mask list")
                status = gr.Markdown("Pick a mask TIFF to begin.")

            with gr.Column(scale=2):
                plot_3d = gr.Plot(label="3D mask view")
                with gr.Row():
                    xy = gr.Image(label="XY projection", type="numpy")
                    xz = gr.Image(label="XZ projection", type="numpy")
                    yz = gr.Image(label="YZ projection", type="numpy")

        refresh.click(
            lambda: gr.update(choices=list_mask_files(mask_dir)),
            outputs=mask_choice,
        )

        loader_inputs = [mask_choice, upload_mask]
        loader_outputs = [frame_slider, filament_ids, xy, xz, yz, plot_3d, status, stored_path]
        mask_choice.change(load_file_state, inputs=loader_inputs, outputs=loader_outputs)
        upload_mask.change(load_file_state, inputs=loader_inputs, outputs=loader_outputs)

        viewer_inputs = [stored_path, frame_slider, filament_ids, point_size]
        viewer_outputs = [xy, xz, yz, plot_3d, status]
        frame_slider.change(update_frame, inputs=viewer_inputs, outputs=viewer_outputs)
        filament_ids.change(update_frame, inputs=viewer_inputs, outputs=viewer_outputs)
        point_size.change(update_frame, inputs=viewer_inputs, outputs=viewer_outputs)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive viewer for filament mask TIFFs.")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR, help="Directory containing labeled mask TIFFs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the UI server to.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the UI server to. If omitted, Gradio picks a free port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(args.mask_dir)
    launch_kwargs = {"server_name": args.host, "share": False}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
