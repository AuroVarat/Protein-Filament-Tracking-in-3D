import glob
import os
import socket
import subprocess
import sys
import time
from colorsys import hsv_to_rgb
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tifffile
from streamlit_image_coordinates import streamlit_image_coordinates


ROOT_DIR = Path(__file__).resolve().parent
PAINTER_PORT = int(os.environ.get("FILAMENT_PAINTER_PORT", "7860"))


st.set_page_config(layout="wide", page_title="Hyperstack TIF Viewer")


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f4f8f5;
            --panel-bg: rgba(255, 255, 255, 0.92);
            --panel-border: #c7d8cc;
            --panel-shadow: 0 18px 42px rgba(37, 65, 49, 0.08);
            --text-main: #1f3528;
            --text-muted: #5a7160;
            --accent: #2f7d5b;
            --accent-soft: #e4f0e8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(85, 144, 111, 0.12), transparent 30%),
                linear-gradient(180deg, #f8fbf9 0%, var(--app-bg) 100%);
            color: var(--text-main);
        }

        #MainMenu,
        header,
        footer,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, p, label, div {
            color: var(--text-main);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eef5f0 0%, #e6efe8 100%);
            border-right: 1px solid var(--panel-border);
        }

        .app-hero,
        .app-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: var(--panel-shadow);
        }

        .app-hero {
            padding: 1.5rem 1.6rem;
            margin-bottom: 1.25rem;
        }

        .app-card {
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .eyebrow {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.45rem;
        }

        .hero-title {
            font-size: 2.1rem;
            line-height: 1.1;
            font-weight: 700;
            margin: 0;
        }

        .hero-copy {
            color: var(--text-muted);
            margin-top: 0.55rem;
            margin-bottom: 0;
            max-width: 56rem;
        }

        .painter-shell {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: var(--panel-shadow);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(228, 240, 232, 0.9);
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.6rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: white !important;
            border-color: var(--accent) !important;
        }

        div[data-testid="stRadio"] > label {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.65rem;
            flex-direction: row;
            flex-wrap: wrap;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: rgba(228, 240, 232, 0.9);
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.6rem 1rem;
            min-height: auto;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--accent);
            border-color: var(--accent);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: var(--text-main);
            margin: 0;
            font-weight: 600;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
            color: white !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] input {
            display: none;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid var(--accent);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2f7d5b 0%, #245c43 100%);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def get_local_tif_files():
    files = glob.glob(str(ROOT_DIR / "**" / "*.tif"), recursive=True)
    files.extend(glob.glob(str(ROOT_DIR / "**" / "*.tiff"), recursive=True))
    files = [f for f in files if "/." not in f and "\\." not in f]
    return sorted(os.path.relpath(f, ROOT_DIR) for f in files)


@st.cache_data(ttl=60)
def get_local_csv_files():
    files = glob.glob(str(ROOT_DIR / "**" / "*.csv"), recursive=True)
    files = [f for f in files if "/." not in f and "\\." not in f]
    return sorted(os.path.relpath(f, ROOT_DIR) for f in files)


@st.cache_data(ttl=60)
def get_local_mask_files():
    files = []
    for folder_name in ["masks3d", os.path.join("models", "masks3d")]:
        folder = ROOT_DIR / folder_name
        if not folder.exists():
            continue
        files.extend(glob.glob(str(folder / "**" / "*.npy"), recursive=True))
        files.extend(glob.glob(str(folder / "**" / "*.tif"), recursive=True))
        files.extend(glob.glob(str(folder / "**" / "*.tiff"), recursive=True))
    files = [f for f in files if "/." not in f and "\\." not in f]
    return sorted(os.path.relpath(f, ROOT_DIR) for f in files)


@st.cache_data(ttl=60)
def get_mask_tiff_files():
    files = []
    for tif_rel in get_local_tif_files():
        tif_lower = tif_rel.lower()
        if "mask" in tif_lower or "masks" in tif_lower:
            files.append(tif_rel)
    return sorted(files)


@st.cache_resource
def load_image(file_source):
    if isinstance(file_source, str):
        data = tifffile.memmap(file_source)
        with tifffile.TiffFile(file_source) as tif:
            axes = tif.series[0].axes
            is_rgb = getattr(tif, "is_rgb", getattr(tif.series[0], "is_rgb", False))
        return data, axes, is_rgb

    with tifffile.TiffFile(file_source) as tif:
        data = tif.asarray()
        axes = tif.series[0].axes
        is_rgb = getattr(tif, "is_rgb", getattr(tif.series[0], "is_rgb", False))
    return data, axes, is_rgb


def normalize_for_display(img):
    if img.dtype == np.uint8:
        return img

    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    return (img_norm * 255).astype(np.uint8)


@st.cache_data(ttl=60)
def load_tracking_csv(csv_path: str):
    data = pd.read_csv(csv_path)
    required_columns = {"frame", "filament_id", "z", "y", "x"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Tracking CSV is missing columns: {', '.join(sorted(missing_columns))}")

    for column in ["frame", "filament_id", "z", "y", "x"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["frame", "filament_id", "z", "y", "x"]).copy()
    data["frame"] = data["frame"].astype(int)
    data["filament_id"] = data["filament_id"].astype(int)
    return data


@st.cache_data(ttl=60)
def load_generic_csv(csv_path: str):
    return pd.read_csv(csv_path)


@st.cache_data(ttl=60)
def load_mask_volume(mask_path: str):
    return np.load(mask_path)


def infer_matching_tif(csv_relative_path: str) -> str | None:
    csv_name = Path(csv_relative_path).name
    tif_name = csv_name.replace("_tracking.csv", ".tif")
    candidate = ROOT_DIR / "manual_crops" / tif_name
    if candidate.exists():
        return str(candidate.resolve())
    return None


def find_saved_mask_path(tif_path: str, frame_idx: int) -> str | None:
    base_name = Path(tif_path).stem
    mask_path = ROOT_DIR / "masks3d" / f"{base_name}_t{frame_idx:04d}.npy"
    if mask_path.exists():
        return str(mask_path.resolve())
    return None


def filament_color(filament_id: int) -> tuple[int, int, int]:
    hue = ((filament_id * 0.173) % 1.0)
    rgb = hsv_to_rgb(hue, 0.75, 1.0)
    return tuple(int(channel * 255) for channel in rgb)


def infer_matching_csv(tif_path: str) -> str | None:
    tif_name = Path(tif_path).stem
    candidate_names = [f"{tif_name}_tracking.csv"]
    for candidate_name in candidate_names:
        direct_candidate = ROOT_DIR / candidate_name
        if direct_candidate.exists():
            return str(direct_candidate.resolve())

        matches = list(ROOT_DIR.glob(f"**/{candidate_name}"))
        if matches:
            return str(matches[0].resolve())
    return None


def get_tracking_rows_for_frame(tracking_rows, frame_idx: int):
    if tracking_rows is None:
        return None
    return tracking_rows[tracking_rows["frame"] == frame_idx].copy()


def blend_mask_on_rgb(rgb_image: np.ndarray, mask_2d: np.ndarray | None, alpha: float, color=(0, 255, 120)) -> np.ndarray:
    blended = rgb_image.copy()
    if mask_2d is None:
        return blended
    mask_binary = mask_2d > 0
    if mask_binary.any():
        blended[mask_binary] = ((1.0 - alpha) * blended[mask_binary] + alpha * np.array(color)).astype(np.uint8)
    return blended


def draw_tracking_points_xy(image_rgb: np.ndarray, tracking_rows, point_radius: int, show_labels: bool) -> np.ndarray:
    overlay = image_rgb.copy()
    if tracking_rows is None or tracking_rows.empty:
        return overlay

    for row in tracking_rows.itertuples(index=False):
        color = filament_color(int(row.filament_id))
        center = (int(round(row.x)), int(round(row.y)))
        cv2.circle(overlay, center, point_radius, color, thickness=-1)
        cv2.circle(overlay, center, max(point_radius + 1, 2), (255, 255, 255), thickness=1)
        if show_labels:
            cv2.putText(
                overlay,
                str(int(row.filament_id)),
                (center[0] + point_radius + 3, center[1] - point_radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return overlay


def build_xy_overlay(
    image_2d: np.ndarray,
    tracking_rows,
    z_idx: int | None,
    point_radius: int,
    show_labels: bool,
    saved_mask: np.ndarray | None = None,
    saved_mask_alpha: float = 0.35,
):
    base = cv2.cvtColor(normalize_for_display(image_2d), cv2.COLOR_GRAY2RGB)
    mask_slice = None
    if saved_mask is not None and z_idx is not None and saved_mask.ndim == 3 and 0 <= z_idx < saved_mask.shape[0]:
        mask_slice = saved_mask[z_idx]
    overlay = blend_mask_on_rgb(base, mask_slice, saved_mask_alpha)

    if tracking_rows is not None and z_idx is not None:
        tracking_rows = tracking_rows[tracking_rows["z"].round().astype(int) == z_idx]
    return draw_tracking_points_xy(overlay, tracking_rows, point_radius, show_labels), tracking_rows


def open_crop_in_results_viewer(crop_path: str, csv_path: str | None = None, roi_index: int | None = None) -> None:
    st.session_state["active_page"] = "Filament Analysis"
    st.session_state["results_selected_tif"] = crop_path
    st.session_state["results_selected_csv"] = csv_path
    if roi_index is not None:
        st.session_state["results_selected_roi_index"] = roi_index


def infer_matching_mask_files(tif_path: str):
    base_name = Path(tif_path).stem
    masks = []
    for folder_name in ["masks3d", os.path.join("models", "masks3d")]:
        folder = ROOT_DIR / folder_name
        if not folder.exists():
            continue
        masks.extend(folder.glob(f"{base_name}*.npy"))
        masks.extend(folder.glob(f"{base_name}*.tif"))
        masks.extend(folder.glob(f"{base_name}*.tiff"))
    return sorted(str(path.resolve()) for path in masks)


def infer_matching_mask_tiffs(tif_path: str):
    base_name = Path(tif_path).stem.lower()
    matches = []
    for tif_rel in get_mask_tiff_files():
        tif_name = Path(tif_rel).stem.lower()
        if base_name in tif_name or tif_name in base_name:
            matches.append(str((ROOT_DIR / tif_rel).resolve()))
    return sorted(matches)


def infer_matching_mask_series(tif_path: str):
    base_name = Path(tif_path).stem
    mask_files = infer_matching_mask_files(tif_path)
    npy_files = [path for path in mask_files if path.lower().endswith(".npy")]
    return sorted(npy_files)


def infer_mask_series_base(tif_path: str) -> str | None:
    base_name = Path(tif_path).stem
    for folder_name in ["masks3d", os.path.join("models", "masks3d")]:
        folder = ROOT_DIR / folder_name
        if not folder.exists():
            continue
        pattern = sorted(folder.glob(f"{base_name}_t*.npy"))
        if pattern:
            return str((folder / base_name).resolve())
    return None


def render_filament_stats(dataframe):
    if dataframe is None:
        st.info("No CSV selected.")
        return

    st.write("### Filament Statistics")
    filament_count = int(dataframe["filament_id"].nunique()) if "filament_id" in dataframe.columns else None
    total_detections = len(dataframe)
    average_length_um = None
    if "size_px" in dataframe.columns:
        average_length_um = float(dataframe["size_px"].mean()) * 0.3

    average_duration_min = None
    if {"filament_id", "frame"}.issubset(dataframe.columns):
        duration_df = (
            dataframe.groupby("filament_id")["frame"]
            .agg(["min", "max"])
            .reset_index()
        )
        duration_df["duration_min"] = (duration_df["max"] - duration_df["min"]) * 15.0
        average_duration_min = float(duration_df["duration_min"].mean()) if not duration_df.empty else 0.0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Detections", total_detections)
    with metric_cols[1]:
        st.metric("Unique Filaments", filament_count if filament_count is not None else "N/A")
    with metric_cols[2]:
        st.metric(
            "Average Length (um)",
            f"{average_length_um:.2f}" if average_length_um is not None else "N/A",
        )
    with metric_cols[3]:
        st.metric(
            "Average Duration (min)",
            f"{average_duration_min:.1f}" if average_duration_min is not None else "N/A",
        )

    numeric_df = dataframe.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("### Numeric Summary")
        st.dataframe(numeric_df.describe().transpose(), width="stretch")

    st.write("### CSV Preview")
    st.dataframe(dataframe.head(100), width="stretch", hide_index=True)


def launch_napari(
    image_path: str,
    mask_tiff_path: str | None,
    mask_series_base: str | None,
    channel_idx: int,
) -> tuple[bool, str]:
    script_path = ROOT_DIR / "scripts" / "open_in_napari.py"
    if not script_path.exists():
        return False, f"Napari launcher script not found: {script_path}"

    launch_cmd = [sys.executable, str(script_path), "--image", image_path, "--channel", str(channel_idx)]
    if mask_tiff_path:
        launch_cmd.extend(["--mask-tiff", mask_tiff_path])
    if mask_series_base:
        launch_cmd.extend(["--mask-series-base", mask_series_base])

    try:
        subprocess.Popen(launch_cmd, cwd=str(ROOT_DIR))
        return True, "Opened napari in a separate process."
    except Exception as exc:
        return False, f"Failed to start napari: {exc}"


def render_projection_plot(title: str, image_rgb: np.ndarray, x_coords, y_coords, tracking_rows, point_radius: int):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        st.error(f"Plotly is required for orthogonal views: {exc}")
        return

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_rgb))

    if tracking_rows is not None and not tracking_rows.empty:
        hover_text = [
            f"Filament {int(row.filament_id)}<br>Frame {int(row.frame)}<br>Z {float(row.z):.2f}<br>Y {float(row.y):.2f}<br>X {float(row.x):.2f}"
            for row in tracking_rows.itertuples(index=False)
        ]
        point_colors = [f"rgb{filament_color(int(fid))}" for fid in tracking_rows["filament_id"]]
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker={
                    "size": max(point_radius * 2, 8),
                    "color": point_colors,
                    "line": {"color": "white", "width": 1},
                },
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Tracked filaments",
            )
        )

    fig.update_layout(
        title=title,
        height=360,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False, "autorange": "reversed", "scaleanchor": "x"},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_orthogonal_views(volume_zyx, tracking_rows, point_radius, show_labels, saved_mask, saved_mask_alpha):
    xy_proj = np.max(volume_zyx, axis=0)
    xz_proj = np.max(volume_zyx, axis=1)
    yz_proj = np.max(volume_zyx, axis=2)

    xy_rgb = cv2.cvtColor(normalize_for_display(xy_proj), cv2.COLOR_GRAY2RGB)
    xz_rgb = cv2.cvtColor(normalize_for_display(xz_proj), cv2.COLOR_GRAY2RGB)
    yz_rgb = cv2.cvtColor(normalize_for_display(yz_proj), cv2.COLOR_GRAY2RGB)

    if saved_mask is not None and saved_mask.ndim == 3:
        xy_rgb = blend_mask_on_rgb(xy_rgb, np.max(saved_mask, axis=0), saved_mask_alpha)
        xz_rgb = blend_mask_on_rgb(xz_rgb, np.max(saved_mask, axis=1), saved_mask_alpha)
        yz_rgb = blend_mask_on_rgb(yz_rgb, np.max(saved_mask, axis=2), saved_mask_alpha)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_projection_plot(
            "XY projection",
            xy_rgb,
            tracking_rows["x"] if tracking_rows is not None else [],
            tracking_rows["y"] if tracking_rows is not None else [],
            tracking_rows,
            point_radius,
        )
    with col2:
        render_projection_plot(
            "XZ projection",
            xz_rgb,
            tracking_rows["x"] if tracking_rows is not None else [],
            tracking_rows["z"] if tracking_rows is not None else [],
            tracking_rows,
            point_radius,
        )
    with col3:
        render_projection_plot(
            "YZ projection",
            yz_rgb,
            tracking_rows["y"] if tracking_rows is not None else [],
            tracking_rows["z"] if tracking_rows is not None else [],
            tracking_rows,
            point_radius,
        )


def render_two_point_five_d_view(volume_zyx, z_idx, tracking_rows, point_radius, show_labels, saved_mask, saved_mask_alpha):
    z_prev = max(0, z_idx - 1)
    z_next = min(volume_zyx.shape[0] - 1, z_idx + 1)
    composite = np.stack(
        [
            normalize_for_display(volume_zyx[z_prev]),
            normalize_for_display(volume_zyx[z_idx]),
            normalize_for_display(volume_zyx[z_next]),
        ],
        axis=-1,
    )
    if saved_mask is not None and saved_mask.ndim == 3:
        composite = blend_mask_on_rgb(composite, saved_mask[z_idx], saved_mask_alpha)

    active_rows = None
    if tracking_rows is not None:
        active_rows = tracking_rows[tracking_rows["z"].round().astype(int) == z_idx]
    composite = draw_tracking_points_xy(composite, active_rows, point_radius, show_labels)

    st.write("**2.5D RGB slab**")
    st.image(composite, width="stretch")
    st.caption(f"Red = Z {z_prev}, Green = Z {z_idx}, Blue = Z {z_next}")


def render_3d_volume_view(volume_zyx, tracking_rows, saved_mask):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        st.error(f"Plotly is required for 3D volume rendering: {exc}")
        return

    volume = volume_zyx.astype(np.float32)
    if volume.max() > volume.min():
        volume = (volume - volume.min()) / (volume.max() - volume.min())
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    threshold = float(np.quantile(volume, 0.985))
    signal_points = np.argwhere(volume >= threshold)
    if len(signal_points) == 0:
        threshold = float(np.quantile(volume, 0.95))
        signal_points = np.argwhere(volume >= threshold)
    signal_step = max(1, len(signal_points) // 12000)
    signal_points = signal_points[::signal_step]
    signal_values = volume[signal_points[:, 0], signal_points[:, 1], signal_points[:, 2]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=signal_points[:, 2],
            y=signal_points[:, 1],
            z=signal_points[:, 0],
            mode="markers",
            marker={
                "size": 2.5,
                "color": signal_values,
                "colorscale": "Gray",
                "opacity": 0.16,
                "showscale": False,
            },
            hoverinfo="skip",
            name="Fluorescence signal",
        )
    )

    if tracking_rows is not None and not tracking_rows.empty:
        point_colors = [f"rgb{filament_color(int(fid))}" for fid in tracking_rows["filament_id"]]
        fig.add_trace(
            go.Scatter3d(
                x=tracking_rows["x"],
                y=tracking_rows["y"],
                z=tracking_rows["z"],
                mode="markers",
                marker={"size": 5, "color": point_colors, "line": {"color": "white", "width": 1}},
                text=[
                    f"Filament {int(row.filament_id)}<br>Frame {int(row.frame)}<br>Z {float(row.z):.2f}<br>Y {float(row.y):.2f}<br>X {float(row.x):.2f}"
                    for row in tracking_rows.itertuples(index=False)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Tracked filaments",
            )
        )

    if saved_mask is not None and saved_mask.ndim == 3 and np.any(saved_mask > 0):
        mask_points = np.argwhere(saved_mask > 0)
        mask_step = max(1, len(mask_points) // 4000)
        mask_points = mask_points[::mask_step]
        fig.add_trace(
            go.Scatter3d(
                x=mask_points[:, 2],
                y=mask_points[:, 1],
                z=mask_points[:, 0],
                mode="markers",
                marker={"size": 1.8, "color": "rgba(0,255,120,0.18)"},
                hoverinfo="skip",
                name="Painter mask",
            )
        )

    fig.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex((host, port)) == 0


def ensure_painter_running(initial_target: str | None = None) -> tuple[bool, str | None]:
    if is_port_open("127.0.0.1", PAINTER_PORT):
        return True, None

    script_path = ROOT_DIR / "scripts" / "filament_5z_painter_web.py"
    if not script_path.exists():
        return False, f"Painter script not found: {script_path}"

    launch_cmd = [sys.executable, str(script_path)]
    if initial_target:
        launch_cmd.append(initial_target)

    env = os.environ.copy()
    env["FILAMENT_PAINTER_PORT"] = str(PAINTER_PORT)

    popen_kwargs = {
        "cwd": str(ROOT_DIR),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    process = subprocess.Popen(launch_cmd, **popen_kwargs)
    st.session_state["filament_painter_pid"] = process.pid

    for _ in range(24):
        time.sleep(0.5)
        if is_port_open("127.0.0.1", PAINTER_PORT):
            return True, None

    return False, f"Painter started with PID {process.pid}, but port {PAINTER_PORT} did not open in time."


def resolve_source():
    local_files = get_local_tif_files()
    input_method = st.sidebar.radio(
        "File Input Method",
        ["Select Found File (Fast)", "Enter Custom Path (Fast)", "Upload File (Slower)"],
    )

    if input_method == "Upload File (Slower)":
        source = st.sidebar.file_uploader("Upload a Hyperstack TIF", type=["tif", "tiff"])
        if source is None:
            st.info("Please upload a TIF file from the sidebar to begin.")
            return None
        return source

    if input_method == "Select Found File (Fast)":
        if not local_files:
            st.info("No .tif files found automatically in this directory. Try uploading or entering a custom path.")
            return None
        selected = st.sidebar.selectbox("Choose a TIF file:", local_files)
        return str((ROOT_DIR / selected).resolve())

    source = st.sidebar.text_input("Absolute path to TIF file:", value="")
    if not source:
        st.info("Please enter the exact file path to your TIF file.")
        return None
    if not os.path.exists(source):
        st.error(f"File not found: {source}")
        return None
    return source


def render_preview_and_analysis_tab():
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Main Workflow</div>
            <h1 class="hero-title">Hyperstack TIF Previewer</h1>
            <p class="hero-copy">
                Load a stack, review slices, extract ROIs, and run the existing ridge-enhancement tracking flow
                without leaving the main application.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source = resolve_source()
    if source is None:
        return None

    with st.spinner("Loading hyperstack..."):
        try:
            data, axes, is_rgb = load_image(source)
        except Exception as exc:
            st.error(f"Error loading TIF: {exc}")
            return None

    st.sidebar.success("File successfully loaded")
    st.sidebar.write(f"**Shape:** {data.shape}")
    st.sidebar.write(f"**Detected Axes:** {axes if axes else 'Unknown'}")

    process_path = source if isinstance(source, str) else "uploaded_temp.tif"

    if "found_centers" not in st.session_state:
        st.session_state.found_centers = None
    if "processed_path" not in st.session_state:
        st.session_state.processed_path = None
    if "extraction_complete" not in st.session_state:
        st.session_state.extraction_complete = False

    if st.session_state.processed_path != process_path:
        st.session_state.found_centers = None
        st.session_state.extraction_complete = False
        st.session_state.processed_path = process_path

    if st.session_state.found_centers is None:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Interactive Preview")

        shape = data.shape
        ndim = len(shape)
        is_rgb = is_rgb or (axes and axes[-1] in ["S", "C"] and shape[-1] in [3, 4])
        display_dims = 3 if is_rgb else 2

        if ndim < 2:
            st.error("Image has less than 2 dimensions. This is not a valid 2D/3D image.")
            st.markdown("</div>", unsafe_allow_html=True)
            return source

        st.sidebar.markdown("---")
        st.sidebar.subheader("Controls")

        indices = []
        for i in range(ndim - display_dims):
            axis_label = axes[i] if axes and i < len(axes) else f"Dimension {i}"
            max_val = shape[i] - 1
            if max_val > 0:
                indices.append(st.sidebar.slider(f"{axis_label} Index", 0, max_val, 0))
            else:
                indices.append(0)

        preview_img = data if ndim <= display_dims else data[tuple(indices)]
        st.image(normalize_for_display(preview_img), width="stretch", clamp=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        if st.button("Return to Raw Preview"):
            st.session_state.found_centers = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    render_roi_and_tracking(source, process_path, data, axes)
    return source


def render_roi_and_tracking(source, process_path, data, axes):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Automated ROI Extraction")

    script_dir = ROOT_DIR / "Fillipo" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))
    from cellpose_roi_extractor import CellposeRoiExtractor

    if st.session_state.found_centers is None:
        if st.button("1. Find Cell ROIs", type="primary"):
            with st.spinner("Processing image through Cellpose..."):
                if not isinstance(source, str) and not os.path.exists(process_path):
                    with open(process_path, "wb") as handle:
                        handle.write(source.getvalue())

                base_name = os.path.splitext(os.path.basename(process_path))[0]
                out_plot = f"{base_name}_dbscan_plot.png"

                extractor = CellposeRoiExtractor(eps=60)
                centers = extractor.visualize_clusters(process_path, output_plot=out_plot, data=data)
                st.session_state.found_centers = centers
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not st.session_state.get("extraction_complete", False):
        render_roi_editor(source, process_path, data, axes, CellposeRoiExtractor)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    render_tracking_view(process_path, data)
    st.markdown("</div>", unsafe_allow_html=True)


def render_roi_editor(source, process_path, data, axes, extractor_cls):
    base_name = os.path.splitext(os.path.basename(process_path))[0]
    out_dir = f"{base_name}_extracted_crops"

    st.markdown("### Interactive ROI Editor")
    st.write("Click the red corner box to delete an ROI. Click outside existing boxes to add one.")

    if len(data.shape) >= 5:
        z_idx = data.shape[1] // 2
        raw_img_bf = data[0, z_idx, 0, :, :]
        other_channel = 1 if data.shape[2] > 1 else 0
        raw_img_other = data[0, z_idx, other_channel, :, :]
    else:
        raw_img_bf = data[0, 0] if data.ndim > 2 else data
        raw_img_other = raw_img_bf

    half = 64
    btn_size = 30
    padding = 10

    def prepare_interactive_img(raw_img):
        img_disp = normalize_for_display(raw_img)
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)

        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            cv2.rectangle(img_rgb, (x - half, y - half), (x + half, y + half), (0, 255, 0), 3)
            tr_x, tr_y = x + half - btn_size, y - half
            cv2.rectangle(img_rgb, (tr_x, tr_y), (tr_x + btn_size, tr_y + btn_size), (255, 0, 0), -1)
            cv2.line(img_rgb, (tr_x + 5, tr_y + 5), (tr_x + btn_size - 5, tr_y + btn_size - 5), (255, 255, 255), 2)
            cv2.line(img_rgb, (tr_x + btn_size - 5, tr_y + 5), (tr_x + 5, tr_y + btn_size - 5), (255, 255, 255), 2)
            cv2.putText(
                img_rgb,
                str(i + 1),
                (x - half + 5, y - half + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        target_width = 600
        h, w = img_rgb.shape[:2]
        if w > target_width:
            scale = target_width / w
            img_resized = cv2.resize(img_rgb, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            img_resized = img_rgb

        return img_resized, scale

    img_bf, scale = prepare_interactive_img(raw_img_bf)
    img_other, _ = prepare_interactive_img(raw_img_other)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Brightfield (Channel 0)**")
        val1 = streamlit_image_coordinates(img_bf, key=f"roi_click_bf_{len(st.session_state.found_centers)}")
    with col2:
        st.write("**Secondary Channel**")
        val2 = streamlit_image_coordinates(img_other, key=f"roi_click_other_{len(st.session_state.found_centers)}")

    clicked_value = val1 if val1 is not None else val2

    if clicked_value is not None:
        cx = int(clicked_value["x"] / scale)
        cy = int(clicked_value["y"] / scale)
        action_taken = False

        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            tr_x, tr_y = x + half - btn_size, y - half
            if (tr_x - padding) <= cx <= (tr_x + btn_size + padding) and (tr_y - padding) <= cy <= (tr_y + btn_size + padding):
                st.session_state.found_centers.pop(i)
                st.rerun()
                action_taken = True
                break

        if not action_taken:
            inside_any = False
            for y, x in st.session_state.found_centers:
                if x - half <= cx <= x + half and y - half <= cy <= y + half:
                    inside_any = True
                    break
            if not inside_any:
                st.session_state.found_centers.append((cy, cx))
                st.rerun()

    if st.button(f"2. Extract & Save {len(st.session_state.found_centers)} Selected Crops", type="primary"):
        with st.spinner("Saving TIF crops..."):
            extractor = extractor_cls(eps=60)
            extractor.extract_and_save(process_path, out_dir, data=data, axes=axes, centers=st.session_state.found_centers)
            st.success(f"Successfully saved {len(st.session_state.found_centers)} crops to `{os.path.abspath(out_dir)}`.")

            if not isinstance(source, str) and os.path.exists("uploaded_temp.tif"):
                os.remove("uploaded_temp.tif")

            st.session_state.extraction_complete = True
            st.rerun()


def render_tracking_view(process_path, data):
    st.markdown("### Filament Tracking Analysis")
    if st.button("Go Back to Interactive ROI Editor"):
        st.session_state.extraction_complete = False
        st.rerun()

    base_name = os.path.splitext(os.path.basename(process_path))[0]
    out_dir = f"{base_name}_extracted_crops"

    if len(data.shape) >= 5:
        z_idx = data.shape[1] // 2
        other_channel = 1 if data.shape[2] > 1 else 0
        raw_img_other = data[0, z_idx, other_channel, :, :]
    else:
        raw_img_other = data[0, 0] if data.ndim > 2 else data

    img_disp = normalize_for_display(raw_img_other)
    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)

    if "selected_roi_index" not in st.session_state:
        st.session_state.selected_roi_index = 0

    half = 64
    crop_files = sorted(glob.glob(os.path.join(out_dir, "*.tif")))

    for i, (y, x) in enumerate(st.session_state.found_centers):
        y, x = int(y), int(x)
        color = (255, 255, 0) if i == st.session_state.selected_roi_index else (0, 255, 0)
        thickness = 4 if i == st.session_state.selected_roi_index else 2
        cv2.rectangle(img_rgb, (x - half, y - half), (x + half, y + half), color, thickness)
        cv2.putText(img_rgb, str(i + 1), (x - half + 5, y - half + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.write("Click a green box to choose the ROI for tracking analysis.")

    target_width = 800
    h, w = img_rgb.shape[:2]
    if w > target_width:
        scale = target_width / w
        img_rgb = cv2.resize(img_rgb, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    val_p3 = streamlit_image_coordinates(img_rgb, key=f"roi_select_p3_{st.session_state.selected_roi_index}")
    if val_p3 is not None:
        cx = int(val_p3["x"] / scale)
        cy = int(val_p3["y"] / scale)
        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            if x - half <= cx <= x + half and y - half <= cy <= y + half:
                if st.session_state.selected_roi_index != i:
                    st.session_state.selected_roi_index = i
                    st.rerun()
                break

    ridge_dir = ROOT_DIR / "Ridge_Enhancement"
    if str(ridge_dir) not in sys.path:
        sys.path.append(str(ridge_dir))

    try:
        import Image as RImage
    except Exception as exc:
        st.error(f"Could not load Ridge Enhancement module: {exc}")
        return

    if not crop_files:
        st.warning("No extracted crops found in the output directory.")
        return
    if st.session_state.selected_roi_index >= len(crop_files):
        st.warning("Selected ROI is out of bounds. Please re-extract.")
        return

    selected_crop_path = crop_files[st.session_state.selected_roi_index]
    crop_stack = tifffile.imread(selected_crop_path)
    if len(crop_stack.shape) == 5:
        other_channel = 1 if crop_stack.shape[2] > 1 else 0
        stack_2d_timeseries = np.max(crop_stack[:, :, other_channel, :, :], axis=1)
    else:
        stack_2d_timeseries = crop_stack

    frames_count = stack_2d_timeseries.shape[0]
    st.info(f"Currently Selected for Analysis: ROI {st.session_state.selected_roi_index + 1}")
    results_csv_path = infer_matching_csv(selected_crop_path)
    if st.button("Open Selected ROI in Filament Analysis", type="primary"):
        open_crop_in_results_viewer(
            crop_path=str(Path(selected_crop_path).resolve()),
            csv_path=results_csv_path,
            roi_index=st.session_state.selected_roi_index,
        )
        st.rerun()
    if results_csv_path:
        st.caption(f"Matched tracking CSV: {os.path.relpath(results_csv_path, ROOT_DIR)}")
    else:
        st.caption("No matching tracking CSV found for this crop yet. The results page will still open on the TIFF.")

    with st.expander("Tracking Tuning Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            bright_filaments = st.checkbox("Bright Filaments", value=True)
            sigmas_input = st.text_input("Sigmas (comma-separated)", value="1, 1.5, 2, 2.5")
            low_q = st.slider("Low Quantile (low_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
            high_q = st.slider("High Quantile (high_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
        with col2:
            background_radius = st.slider("Background Radius", 1, 100, 15, 1)
            remove_objects_leq = st.number_input("Remove objects <= (px)", value=20, min_value=0)
            remove_holes_leq = st.number_input("Remove holes <= (px)", value=10, min_value=0)
            min_branch_length_um = st.number_input("Min Branch Length (um)", value=0.200, min_value=0.0, format="%.3f", step=0.001)

        pixel_size_um = 0.184
        mins_per_frame = 15.0

        try:
            sigmas = tuple(float(x.strip()) for x in sigmas_input.split(","))
        except ValueError:
            st.error("Invalid sigmas format")
            sigmas = (1.5,)

        min_branch_length_px = min_branch_length_um / pixel_size_um if pixel_size_um > 0 else 0.0
        params = RImage.Params(
            bright_filaments=bright_filaments,
            sigmas=sigmas,
            low_q=low_q,
            high_q=high_q,
            background_radius=background_radius,
            remove_objects_leq=remove_objects_leq,
            remove_holes_leq=remove_holes_leq,
            min_branch_length_px=min_branch_length_px,
            pixel_size_um=pixel_size_um,
        )

    st.markdown("### Interactive Parameter Preview")
    if "p3_frame" not in st.session_state:
        st.session_state.p3_frame = 1
    if "p3_playing" not in st.session_state:
        st.session_state.p3_playing = False

    def toggle_play_p3():
        st.session_state.p3_playing = not st.session_state.p3_playing

    col_play, col_slide = st.columns([1, 6])
    with col_play:
        if frames_count > 1:
            st.button("Pause" if st.session_state.p3_playing else "Play", on_click=toggle_play_p3, key="btn_play_p3")
        else:
            st.write("**Single Frame**")
    with col_slide:
        if frames_count > 1:
            st.session_state.p3_frame = st.slider("Frame Number", 1, frames_count, st.session_state.p3_frame)

    frame_idx = st.session_state.p3_frame - 1
    if frame_idx >= frames_count:
        frame_idx = 0
        st.session_state.p3_frame = 1

    preview_frame = stack_2d_timeseries[frame_idx]
    pre = RImage.preprocess(preview_frame, params)
    ridge = RImage.ridge_enhance(pre, params)
    mask = RImage.segment_ridges(ridge, params)
    skeleton = RImage.morphology.skeletonize(mask)
    rows = RImage.extract_segments(skeleton, frame_idx, params)

    st.write(f"Extracted segments in this frame: {len(rows)}")
    if rows:
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Segment ID": r["segment_id"],
                        "Length (px)": round(r["length_px"], 2),
                        "Length (um)": round(r["length_um"], 3),
                    }
                    for r in rows
                ]
            ),
            width="stretch",
        )

    prev1, prev2, prev3, prev4 = st.columns(4)
    with prev1:
        st.write("**Original**")
        st.image(preview_frame, clamp=True, width="stretch")
    with prev2:
        st.write("**Preprocessed**")
        st.image(pre, clamp=True, width="stretch")
    with prev3:
        st.write("**Ridge Enhance**")
        st.image(ridge, clamp=True, width="stretch")
    with prev4:
        st.write("**Overlay**")
        rgb = np.stack([pre, pre, pre], axis=-1)
        rgb[skeleton > 0] = [1.0, 0.0, 0.0]
        st.image(rgb, clamp=True, width="stretch")

    if st.button("Run Full Tracking Analysis on Crop", type="primary"):
        with st.spinner("Processing hyperstack crop through Ridge Enhancement pipeline..."):
            try:
                _, _, tracks = RImage.process_stack(stack_2d_timeseries, params)
                if tracks.empty:
                    st.warning("No filaments tracked across the stack. Try adjusting tuning parameters.")
                else:
                    import pandas as pd

                    active_frames = sorted(tracks["frame"].unique())
                    clusters = []
                    if active_frames:
                        start_f = active_frames[0]
                        prev_f = active_frames[0]
                        for frame in active_frames[1:]:
                            if frame == prev_f + 1:
                                prev_f = frame
                            else:
                                clusters.append((start_f + 1, prev_f + 1))
                                start_f = frame
                                prev_f = frame
                        clusters.append((start_f + 1, prev_f + 1))

                    cluster_df = pd.DataFrame(clusters, columns=["Start Frame", "End Frame"])
                    cluster_df["Duration (Frames)"] = cluster_df["End Frame"] - cluster_df["Start Frame"] + 1
                    cluster_df["Start Time (min)"] = (cluster_df["Start Frame"] - 1) * mins_per_frame
                    cluster_df["End Time (min)"] = (cluster_df["End Frame"] - 1) * mins_per_frame
                    cluster_df["Duration (min)"] = cluster_df["Duration (Frames)"] * mins_per_frame

                    stable_clusters = cluster_df[cluster_df["Duration (Frames)"] > 2]
                    transient_clusters = cluster_df[cluster_df["Duration (Frames)"] <= 2]

                    st.write("#### Stable Frame Clusters (> 2 frames)")
                    st.dataframe(stable_clusters, width="stretch")
                    st.write("#### Transient Frame Clusters (1-2 frames)")
                    st.dataframe(transient_clusters, width="stretch")
            except Exception as exc:
                st.error(f"Error during tracking: {exc}")

    if st.session_state.get("p3_playing", False) and frames_count > 1:
        time.sleep(0.3)
        st.session_state.p3_frame += 1
        if st.session_state.p3_frame > frames_count:
            st.session_state.p3_frame = 1
        st.rerun()


def render_painter_tab(source):
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Annotation</div>
            <h1 class="hero-title">Filament Painter</h1>
            <p class="hero-copy">
                Paint masks for the five Z-planes directly inside the main application. The painter runs in a local
                companion process, but it is presented here as a first-class tab with the same palette and shell.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    initial_target = source if isinstance(source, str) else None
    ready, error = ensure_painter_running(initial_target=initial_target)
    if not ready:
        st.error(error or "Failed to start the filament painter.")
        return

    st.markdown('<div class="painter-shell">', unsafe_allow_html=True)
    st.caption(f"Embedded painter served from http://127.0.0.1:{PAINTER_PORT}")
    components.iframe(f"http://127.0.0.1:{PAINTER_PORT}", height=1100, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_results_tab():
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Analysis</div>
            <h1 class="hero-title">Filament Analysis</h1>
            <p class="hero-copy">
                Review the filament CSV as an analysis table, inspect summary statistics, and open the selected crop
                with an optional mask in napari for proper local 3D exploration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    tif_files = get_local_tif_files()
    if not tif_files:
        st.info("No TIFF files were found in the project.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    csv_files = get_local_csv_files()
    requested_tif = st.session_state.get("results_selected_tif")
    default_tif_index = 0
    if requested_tif:
        requested_rel = os.path.relpath(requested_tif, ROOT_DIR)
        if requested_rel in tif_files:
            default_tif_index = tif_files.index(requested_rel)

    selected_tif_rel = st.selectbox("Crop TIFF", tif_files, index=default_tif_index)
    selected_tif_path = str((ROOT_DIR / selected_tif_rel).resolve())

    inferred_csv = infer_matching_csv(selected_tif_path)
    requested_csv = st.session_state.get("results_selected_csv")
    csv_options = ["None"] + csv_files
    default_csv_value = "None"
    for candidate in [requested_csv, inferred_csv]:
        if candidate:
            candidate_rel = os.path.relpath(candidate, ROOT_DIR)
            if candidate_rel in csv_files:
                default_csv_value = candidate_rel
                break
    selected_csv_rel = st.selectbox("Tracking CSV", csv_options, index=csv_options.index(default_csv_value))
    selected_csv_path = None if selected_csv_rel == "None" else str((ROOT_DIR / selected_csv_rel).resolve())

    try:
        crop_data, crop_axes, _ = load_image(selected_tif_path)
        stats_df = load_generic_csv(selected_csv_path) if selected_csv_path else None
    except Exception as exc:
        st.error(f"Could not load analysis inputs: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if crop_data.ndim != 5:
        st.error(f"Expected a 5D crop with TZCYX axes. Found shape {crop_data.shape} with axes `{crop_axes}`.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    t_dim, _, c_dim, _, _ = crop_data.shape
    default_channel = 1 if c_dim > 1 else 0

    matching_mask_tiffs = infer_matching_mask_tiffs(selected_tif_path)
    mask_tiff_files = get_mask_tiff_files()
    mask_tiff_options = ["None"] + mask_tiff_files
    default_mask_tiff_value = "None"
    if matching_mask_tiffs:
        first_match_rel = os.path.relpath(matching_mask_tiffs[0], ROOT_DIR)
        if first_match_rel in mask_tiff_files:
            default_mask_tiff_value = first_match_rel

    matching_mask_series = infer_matching_mask_series(selected_tif_path)
    mask_series_options = ["None"]
    default_mask_series_value = "None"
    if matching_mask_series:
        mask_series_options.append("Auto-detected NPY movie")
        default_mask_series_value = "Auto-detected NPY movie"

    controls1, controls2, controls3 = st.columns(3)
    with controls1:
        channel_idx = st.selectbox("Napari channel", list(range(c_dim)), index=default_channel)
    with controls2:
        selected_mask_tiff_rel = st.selectbox(
            "Mask TIFF",
            mask_tiff_options,
            index=mask_tiff_options.index(default_mask_tiff_value),
        )
    with controls3:
        selected_mask_series_label = st.selectbox(
            "Mask NPY series",
            mask_series_options,
            index=mask_series_options.index(default_mask_series_value),
        )

    selected_mask_tiff_path = (
        None if selected_mask_tiff_rel == "None" else str((ROOT_DIR / selected_mask_tiff_rel).resolve())
    )
    selected_mask_series_base = None
    if selected_mask_series_label != "None" and matching_mask_series:
        selected_mask_series_base = infer_mask_series_base(selected_tif_path)

    st.caption(
        f"Crop shape `{crop_data.shape}` with axes `{crop_axes}`. "
        f"Napari opens the full movie for the selected channel as a `TZYX` volume."
    )

    if matching_mask_tiffs:
        st.caption(f"Matching mask TIFFs found: {len(matching_mask_tiffs)}")
    if matching_mask_series:
        st.caption(f"Matching per-frame NPY masks found: {len(matching_mask_series)}")

    if st.button("View in napari", type="primary"):
        ok, message = launch_napari(
            selected_tif_path,
            selected_mask_tiff_path,
            selected_mask_series_base,
            channel_idx,
        )
        if ok:
            st.success(message)
        else:
            st.error(message)

    render_filament_stats(stats_df)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    apply_app_styles()
    pages = ["TIFF Viewer + Analysis", "Filament Painter", "Filament Analysis"]
    current_page = st.session_state.get("active_page", pages[0])
    if current_page not in pages:
        current_page = pages[0]
    st.session_state["active_page"] = st.radio("Navigation", pages, index=pages.index(current_page), horizontal=True)

    source = st.session_state.get("current_source")
    if st.session_state["active_page"] == "TIFF Viewer + Analysis":
        source = render_preview_and_analysis_tab()
        st.session_state["current_source"] = source
    elif st.session_state["active_page"] == "Filament Painter":
        render_painter_tab(source)
    else:
        render_results_tab()


if __name__ == "__main__":
    main()
