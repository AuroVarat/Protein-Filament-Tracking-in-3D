"""
Gradio-based Interactive Dashboard for 3D Filament Results.
Allows interactive exploration of inference results across time and Z-planes.
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import tifffile
import gradio as gr
import pandas as pd
import subprocess
from scipy import ndimage
import plotly.graph_objects as go

# Add scripts dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from unet3d import TinyUNet3D
from utils import best_device
from filament_3d_mp4 import render_2p5d, make_projections, load_data

from PIL import Image, ImageDraw, ImageFont

# Settings
TIFF_DIR = "tiffs3d"
device = best_device()

# Global Model Cache
models = {} # {type: model}

def get_model(model_type="Default"):
    if model_type not in models:
        # Default is 1-ch, Temporal variants are 3-ch (t-1, t, t+1)
        in_ch = 3 if "Temporal" in model_type else 1
        model = TinyUNet3D(in_ch=in_ch, out_ch=in_ch).to(device)
        
        # Determine path
        if model_type == "Temporal (Auto)":
            path = "models/filament_unet3d_temporal_auto.pt"
        elif model_type == "Temporal":
            path = "models/filament_unet3d_temporal.pt"
        else:
            path = "models/filament_unet3d.pt"
            
        if os.path.exists(path):
            print(f"Loading {model_type} model from {path}...")
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models[model_type] = model
    return models[model_type]

import threading
# Performance optimization caches
# (model_type, abs_fp, t) -> (vol, mask)
inference_cache = {}
inference_lock = threading.Lock()

# (model_type, abs_fp) -> (logits_sum, counts) dictionaries for sliding window
temporal_acc = {} 

video_cache = {}  # {abspath: norm_vol}
analysis_cache = {} # {abspath: {'labels': npz['filament_labels'], 'cells': npz['cell_masks']}}
filament_coords_cache = {} # {(abs_fp, fid): {t: (z, y, x)}}
render_cache = {} # (params) -> (pano, stack, ortho, det_bar)

def run_background_inference(abs_fp, model_type="Default"):
    """Background thread to pre-calculate all masks."""
    if abs_fp not in video_cache: return
    data = video_cache[abs_fp]
    T = data.shape[0]
    m = get_model(model_type)
    is_temporal = "Temporal" in model_type
    
    for t in range(T):
        key = (model_type, abs_fp, t)
        with inference_lock:
            if key in inference_cache: continue
        
        try:
            if is_temporal:
                # 3-frame sliding window logic
                t_prev, t_curr, t_next = max(0, t-1), t, min(T-1, t+1)
                window = np.stack([data[t_prev], data[t_curr], data[t_next]], axis=0) # (3, Z, H, W)
                inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = m(inp).squeeze(0).cpu().numpy() # (3, Z, H, W)
                
                # For now, just take the center frame for the cache to keep it simple,
                # or we could implement the full logit accumulator here.
                # To be consistent with update_dashboard, we'll just store the center.
                pred_mask = (logits[1] > 0.0).astype(np.float32)
                vol = data[t]
            else:
                vol = data[t].astype(np.float32)
                inp = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = m(inp).squeeze().cpu().numpy()
                pred_mask = (logits > 0.0).astype(np.float32)
            
            with inference_lock:
                inference_cache[key] = (vol, pred_mask)
        except Exception as e:
            print(f"Background inference error @ t={t}: {e}")
            break

def find_analysis_data(abs_fp):
    if abs_fp in analysis_cache:
        return analysis_cache[abs_fp]
        
    base_name = os.path.splitext(os.path.basename(abs_fp))[0]
    out_npz = f"results/{base_name}_analysis.npz"
    if os.path.exists(out_npz):
        try:
            npz = np.load(out_npz)
            # Load into memory to avoid repeated disk reads
            data = {
                'filament_labels': npz['filament_labels'],
                'cell_masks': npz['cell_masks']
            }
            analysis_cache[abs_fp] = data
            return data
        except Exception as e:
            print(f"Error loading analysis NPZ: {e}")
            return None
    return None

def find_training_mask(abs_fp, t):
    base_name = os.path.splitext(os.path.basename(abs_fp))[0]
    m_path = os.path.join("models/masks3d", f"{base_name}_t{t:04d}.npy")
    if os.path.exists(m_path):
        return np.load(m_path)
    return None

def get_color_for_id(fid, as_hex=False):
    """Simple deterministic color map for filament IDs."""
    if fid == 0: return "#000000" if as_hex else [0, 0, 0]
    colors = [
        [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0],
        [0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5], [0.5, 0, 1], [1, 0.5, 0]
    ]
    c = colors[int(fid) % len(colors)]
    if as_hex:
        return '#{:02x}{:02x}{:02x}'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
    return c

def generate_3d_plot(abs_fp, t, specific_fid=None):
    """
    Generates a Plotly 3D scatter plot of filaments at time t.
    If specific_fid is provided, only that filament is shown.
    """
    analysis = find_analysis_data(abs_fp)
    if analysis is None:
        return go.Figure()
    
    # Check coord cache first
    cache_key = (abs_fp, specific_fid)
    if specific_fid is not None and cache_key in filament_coords_cache:
        cached_t = filament_coords_cache[cache_key]
        if t in cached_t:
            z_idx, y_idx, x_idx = cached_t[t]
            print(f"DEBUG: Plotting {len(z_idx)} voxels for Fid {specific_fid} at T={t} (CACHED)")
            
            fig = go.Figure()
            color = get_color_for_id(specific_fid, as_hex=True)
            fig.add_trace(go.Scatter3d(
                z=z_idx, y=y_idx, x=x_idx,
                mode='markers',
                marker=dict(size=2.5, color=color, opacity=0.8, 
                            line=dict(width=0)),
                name=f"Filament {specific_fid}"
            ))
            
            target_vol = analysis['filament_labels']
            T, Z, H, W = target_vol.shape
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X (px)', range=[0, W]),
                    yaxis=dict(title='Y (px)', range=[0, H]),
                    zaxis=dict(title='Z (slices)', range=[0, Z]),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=H/W if W>0 else 1, z=Z*5/W if W>0 else 1) # Stretch Z for visibility
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                title=dict(text=f"Filament {specific_fid} — Frame {t} ({len(z_idx)} voxels)", 
                           x=0.5, y=0.9, xanchor='center', font=dict(color='white')),
                template="plotly_dark"
            )
            return fig
        else:
            # print(f"DEBUG: Fid {specific_fid} not present at T={t}")
            pass

    # Check tracking CSV for persistence/lookup
    base_name = os.path.splitext(os.path.basename(abs_fp))[0]
    csv_path = f"results/{base_name}_tracking.csv"
    if not os.path.exists(csv_path):
        return go.Figure()
    
    df = pd.read_csv(csv_path)
    
    if specific_fid is None:
        # Show all persistent
        counts = df['filament_id'].value_counts()
        target_ids = counts[counts >= 3].index.tolist()
    else:
        target_ids = [specific_fid]
    
    labels_vol = analysis['filament_labels'][t]
    fig = go.Figure()
    
    found_any = False
    for fid in target_ids:
        z_idx, y_idx, x_idx = np.where(labels_vol == fid)
        if len(z_idx) == 0: continue
        
        found_any = True
        color = get_color_for_id(fid, as_hex=True)
        
        fig.add_trace(go.Scatter3d(
            z=z_idx, y=y_idx, x=x_idx,
            mode='markers',
            marker=dict(size=2.5, color=color, opacity=0.8, 
                        line=dict(width=0)),
            name=f"Filament {fid}"
        ))
    
    if not found_any:
        # Show empty plot with message or something
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(opacity=0), name="No persistent filaments"))

    Z, H, W = labels_vol.shape
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (px)', range=[0, W]),
            yaxis=dict(title='Y (px)', range=[0, H]),
            zaxis=dict(title='Z (slices)', range=[0, Z]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=H/W if W>0 else 1, z=Z*5/W if W>0 else 1)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=dict(text=f"All Filaments — Frame {t}", x=0.5, y=0.9, xanchor='center', font=dict(color='white')),
        template="plotly_dark"
    )
    return fig

def get_persistent_filaments(abs_fp):
    """Returns a list of (filament_id, track_length) for persistent filaments."""
    base_name = os.path.splitext(os.path.basename(abs_fp))[0]
    csv_path = f"results/{base_name}_tracking.csv"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    counts = df['filament_id'].value_counts()
    persistent = counts[counts >= 3]
    return [(int(fid), int(length)) for fid, length in persistent.items()]

def update_dashboard(fp, t, model_type, show_mask, mask_alpha, show_train, live_3d=False):
    if not fp or t is None:
        return None, None, None, None, go.Figure()
    
    t = int(t)
    abs_fp = os.path.abspath(fp)
    
    render_key = (model_type, abs_fp, t, show_mask, mask_alpha, show_train)
    if render_key in render_cache and not live_3d:
        # We still need to generate the plot if live_3d is true, even if rest is cached
        pano, stack, ortho, det_bar = render_cache[render_key]
        return pano, stack, ortho, det_bar, go.Figure()

    if abs_fp not in video_cache:
        try:
            video_cache[abs_fp] = load_data(abs_fp)
        except Exception as e:
            print(f"Error loading video {abs_fp}: {e}")
            return None, None, None, None, go.Figure()
            
    data = video_cache[abs_fp]
    T = data.shape[0]
    if t >= T: return None, None, None, None, go.Figure()
    vol = data[t].astype(np.float32)

    key = (model_type, abs_fp, t)
    with inference_lock:
        in_cache = key in inference_cache
        
    if not in_cache:
        try:
            m = get_model(model_type)
            if "Temporal" in model_type:
                t_prev, t_curr, t_next = max(0, t-1), t, min(T-1, t+1)
                window = np.stack([data[t_prev], data[t_curr], data[t_next]], axis=0)
                inp = torch.from_numpy(window).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = m(inp).squeeze(0).cpu().numpy()
                pred_mask = (logits[1] > 0.0).astype(np.float32)
            else:
                inp = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = m(inp).squeeze().cpu().numpy()
                pred_mask = (logits > 0.0).astype(np.float32)
            
            with inference_lock:
                inference_cache[key] = (vol, pred_mask)
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None, None, None, go.Figure()
        
    with inference_lock:
        vol, mask = inference_cache[key]
        
    Z, H, W = vol.shape
    
    # Check render cache for the visual part
    if render_key in render_cache:
        pano_to_show, stack_to_show, ortho_to_show, det_bar = render_cache[render_key]
    else:
        # 1. Panorama
        pano_raw = np.hstack([vol[z] for z in range(Z)])
        pano_rgb = np.stack([pano_raw]*3, axis=-1)
        pano_display = pano_rgb.copy()
        
        analysis = find_analysis_data(abs_fp)
        if show_mask:
            if analysis is not None:
                labels_vol = analysis['filament_labels'][t]
                labels_pano = np.hstack([labels_vol[z] for z in range(Z)])
                overlay = np.zeros_like(pano_rgb)
                unique_ids = np.unique(labels_pano)
                for fid in unique_ids:
                    if fid == 0: continue
                    overlay[labels_pano == fid] = get_color_for_id(fid)
                m_idx = labels_pano > 0
                pano_display[m_idx] = pano_rgb[m_idx] * (1.0 - mask_alpha) + overlay[m_idx] * mask_alpha
                
                cells_2d = analysis['cell_masks'][t]
                cells_pano = np.tile(cells_2d, (1, Z))
                struct = ndimage.generate_binary_structure(2, 1)
                dilated = ndimage.binary_dilation(cells_pano > 0, structure=struct)
                edge = dilated ^ (cells_pano > 0)
                pano_display[edge] = [0.2, 0.2, 1.0]
            else:
                mask_pano = np.hstack([mask[z] for z in range(Z)])
                m_idx = mask_pano > 0.5
                green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                pano_display[m_idx] = pano_rgb[m_idx] * (1.0 - mask_alpha) + green * mask_alpha
                
        if show_train:
            t_mask = find_training_mask(abs_fp, t)
            if t_mask is not None:
                t_pano = np.hstack([t_mask[z] for z in range(Z)])
                t_idx = t_pano > 0.5
                orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
                pano_display[t_idx] = pano_display[t_idx] * 0.4 + orange * 0.6
                
        # 2. 2.5D Render
        stack_img = render_2p5d(vol, mask if show_mask else np.zeros_like(mask), shift_x=25, shift_y=-25)
        
        # 3. Ortho Grid
        ortho_grid = make_projections(vol, mask if show_mask else np.zeros_like(mask), z_scale=10)
        
        def to_uint8(img):
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)

        pano_to_show = to_uint8(pano_display)
        stack_to_show = to_uint8(stack_img)
        ortho_to_show = to_uint8(ortho_grid)

        # 4. Generate Detection Bar
        def render_detection_bar(abs_fp, model_type, T, current_t):
            bar_h, bar_w = 40, 1200
            bar = np.full((bar_h, bar_w, 3), 40, dtype=np.uint8)
            seg_w = bar_w / max(1, T)
            with inference_lock:
                for i in range(T):
                    k = (model_type, abs_fp, i)
                    if k in inference_cache:
                        _, m = inference_cache[k]
                        if m.max() > 0.5:
                            x_start = int(i * seg_w)
                            x_end = int((i+1) * seg_w)
                            bar[:, x_start:x_end] = [0, 255, 0]
            cx = int(current_t * seg_w)
            bar[:, cx:min(bar_w, cx+3)] = [255, 255, 255]
            return bar

        det_bar = render_detection_bar(abs_fp, model_type, T, t)
        
        # Cache visual results
        render_cache[render_key] = (pano_to_show, stack_to_show, ortho_to_show, det_bar)

    # 3D Plot - ONLY if Live 3D enabled (to save smoothness during playback)
    plot_3d = go.Figure()
    if live_3d:
        plot_3d = generate_3d_plot(abs_fp, t)
    else:
        # Minimal plot to indicate it's disabled
        plot_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='text', text=["3D Disabled (Enable 'Live 3D' for playback view)"]))

    return pano_to_show, stack_to_show, ortho_to_show, det_bar, plot_3d

def run_tracking_ui(fp, model_type):
    if not fp: return "No file selected.", None, None
    
    abs_fp = os.path.abspath(fp)
    is_auto = "--auto" if "Auto" in model_type else ""
    
    cmd = ["uv", "run", "python", "scripts/filament_3d_tracker.py", abs_fp]
    if is_auto: cmd.insert(-1, "--auto")
    
    print(f"Running tracking: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error: {result.stderr}", None, None
            
        base_name = os.path.splitext(os.path.basename(abs_fp))[0]
        csv_path = f"results/{base_name}_tracking.csv"
        png_path = f"results/{base_name}_tracking_summary.png"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            summary = f"✅ Tracking complete. Found {df['filament_id'].nunique()} filaments."
            return summary, df, png_path
        else:
            return "Tracking finished but no CSV found.", None, None
    except Exception as e:
        return f"Exception: {str(e)}", None, None

def on_file_change(fp, model_type):
    if not fp:
        return gr.update(maximum=0, value=0), "No data selected.", 0
    try:
        abs_fp = os.path.abspath(fp)
        if abs_fp not in video_cache:
            video_cache[abs_fp] = load_data(abs_fp)
        
        # Pre-load analysis data too
        find_analysis_data(abs_fp)
        
        data = video_cache[abs_fp]
        T = int(data.shape[0])
        status = f"✅ Loaded {os.path.basename(fp)}. {T} timepoints cached."
        
        # Auto-run tracking if results don't exist
        analysis = find_analysis_data(abs_fp)
        if analysis is None:
            status += " 🚀 Auto-starting tracking..."
            threading.Thread(target=run_tracking_ui, args=(abs_fp, model_type), daemon=True).start()
        else:
            status += " 📊 Tracking results found."
        
        # Start background inference for the SELECTED model
        threading.Thread(target=run_background_inference, args=(abs_fp, model_type), daemon=True).start()
        
        return gr.update(maximum=max(0, T-1), value=0), status, T
    except Exception as e:
        return gr.update(maximum=0, value=0), f"❌ Error: {str(e)}", 0

def get_file_list():
    exts = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(TIFF_DIR, ext)))
        files.extend(glob.glob(ext))
    return sorted(list(set([os.path.abspath(f) for f in files if os.path.isfile(f)])))

def play_video(fp, t_start, fps, T_state):
    if not fp or T_state is None or T_state <= 1: return
    
    current_t = int(t_start) if t_start is not None else 0
    total_t = int(T_state)
    
    # Play through all frames once from current
    for t in range(current_t, total_t):
        yield t
        time.sleep(1.0 / max(1, fps))

# Premium Dashboard CSS
CSS = """
.gradio-container { padding: 0px !important; max-width: 100% !important; margin: 0 !important; }
.gr-block { border-radius: 0px !important; border: none !important; margin: 0 !important; padding: 0 !important; }

/* Sidebar styling: Minimal and transparent */
.sidebar { background-color: #fbfbfb !important; border-right: 1px solid #eee !important; padding: 10px !important; }

/* Views Label Styling: separate Markdown blocks instead of absolute overlays */
.view-title { margin: 5px 0 0 10px !important; padding: 0 !important; }
.view-title h3 { margin: 0 !important; padding: 0 !important; font-size: 14px !important; font-weight: 600 !important; color: #444 !important;}

/* Image hover buttons */
.image-container button.icon-button { 
    opacity: 0; 
    transition: opacity 0.2s ease;
}
.image-container:hover button.icon-button { 
    opacity: 1 !important; 
}

.image-container .image-buttons {
    top: 2px !important;
    right: 2px !important;
    position: absolute !important;
    z-index: 100;
}

/* Image Viewer Cleanup */
.image-container { padding: 0 !important; background: #000 !important; }
.image-container img { border-radius: 0px; object-fit: contain !important; width: 100% !important; }

/* Control Group cleanup */
.gr-group { border: 1px solid #f0f0f0 !important; border-radius: 8px !important; }

/* Detection bar: full width at top */
.det-bar-container { margin: 0 !important; border-radius: 0 !important; border-bottom: 1px solid #eee !important; height: 30px !important; }
.gr-row, .gr-column { gap: 0 !important; }
"""

# GUI Construction
with gr.Blocks(title="3D Filament Dashboard — Temporal Edition") as demo:
    # State
    T_state = gr.State(value=int(0))
    
    with gr.Sidebar(label="Controls") as sidebar:
        gr.Markdown("## 🦠 Dashboard Controls")
        
        file_dropdown = gr.Dropdown(label="Select 3D TIFF", 
                                   choices=get_file_list(),
                                   value=get_file_list()[0] if get_file_list() else None)
        
        model_radio = gr.Radio(choices=["Default", "Temporal", "Temporal (Auto)"], 
                              label="Inference Model", value="Default")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            play_btn = gr.Button("▶️ Play", variant="primary")
            stop_btn = gr.Button("⏹️ Stop")
        
        with gr.Row():
            nav_prev = gr.Button("⬅️ Prev")
            nav_next = gr.Button("Next ➡️")
        
        fps_slider = gr.Slider(label="Playback FPS", minimum=1, maximum=30, step=1, value=10)
        time_slider = gr.Slider(label="Timepoint", minimum=0, maximum=10000, step=1, value=0)
        
        with gr.Group():
            mask_checkbox = gr.Checkbox(label="Show Mask Overlay", value=True)
            mask_alpha_slider = gr.Slider(label="Mask Alpha", minimum=0.0, maximum=1.0, value=0.7)
            show_train_checkbox = gr.Checkbox(label="Show Training Masks (Orange)", value=False)
            live_3d_checkbox = gr.Checkbox(label="Live 3D Visualization (Heavier)", value=False)
            
        status_text = gr.Markdown("Ready.")
        gr.Markdown("---")
        gr.Markdown("### Interactive Controls\n- **Full Screen**: Hover for buttons.\n- **Download**: High-res composite frames.")

    with gr.Column():
        # Detection Progress Bar
        gr.Markdown("### Detection Timeline (Green = Detected, White = Current)", elem_classes=["view-title"])
        det_bar_viewer = gr.Image(show_label=False, interactive=False, height=30,
                                 elem_classes=["det-bar-container"])
        
        with gr.Tabs():
            with gr.Tab("Visual Exploration"):
                # Main Viewers
                gr.Markdown("### 5-Plane Panorama (Z=0-4)", elem_classes=["view-title"])
                pano_image = gr.Image(show_label=False, 
                                     interactive=True, sources=[],
                                     elem_classes=["image-container"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 2.5D Volumetric Stack", elem_classes=["view-title"])
                        stack_image = gr.Image(show_label=False, 
                                              interactive=True, sources=[],
                                              elem_classes=["image-container"])
                    with gr.Column(scale=1):
                        gr.Markdown("### Orthogonal MIP (XY, YZ, XZ)", elem_classes=["view-title"])
                        ortho_image = gr.Image(show_label=False, 
                                              interactive=True, sources=[],
                                              elem_classes=["image-container"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🚀 Global 3D Filament View", elem_classes=["view-title"])
                        main_3d_plot = gr.Plot(label="Global 3D View")
            
            with gr.Tab("Analysis & Tracking"):
                gr.Markdown("### 3D Filament Quantitative Tracking")
                with gr.Row():
                    run_track_btn = gr.Button("🚀 Run 3D Tracking & Stats", variant="primary")
                
                tracking_status = gr.Markdown("Click to run tracking on the current file.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        tracking_table = gr.DataFrame(label="Tracking Data (CSV)", wrap=True)
                    with gr.Column(scale=3):
                        tracking_plot = gr.Image(label="Summary Stats", interactive=False)
            
            with gr.Tab("Interactive 3D View"):
                gr.Markdown("### Individual Filament 3D Histories (Track Length >= 3)")
                filament_selector = gr.CheckboxGroup(label="Select Filaments to Visualize", choices=[])
                
                with gr.Row():
                    # Slot 1
                    with gr.Column(visible=False) as slot1_col:
                        slot1_plot = gr.Plot(label="Slot 1")
                        slot1_slider = gr.Slider(label="Time (F1)", interactive=True)
                    # Slot 2
                    with gr.Column(visible=False) as slot2_col:
                        slot2_plot = gr.Plot(label="Slot 2")
                        slot2_slider = gr.Slider(label="Time (F2)", interactive=True)
                
                with gr.Row():
                    # Slot 3
                    with gr.Column(visible=False) as slot3_col:
                        slot3_plot = gr.Plot(label="Slot 3")
                        slot3_slider = gr.Slider(label="Time (F3)", interactive=True)
                    # Slot 4
                    with gr.Column(visible=False) as slot4_col:
                        slot4_plot = gr.Plot(label="Slot 4")
                        slot4_slider = gr.Slider(label="Time (F4)", interactive=True)
                
                with gr.Row():
                    # Slot 5
                    with gr.Column(visible=False) as slot5_col:
                        slot5_plot = gr.Plot(label="Slot 5")
                        slot5_slider = gr.Slider(label="Time (F5)", interactive=True)
                    # Slot 6
                    with gr.Column(visible=False) as slot6_col:
                        slot6_plot = gr.Plot(label="Slot 6")
                        slot6_slider = gr.Slider(label="Time (F6)", interactive=True)

                slot_cols = [slot1_col, slot2_col, slot3_col, slot4_col, slot5_col, slot6_col]
                slot_plots = [slot1_plot, slot2_plot, slot3_plot, slot4_plot, slot5_plot, slot6_plot]
                slot_sliders = [slot1_slider, slot2_slider, slot3_slider, slot4_slider, slot5_slider, slot6_slider]

    # Event Wiring
    def refresh_files():
        return gr.update(choices=get_file_list())

    refresh_btn.click(refresh_files, outputs=[file_dropdown])
    
    file_dropdown.change(on_file_change, 
                        inputs=[file_dropdown, model_radio], 
                        outputs=[time_slider, status_text, T_state])
    
    model_radio.change(on_file_change, 
                      inputs=[file_dropdown, model_radio], 
                      outputs=[time_slider, status_text, T_state])

    # Navigation logic
    def on_prev(t):
        if t is None: return 0
        return max(0, int(t) - 1)
    def on_next(t, T_state):
        if t is None: return 0
        ts = int(T_state) if T_state is not None else 0
        return min(max(0, ts - 1), int(t) + 1)

    nav_prev.click(on_prev, inputs=[time_slider], outputs=[time_slider])
    nav_next.click(on_next, inputs=[time_slider, T_state], outputs=[time_slider])

    # Playback logic
    play_event = play_btn.click(play_video, 
                                inputs=[file_dropdown, time_slider, fps_slider, T_state], 
                                outputs=[time_slider])
    
    stop_btn.click(fn=None, cancels=[play_event])

    # Dashboard update logic
    update_inputs = [file_dropdown, time_slider, model_radio, mask_checkbox, mask_alpha_slider, show_train_checkbox, live_3d_checkbox]
    update_outputs = [pano_image, stack_image, ortho_image, det_bar_viewer, main_3d_plot]
    
    # Selection logic for filaments
    def update_filament_selection(fp):
        if not fp: return gr.update(choices=[])
        abs_fp = os.path.abspath(fp)
        p_fils = get_persistent_filaments(abs_fp)
        choices = [f"Filament {fid} (Len {l})" for fid, l in sorted(p_fils)]
        return gr.update(choices=choices, value=[])

    file_dropdown.change(update_filament_selection, inputs=[file_dropdown], outputs=[filament_selector])

    def on_filament_select(selected, fp, T):
        # selected is list of strings like "Filament 1 (Len 10)"
        updates = []
        abs_fp = os.path.abspath(fp) if fp else None
        
        if abs_fp:
            analysis = find_analysis_data(abs_fp)
            if analysis is not None:
                labels_vol = analysis['filament_labels'] # (T, Z, H, W)
                
                for s in selected[:6]:
                    fid = int(s.split(" ")[1])
                    cache_key = (abs_fp, fid)
                    
                    # Pre-calculate coords for all timepoints if not cached
                    if cache_key not in filament_coords_cache:
                        print(f"Pre-calculating coords for Filament {fid}...")
                        t_coords = {}
                        for t_idx in range(labels_vol.shape[0]):
                            z, y, x = np.where(labels_vol[t_idx] == fid)
                            if len(z) > 0:
                                t_coords[t_idx] = (z, y, x)
                        filament_coords_cache[cache_key] = t_coords
        
        for i in range(len(selected[:6])):
            s = selected[i]
            fid = int(s.split(" ")[1])
            
            # Find first valid frame for this filament
            cache_key = (abs_fp, fid)
            t_start = 0
            if cache_key in filament_coords_cache:
                valid_ts = sorted(filament_coords_cache[cache_key].keys())
                if valid_ts:
                    t_start = valid_ts[0]
            
            # Initial plot at t_start
            fig = generate_3d_plot(abs_fp, t_start, specific_fid=fid) if abs_fp else go.Figure()
            updates.extend([gr.update(visible=True), fig, gr.update(maximum=max(0, int(T)-1), value=int(t_start))])
        
        # Hide remaining slots
        for _ in range(len(selected), 6):
            updates.extend([gr.update(visible=False), go.Figure(), gr.update()])
            
        return updates

    filament_selector.change(on_filament_select, 
                             inputs=[filament_selector, file_dropdown, T_state], 
                             outputs=slot_cols[0:1] + slot_plots[0:1] + slot_sliders[0:1] + \
                                     slot_cols[1:2] + slot_plots[1:2] + slot_sliders[1:2] + \
                                     slot_cols[2:3] + slot_plots[2:3] + slot_sliders[2:3] + \
                                     slot_cols[3:4] + slot_plots[3:4] + slot_sliders[3:4] + \
                                     slot_cols[4:5] + slot_plots[4:5] + slot_sliders[4:5] + \
                                     slot_cols[5:6] + slot_plots[5:6] + slot_sliders[5:6])

    # Local slider event handlers
    def make_slider_update(slot_idx):
        def update_local_plot(fp, t_local, selection):
            if not fp or not selection or len(selection) <= slot_idx:
                return go.Figure()
            abs_fp = os.path.abspath(fp)
            fid = int(selection[slot_idx].split(" ")[1])
            return generate_3d_plot(abs_fp, int(t_local), specific_fid=fid)
        return update_local_plot

    for i in range(6):
        fn = make_slider_update(i)
        slot_sliders[i].change(fn, inputs=[file_dropdown, slot_sliders[i], filament_selector], outputs=[slot_plots[i]])

    # Update on interaction
    time_slider.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    mask_checkbox.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    mask_alpha_slider.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    model_radio.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    show_train_checkbox.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    live_3d_checkbox.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)
    
    # Initial load update
    file_dropdown.change(update_dashboard, inputs=update_inputs, outputs=update_outputs)

    # Tracking event
    run_track_btn.click(run_tracking_ui, 
                       inputs=[file_dropdown, model_radio], 
                       outputs=[tracking_status, tracking_table, tracking_plot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, theme=gr.themes.Soft(), css=CSS)

