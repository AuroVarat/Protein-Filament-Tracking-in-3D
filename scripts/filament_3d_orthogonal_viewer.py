#!/usr/bin/env python3
"""
Filament 3D Orthogonal Viewer — Interactive Mask & Projection Explorer

Features:
- Orthogonal MIPs (XY, XZ, YZ) of the selected filament mask.
- Interactive 3D Voxel View (Plotly).
- Temporal slider to scrub through frames.
- On-the-fly inference to extract masks for any selected Cell ID.
"""

import os
import glob
import tifffile
import numpy as np
import pandas as pd
import torch
import gradio as gr
import plotly.graph_objects as go
from scipy import ndimage
from PIL import Image

from unet3d import TinyUNet3D
from utils import best_device

# Global Model & Device
DEVICE = best_device()
MODEL = None

def load_model(path="models/filament_unet3d_temporal_auto.pt"):
    global MODEL
    if MODEL is None:
        MODEL = TinyUNet3D(in_ch=3, out_ch=3).to(DEVICE)
        MODEL.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        MODEL.eval()
    return MODEL

# State Caching
CACHE = {
    "video_path": None,
    "img_stack": None, # (T, Z, C, H, W)
    "cell_masks": {},  # t -> cell_mask
    "filament_masks": {} # (t, fid) -> masked_voxels
}

def get_video_files():
    return sorted(glob.glob("tiffs3d/*.tif"))

def get_cellpose_model():
    from cellpose import models
    return models.CellposeModel(model_type='cyto', gpu=torch.cuda.is_available())

CP_MODEL = None

def get_mask_for_frame(video_path, t, fid):
    global CP_MODEL, MODEL
    if CACHE["video_path"] != video_path:
        CACHE["video_path"] = video_path
        CACHE["img_stack"] = tifffile.imread(video_path).astype(np.float32)
        CACHE["cell_masks"] = {}
        CACHE["filament_masks"] = {}
        
    img = CACHE["img_stack"]
    T, Z, C, H, W = img.shape
    
    # 1. Inference for frame t
    t_p, t_c, t_n = max(0, t-1), t, min(T-1, t+1)
    # Filament channel is 1
    v_fil = img[:, :, 1, :, :]
    # Local normalization for the window
    win = v_fil[[t_p, t_c, t_n]]
    mn, mx = win.min(), win.max()
    win_norm = (win - mn) / (mx - mn + 1e-6)
    
    load_model()
    with torch.no_grad():
        inp = torch.from_numpy(win_norm).float().unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(MODEL(inp)).squeeze(0).cpu().numpy()[1]
        pred_mask = (probs > 0.5).astype(np.uint8)

    # 2. Z-Localization (optional but helps clarity)
    # We'll stick to full 3D mask for "collection of voxels"
    
    # 3. Cell Segmentation for frame t
    if t not in CACHE["cell_masks"]:
        if CP_MODEL is None: CP_MODEL = get_cellpose_model()
        bf_mid = img[t, Z//2, 0, :, :]
        c_mask, _, _ = CP_MODEL.eval(bf_mid, diameter=25, channels=[0,0])
        CACHE["cell_masks"][t] = c_mask
    
    cell_mask = CACHE["cell_masks"][t]
    
    # 4. Filter for selected fid (cell_id)
    # Replicate 2D cell mask to 3D
    cell_mask_3d = np.repeat(cell_mask[np.newaxis, :, :], Z, axis=0)
    final_mask = (pred_mask > 0) & (cell_mask_3d == fid)
    
    return final_mask

def plot_orthogonal_views(mask):
    if not np.any(mask):
        return [None]*3
    
    # MIPs
    xy = np.max(mask, axis=0) # (H, W)
    xz = np.max(mask, axis=1) # (Z, W)
    yz = np.max(mask, axis=2) # (Z, H)
    
    def to_img(m, aspect=None):
        img_arr = (m * 255).astype(np.uint8)
        # Apply some coloring/upscaling if needed
        return Image.fromarray(img_arr)

    return to_img(xy), to_img(xz), to_img(yz)

def generate_3d_voxels(mask):
    if not np.any(mask):
        return go.Figure()
    
    z, y, x = np.where(mask)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z, # Color by Z-plane
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    # Fixed range to match video dimensions if we have stack
    if CACHE["img_stack"] is not None:
        T, Z, H, W = CACHE["img_stack"].shape[0], CACHE["img_stack"].shape[1], CACHE["img_stack"].shape[3], CACHE["img_stack"].shape[4]
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, W]),
                yaxis=dict(range=[0, H]),
                zaxis=dict(range=[0, Z]),
                aspectmode='data'
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0)
        )
    return fig

# UI logic
def update_options(video_path):
    base = os.path.basename(video_path).replace(".tif", "_tracking.csv")
    csv_path = os.path.join("results", base)
    if not os.path.exists(csv_path):
        return gr.update(choices=[], value=None), gr.update(maximum=0), "No tracking data found for this video."
    
    df = pd.read_csv(csv_path)
    fids = sorted(df['filament_id'].unique())
    choices = [int(f) for f in fids]
    max_t = df['frame'].max()
    
    return gr.update(choices=choices, value=choices[0] if choices else None), gr.update(maximum=int(max_t)), f"Loaded {len(choices)} IDs."

def update_view(video_path, fid, t):
    if not video_path or fid is None:
        return [None]*3 + [go.Figure()]
    
    mask = get_mask_for_frame(video_path, int(t), int(fid))
    xy, xz, yz = plot_orthogonal_views(mask)
    fig_3d = generate_3d_voxels(mask)
    return xy, xz, yz, fig_3d

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🔳 Filament Orthogonal Mask Viewer")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Dropdown(label="Select Video", choices=get_video_files())
            fid_input = gr.Dropdown(label="Filament (Cell) ID", choices=[])
            frame_slider = gr.Slider(label="Frame", minimum=0, maximum=50, step=1, value=0)
            status = gr.Markdown("Select a video to begin.")
            
        with gr.Column(scale=3):
            with gr.Row():
                xy_view = gr.Image(label="XY Projection (Top View)", height=300)
                xz_view = gr.Image(label="XZ Projection (Side View)", height=150)
                yz_view = gr.Image(label="YZ Projection (Front View)", height=300)
            
            view_3d = gr.Plot(label="Interactive 3D Voxels")

    # Events
    video_input.change(update_options, inputs=[video_input], outputs=[fid_input, frame_slider, status])
    
    inputs = [video_input, fid_input, frame_slider]
    outputs = [xy_view, xz_view, yz_view, view_3d]
    
    video_input.change(update_view, inputs=inputs, outputs=outputs)
    fid_input.change(update_view, inputs=inputs, outputs=outputs)
    frame_slider.change(update_view, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
