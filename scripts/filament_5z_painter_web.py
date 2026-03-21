import gradio as gr
import tifffile
import numpy as np
import os
import sys
import glob

MASK_DIR = "models/masks3d"
os.makedirs(MASK_DIR, exist_ok=True)

# Global variables to hold the loaded TIFF
normd_vol = None
current_filepath = None
total_t = 0
Z_DIM, H_DIM, W_DIM = 0, 0, 0

def load_video(filepath):
    global normd_vol, current_filepath, total_t, Z_DIM, H_DIM, W_DIM
    if not os.path.exists(filepath):
        return f"File {filepath} not found."
        
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    elif img.ndim == 5:
        img = img[:, :, 1, :, :]  # Use SECOND channel
        
    total_t, Z_DIM, H_DIM, W_DIM = img.shape
    
    # Normalize per 2D slice
    norm = np.zeros_like(img)
    for t in range(total_t):
        for z in range(Z_DIM):
            mn, mx = img[t, z].min(), img[t, z].max()
            if mx > mn:
                norm[t, z] = (img[t, z] - mn) / (mx - mn)
            
    normd_vol = norm
    current_filepath = filepath
    return f"Loaded: {os.path.basename(filepath)} | Timesteps: {total_t} | Shape: {Z_DIM}x{H_DIM}x{W_DIM}"

def get_pano_for_t(t_idx):
    if normd_vol is None:
        return None
    vol = normd_vol[t_idx]  # (Z, H, W)
    # Stitch Z slices horizontally: shape (H, Z*W)
    pano = np.hstack([vol[z] for z in range(Z_DIM)])
    # Convert to RGB uint8 for Gradio
    pano_rgb = (np.stack([pano]*3, axis=-1) * 255).astype(np.uint8)
    
    # Check if mask exists to pre-load as a faint green layer
    base = os.path.splitext(os.path.basename(current_filepath))[0]
    mask_path = os.path.join(MASK_DIR, f"{base}_t{t_idx:04d}.npy")
    
    if os.path.exists(mask_path):
        mask_vol = np.load(mask_path)
        mask_pano = np.hstack([mask_vol[z] for z in range(Z_DIM)])
        # Create an RGBA layer for the mask (green with opacity)
        rgba = np.zeros((*mask_pano.shape, 4), dtype=np.uint8)
        rgba[mask_pano > 0.5] = [0, 255, 0, 150]
        
        # Gradio 4+ ImageEditor accepts a dict to pre-populate layers
        return {
            "background": pano_rgb,
            "layers": [rgba],
            "composite": None
        }
    
    return pano_rgb

def save_mask(t_idx, editor_dict):
    if normd_vol is None or editor_dict is None:
        return "No data to save."
        
    # editor_dict is a dict with 'background', 'layers', 'composite'
    layers = editor_dict.get('layers', [])
    
    if not layers:
        return "No mask drawn!"
        
    # The user might have drawn on multiple layers, combine their alpha channels
    combined_alpha = np.zeros((H_DIM, Z_DIM * W_DIM), dtype=np.float32)
    
    for layer in layers:
        # layer is an RGBA numpy array
        if layer.shape[-1] == 4:
            alpha = layer[:, :, 3] / 255.0
            combined_alpha = np.maximum(combined_alpha, alpha)
            
    # Threshold to create boolean mask
    binary_pano = (combined_alpha > 0.1).astype(np.float32)
    
    # Split the panoramic mask back into Z slices
    mask_vol = np.zeros((Z_DIM, H_DIM, W_DIM), dtype=np.float32)
    for z in range(Z_DIM):
        start_x = z * W_DIM
        end_x = (z + 1) * W_DIM
        mask_vol[z] = binary_pano[:, start_x:end_x]
        
    # Save it
    base = os.path.splitext(os.path.basename(current_filepath))[0]
    out_path = os.path.join(MASK_DIR, f"{base}_t{t_idx:04d}.npy")
    np.save(out_path, mask_vol)
    
    # Count total saves
    total_saved = sum(1 for p in glob.glob(os.path.join(MASK_DIR, "*.npy")) if np.load(p).max() > 0)
    
    return f"Saved mask for frame {t_idx} to {os.path.basename(out_path)}. Total annotated: {total_saved}"

with gr.Blocks(title="Filament 3D Annotator") as demo:
    gr.Markdown("# 🦠 Filament 3D Annotator (Web UI)")
    gr.Markdown("Since you are on a headless server, this web interface allows you to annotate the 5 Z-planes side-by-side.")
    
    with gr.Row():
        file_input = gr.Textbox(label="TIFF File Path", placeholder="tifs3d/volume1.tif", scale=3)
        load_btn = gr.Button("Load Video", variant="primary", scale=1)
    
    status_txt = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        t_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Timepoint (T)")
        nav_prev = gr.Button("⬅️ Prev Frame")
        nav_next = gr.Button("Next Frame ➡️")
        
    gr.Markdown("""
    **Instructions**: 
    1. Select a brush color and size in the editor below. 
    2. Paint over the filaments. The 5 panels represent the 5 Z-planes horizontally `[Z=0 | Z=1 | Z=2 | Z=3 | Z=4]`. 
    3. Click **Save 3D Mask** when done with this frame.
    """)
    
    # The unified image editor for the panorama
    editor = gr.ImageEditor(
        label="5-Plane Z-Stack Panorama (Paint Here)",
        type="numpy",
        image_mode="RGBA",
        brush=gr.Brush(colors=["#00FF00"], color_mode="fixed")
    )
    
    save_btn = gr.Button("💾 Save 3D Mask", variant="primary")
    save_status = gr.Textbox(label="Save Status", interactive=False)

    def on_load(filepath):
        msg = load_video(filepath)
        if normd_vol is not None:
            # Update slider to max T
            update_slider = gr.Slider(maximum=total_t - 1, value=0)
            img = get_pano_for_t(0)
            return msg, update_slider, img
        return msg, gr.Slider(), None

    load_btn.click(on_load, inputs=[file_input], outputs=[status_txt, t_slider, editor])
    
    def on_slider_change(t):
        if normd_vol is None: return None
        return get_pano_for_t(int(t))
        
    t_slider.change(on_slider_change, inputs=[t_slider], outputs=[editor])
    
    def on_prev(t):
        new_t = max(0, int(t) - 1)
        return new_t, get_pano_for_t(new_t)
        
    def on_next(t):
        if normd_vol is None: return t, None
        new_t = min(total_t - 1, int(t) + 1)
        return new_t, get_pano_for_t(new_t)

    nav_prev.click(on_prev, inputs=[t_slider], outputs=[t_slider, editor])
    nav_next.click(on_next, inputs=[t_slider], outputs=[t_slider, editor])
    
    save_btn.click(save_mask, inputs=[t_slider, editor], outputs=[save_status])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        load_video(sys.argv[1])
        
    print("Starting Web UI on port 7860... Access it via your browser!")
    demo.launch(server_name="0.0.0.0", server_port=7860)
