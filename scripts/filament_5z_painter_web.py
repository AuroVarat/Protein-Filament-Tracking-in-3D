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
current_folder = None
initial_folder = None
initial_file = None

if len(sys.argv) > 1:
    arg_path = sys.argv[1]
    if os.path.isdir(arg_path):
        current_folder = arg_path
        initial_folder = arg_path
    else:
        current_folder = os.path.dirname(arg_path) or "."
        initial_folder = current_folder
        initial_file = arg_path


def list_tiff_files(folderpath):
    if not folderpath:
        return []
    folderpath = os.path.abspath(os.path.expanduser(str(folderpath).strip()))
    if not os.path.isdir(folderpath):
        return []
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folderpath, pattern)))
    return sorted(set(files))

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


def load_folder(folderpath):
    global current_folder
    folderpath = os.path.abspath(os.path.expanduser(str(folderpath).strip())) if folderpath else ""
    current_folder = folderpath
    files = list_tiff_files(folderpath)
    if not files:
        return f"No TIFF files found in {folderpath}", gr.Dropdown(choices=[], value=None), gr.Slider(), None
    first_file = files[0]
    msg = load_video(first_file)
    update_dropdown = gr.Dropdown(choices=files, value=first_file)
    update_slider = gr.Slider(maximum=total_t - 1, value=0)
    img = get_pano_for_t(0)
    return f"{msg} | Folder files: {len(files)}", update_dropdown, update_slider, img

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
        folder_input = gr.Textbox(label="TIFF Folder", placeholder="tifs3d", scale=3, value=initial_folder)
        load_btn = gr.Button("Load Folder", variant="primary", scale=1)

    initial_choices = list_tiff_files(initial_folder) if initial_folder else []
    file_dropdown = gr.Dropdown(label="TIFF File", choices=initial_choices, value=initial_file)
    
    status_txt = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        t_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Timepoint (T)")
        nav_prev = gr.Button("⬅️ Prev Frame", elem_id="nav-prev-frame")
        nav_next = gr.Button("Next Frame ➡️", elem_id="nav-next-frame")

    gr.HTML(
        """
        <script>
        (() => {
          const state = {
            scale: 1,
            tx: 0,
            ty: 0,
            dragging: false,
            startX: 0,
            startY: 0,
          };

          const getEditorRoot = () => document.getElementById("filament-editor");

          const getViewport = () => {
            const root = getEditorRoot();
            if (!root) return null;
            return (
              root.querySelector('[data-testid="image-editor"]') ||
              root.querySelector('.image-container') ||
              root.querySelector('.gr-image-editor') ||
              root
            );
          };

          const getTransformTarget = () => {
            const viewport = getViewport();
            if (!viewport) return null;
            return (
              viewport.querySelector('canvas')?.closest('div') ||
              viewport.querySelector('canvas') ||
              viewport.querySelector('img') ||
              viewport
            );
          };

          const applyTransform = () => {
            const viewport = getViewport();
            const target = getTransformTarget();
            if (!viewport || !target) return;
            viewport.style.overflow = 'hidden';
            target.style.transformOrigin = '0 0';
            target.style.transform = `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;
            target.style.cursor = state.dragging ? 'grabbing' : 'default';
          };

          const eventInsideEditor = (event) => {
            const root = getEditorRoot();
            return !!(root && root.contains(event.target));
          };

          const handler = (event) => {
            const tag = (event.target && event.target.tagName || "").toLowerCase();
            const editable = tag === "input" || tag === "textarea" || event.target?.isContentEditable;
            if (editable) return;
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              document.getElementById("nav-prev-frame")?.click();
            } else if (event.key === "ArrowRight") {
              event.preventDefault();
              document.getElementById("nav-next-frame")?.click();
            } else if (event.key === "s" || event.key === "S") {
              event.preventDefault();
              document.getElementById("save-3d-mask")?.click();
            }
          };

          const wheelHandler = (event) => {
            if (!eventInsideEditor(event)) return;
            event.preventDefault();
            const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
            state.scale = Math.min(8, Math.max(0.25, state.scale * factor));
            applyTransform();
          };

          const mouseDownHandler = (event) => {
            if (event.button !== 2) return;
            if (!eventInsideEditor(event)) return;
            event.preventDefault();
            state.dragging = true;
            state.startX = event.clientX - state.tx;
            state.startY = event.clientY - state.ty;
            applyTransform();
          };

          const mouseMoveHandler = (event) => {
            if (!state.dragging) return;
            event.preventDefault();
            state.tx = event.clientX - state.startX;
            state.ty = event.clientY - state.startY;
            applyTransform();
          };

          const mouseUpHandler = (event) => {
            if (event.button !== 2 && !state.dragging) return;
            state.dragging = false;
            applyTransform();
          };

          const contextMenuHandler = (event) => {
            if (eventInsideEditor(event)) {
              event.preventDefault();
            }
          };

          const resetViewHandler = () => {
            state.scale = 1;
            state.tx = 0;
            state.ty = 0;
            applyTransform();
          };

          window.removeEventListener("keydown", window.__filamentPainterNavHandler__);
          window.__filamentPainterNavHandler__ = handler;
          window.addEventListener("keydown", handler);
          window.removeEventListener("wheel", window.__filamentPainterWheelHandler__, {capture: true});
          window.__filamentPainterWheelHandler__ = wheelHandler;
          window.addEventListener("wheel", wheelHandler, {passive: false, capture: true});
          window.removeEventListener("mousedown", window.__filamentPainterMouseDownHandler__, {capture: true});
          window.__filamentPainterMouseDownHandler__ = mouseDownHandler;
          window.addEventListener("mousedown", mouseDownHandler, {capture: true});
          window.removeEventListener("mousemove", window.__filamentPainterMouseMoveHandler__, {capture: true});
          window.__filamentPainterMouseMoveHandler__ = mouseMoveHandler;
          window.addEventListener("mousemove", mouseMoveHandler, {capture: true});
          window.removeEventListener("mouseup", window.__filamentPainterMouseUpHandler__, {capture: true});
          window.__filamentPainterMouseUpHandler__ = mouseUpHandler;
          window.addEventListener("mouseup", mouseUpHandler, {capture: true});
          window.removeEventListener("contextmenu", window.__filamentPainterContextMenuHandler__, {capture: true});
          window.__filamentPainterContextMenuHandler__ = contextMenuHandler;
          window.addEventListener("contextmenu", contextMenuHandler, {capture: true});
          window.removeEventListener("dblclick", window.__filamentPainterResetViewHandler__, {capture: true});
          window.__filamentPainterResetViewHandler__ = resetViewHandler;
          window.addEventListener("dblclick", resetViewHandler, {capture: true});
          const installObserver = () => {
            const root = getEditorRoot();
            if (!root) return false;
            const observer = new MutationObserver(() => applyTransform());
            observer.observe(root, { childList: true, subtree: true });
            setTimeout(applyTransform, 100);
            return true;
          };

          if (!installObserver()) {
            const retry = setInterval(() => {
              if (installObserver()) clearInterval(retry);
            }, 300);
            setTimeout(() => clearInterval(retry), 10000);
          }
        })();
        </script>
        """
    )
        
    gr.Markdown("""
    **Instructions**: 
    1. Select a brush color and size in the editor below. 
    2. Paint over the filaments. The 5 panels represent the 5 Z-planes horizontally `[Z=0 | Z=1 | Z=2 | Z=3 | Z=4]`. 
    3. Click **Save 3D Mask** when done with this frame.
    4. Mouse wheel zooms in and out. Right-click drag pans the view. Double-click resets the view.
    """)
    
    # The unified image editor for the panorama
    editor = gr.ImageEditor(
        label="5-Plane Z-Stack Panorama (Paint Here)",
        type="numpy",
        image_mode="RGBA",
        brush=gr.Brush(colors=["#00FF00"], color_mode="fixed"),
        elem_id="filament-editor"
    )
    
    save_btn = gr.Button("💾 Save 3D Mask", variant="primary", elem_id="save-3d-mask")
    save_status = gr.Textbox(label="Save Status", interactive=False)

    def on_load(folderpath):
        return load_folder(folderpath)

    load_btn.click(on_load, inputs=[folder_input], outputs=[status_txt, file_dropdown, t_slider, editor])

    def on_select_file(filepath):
        if not filepath:
            return "No TIFF selected.", gr.Slider(), None
        msg = load_video(filepath)
        if normd_vol is not None:
            update_slider = gr.Slider(maximum=total_t - 1, value=0)
            img = get_pano_for_t(0)
            return msg, update_slider, img
        return msg, gr.Slider(), None

    file_dropdown.change(on_select_file, inputs=[file_dropdown], outputs=[status_txt, t_slider, editor])
    
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

    if initial_folder:
        demo.load(
            lambda folder: on_load(folder or initial_folder),
            inputs=[folder_input],
            outputs=[status_txt, file_dropdown, t_slider, editor],
        )

if __name__ == "__main__":
    print("Starting Web UI on port 7860... Access it via your browser!")
    demo.launch(server_name="0.0.0.0", server_port=7860)
