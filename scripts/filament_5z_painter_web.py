import glob
import os
import sys

import gradio as gr
import numpy as np
import tifffile


MASK_DIR = "models/masks3d"
PAINTER_PORT = int(os.environ.get("FILAMENT_PAINTER_PORT", "7860"))

os.makedirs(MASK_DIR, exist_ok=True)

PAINTER_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(85, 144, 111, 0.10), transparent 28%),
    linear-gradient(180deg, #f8fbf9 0%, #f4f8f5 100%);
  color: #1f3528;
  font-family: "Segoe UI", sans-serif;
}

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-group {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #c7d8cc;
  border-radius: 18px;
  box-shadow: 0 18px 42px rgba(37, 65, 49, 0.08);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label {
  color: #1f3528;
}

.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  background: linear-gradient(135deg, #2f7d5b 0%, #245c43 100%) !important;
  border-color: #245c43 !important;
  color: white !important;
}

.gradio-container button:not(.primary) {
  border-color: #2f7d5b !important;
  color: #245c43 !important;
}

#filament-editor {
  border: 1px solid #c7d8cc;
  border-radius: 18px;
  overflow: hidden;
}
"""


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

    img = tifffile.imread(filepath).astype(np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    elif img.ndim == 5:
        img = img[:, :, 1, :, :]

    total_t, Z_DIM, H_DIM, W_DIM = img.shape

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

    vol = normd_vol[t_idx]
    pano = np.hstack([vol[z] for z in range(Z_DIM)])
    pano_rgb = (np.stack([pano] * 3, axis=-1) * 255).astype(np.uint8)

    base = os.path.splitext(os.path.basename(current_filepath))[0]
    mask_path = os.path.join(MASK_DIR, f"{base}_t{t_idx:04d}.npy")
    if os.path.exists(mask_path):
        mask_vol = np.load(mask_path)
        mask_pano = np.hstack([mask_vol[z] for z in range(Z_DIM)])
        rgba = np.zeros((*mask_pano.shape, 4), dtype=np.uint8)
        rgba[mask_pano > 0.5] = [0, 255, 0, 150]
        return {"background": pano_rgb, "layers": [rgba], "composite": None}

    return pano_rgb


def load_folder(folderpath):
    global current_folder
    folderpath = os.path.abspath(os.path.expanduser(str(folderpath).strip())) if folderpath else ""
    current_folder = folderpath
    files = list_tiff_files(folderpath)
    if not files:
        return f"No TIFF files found in {folderpath}", gr.Dropdown(choices=[], value=None), gr.Slider(), None

    first_file = files[0]
    msg = load_video(first_file)
    return (
        f"{msg} | Folder files: {len(files)}",
        gr.Dropdown(choices=files, value=first_file),
        gr.Slider(maximum=total_t - 1, value=0),
        get_pano_for_t(0),
    )


def save_mask(t_idx, editor_dict):
    if normd_vol is None or editor_dict is None:
        return "No data to save."

    layers = editor_dict.get("layers", [])
    if not layers:
        return "No mask drawn."

    combined_alpha = np.zeros((H_DIM, Z_DIM * W_DIM), dtype=np.float32)
    for layer in layers:
        if layer.shape[-1] == 4:
            alpha = layer[:, :, 3] / 255.0
            combined_alpha = np.maximum(combined_alpha, alpha)

    binary_pano = (combined_alpha > 0.1).astype(np.float32)
    mask_vol = np.zeros((Z_DIM, H_DIM, W_DIM), dtype=np.float32)
    for z in range(Z_DIM):
        start_x = z * W_DIM
        end_x = (z + 1) * W_DIM
        mask_vol[z] = binary_pano[:, start_x:end_x]

    base = os.path.splitext(os.path.basename(current_filepath))[0]
    out_path = os.path.join(MASK_DIR, f"{base}_t{int(t_idx):04d}.npy")
    np.save(out_path, mask_vol)

    total_saved = sum(1 for path in glob.glob(os.path.join(MASK_DIR, "*.npy")) if np.load(path).max() > 0)
    return f"Saved mask for frame {int(t_idx)} to {os.path.basename(out_path)}. Total annotated: {total_saved}"


with gr.Blocks(title="Filament 3D Annotator", css=PAINTER_CSS) as demo:
    gr.Markdown("# Filament 3D Annotator")
    gr.Markdown("Annotate the 5 Z-planes side-by-side from the main application.")

    with gr.Row():
        folder_input = gr.Textbox(label="TIFF Folder", placeholder="tifs3d", scale=3, value=initial_folder)
        load_btn = gr.Button("Load Folder", variant="primary", scale=1)

    initial_choices = list_tiff_files(initial_folder) if initial_folder else []
    file_dropdown = gr.Dropdown(label="TIFF File", choices=initial_choices, value=initial_file)
    status_txt = gr.Textbox(label="Status", interactive=False)

    gr.HTML(
        """
        <script>
        (() => {
          const state = { scale: 1, tx: 0, ty: 0, dragging: false, startX: 0, startY: 0 };
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
          const keyHandler = (event) => {
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
            if (event.button !== 2 || !eventInsideEditor(event)) return;
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
            if (eventInsideEditor(event)) event.preventDefault();
          };
          const resetViewHandler = () => {
            state.scale = 1;
            state.tx = 0;
            state.ty = 0;
            applyTransform();
          };

          window.removeEventListener("keydown", window.__filamentPainterNavHandler__);
          window.__filamentPainterNavHandler__ = keyHandler;
          window.addEventListener("keydown", keyHandler);

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

    editor = gr.ImageEditor(
        label="5-Plane Z-Stack Panorama (Paint Here)",
        type="numpy",
        image_mode="RGBA",
        brush=gr.Brush(colors=["#00FF00"], color_mode="fixed"),
        elem_id="filament-editor",
    )

    with gr.Row():
        nav_prev = gr.Button("Prev Frame", elem_id="nav-prev-frame")
        t_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Timepoint (T)")
        nav_next = gr.Button("Next Frame", elem_id="nav-next-frame")

    save_btn = gr.Button("Save 3D Mask", variant="primary", elem_id="save-3d-mask")
    save_status = gr.Textbox(label="Save Status", interactive=False)

    gr.Markdown(
        """
        **Instructions**:
        1. Choose the TIFF folder and file.
        2. Paint over the filaments in the panoramic view.
        3. The five horizontal panels represent `[Z=0 | Z=1 | Z=2 | Z=3 | Z=4]`.
        4. Mouse wheel zooms, right-click drag pans, and double-click resets the view.
        5. Press `S` or click **Save 3D Mask** to persist the current frame.
        """
    )

    def on_load(folderpath):
        return load_folder(folderpath)

    def on_select_file(filepath):
        if not filepath:
            return "No TIFF selected.", gr.Slider(), None
        msg = load_video(filepath)
        if normd_vol is not None:
            return msg, gr.Slider(maximum=total_t - 1, value=0), get_pano_for_t(0)
        return msg, gr.Slider(), None

    def on_slider_change(t_value):
        if normd_vol is None:
            return None
        return get_pano_for_t(int(t_value))

    def on_prev(t_value):
        new_t = max(0, int(t_value) - 1)
        return new_t, get_pano_for_t(new_t)

    def on_next(t_value):
        if normd_vol is None:
            return t_value, None
        new_t = min(total_t - 1, int(t_value) + 1)
        return new_t, get_pano_for_t(new_t)

    load_btn.click(on_load, inputs=[folder_input], outputs=[status_txt, file_dropdown, t_slider, editor])
    file_dropdown.change(on_select_file, inputs=[file_dropdown], outputs=[status_txt, t_slider, editor])
    t_slider.change(on_slider_change, inputs=[t_slider], outputs=[editor])
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
    print(f"Starting Web UI on port {PAINTER_PORT}... Access it via your browser.")
    demo.launch(server_name="0.0.0.0", server_port=PAINTER_PORT)
