import streamlit as st
import tifffile
import numpy as np
import os
import glob
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates

# Configure dashboard page
st.set_page_config(layout="wide", page_title="Hyperstack TIF Viewer")

st.title("Hyperstack TIF Previewer")

@st.cache_data(ttl=60)
def get_local_tif_files():
    base_dir = "."
    files = glob.glob(os.path.join(base_dir, "**", "*.tif"), recursive=True)
    files.extend(glob.glob(os.path.join(base_dir, "**", "*.tiff"), recursive=True))
    files = [f for f in files if "/." not in f and "\\." not in f] # Ignore hidden folders
    return sorted(files)

local_files = get_local_tif_files()

# --- FILE UPLOAD / SELECTION ---
input_method = st.sidebar.radio("File Input Method", [
    "Select Found File (Fast)", 
    "Enter Custom Path (Fast)", 
    "Upload File (Slower)"
])

if input_method == "Upload File (Slower)":
    source = st.sidebar.file_uploader("Upload a Hyperstack TIF", type=["tif", "tiff"])
    if source is None:
        st.info("Please upload a TIF file from the sidebar to begin.")
        st.stop()
elif input_method == "Select Found File (Fast)":
    if not local_files:
        st.info("No .tif files found automatically in this directory. Try uploading or entering a custom path.")
        st.stop()
    selected = st.sidebar.selectbox("Choose a TIF file:", local_files)
    source = os.path.abspath(selected)
else:
    source = st.sidebar.text_input("Absolute path to TIF file:", value="")
    if not source:
        st.info("Please enter the exact file path to your TIF file (e.g., /Users/.../file.tif)")
        st.stop()
    if not os.path.exists(source):
        st.error(f"File not found: {source}")
        st.stop()

# --- DATA LOADING ---
@st.cache_resource
def load_image(file_source):
    # If the user provides a string path, we can memory map it! This means the 1.44GB 
    # file isn't loaded into RAM; numpy just streams exactly the slice it needs from disk.
    if isinstance(file_source, str):
        data = tifffile.memmap(file_source)
        with tifffile.TiffFile(file_source) as tif:
            axes = tif.series[0].axes
            is_rgb = getattr(tif, 'is_rgb', getattr(tif.series[0], 'is_rgb', False))
        return data, axes, is_rgb
    else:
        # Standard in-memory upload
        with tifffile.TiffFile(file_source) as tif:
            data = tif.asarray()
            axes = tif.series[0].axes
            is_rgb = getattr(tif, 'is_rgb', getattr(tif.series[0], 'is_rgb', False))
        return data, axes, is_rgb

with st.spinner("Loading hyperstack... memmapping takes 0 seconds!"):
    try:
        data, axes, is_rgb = load_image(source)
    except Exception as e:
        st.error(f"Error loading TIF: {e}")
        st.stop()

st.sidebar.success(f"File successfully loaded!")
st.sidebar.write(f"**Shape:** {data.shape}")
st.sidebar.write(f"**Detected Axes:** {axes if axes else 'Unknown'}")

# Initialize session state for ROI editing
process_path = source if isinstance(source, str) else "uploaded_temp.tif"

if "found_centers" not in st.session_state:
    st.session_state.found_centers = None
if "processed_path" not in st.session_state:
    st.session_state.processed_path = None
if "extraction_complete" not in st.session_state:
    st.session_state.extraction_complete = False

# Reset state if new file is loaded
if st.session_state.processed_path != process_path:
    st.session_state.found_centers = None
    st.session_state.extraction_complete = False
    st.session_state.processed_path = process_path

# Normalize pixel intensity safely so Streamlit can render it
def normalize_for_display(img):
    if img.dtype == np.uint8:
        return img
    
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
        
    return (img_norm * 255).astype(np.uint8)

# --- DYNAMIC PREVIEW UI ---
if st.session_state.found_centers is None:
    st.subheader("Interactive Preview")

    shape = data.shape
    ndim = len(shape)

    # Determine the spatial dimensions (Y, X). If the image is natively RGB, 
    # 'tifffile' typically puts the color channel last.
    is_rgb = is_rgb or (axes and axes[-1] in ['S', 'C'] and shape[-1] in [3, 4])
    display_dims = 3 if is_rgb else 2

    if ndim < 2:
        st.error("Image has less than 2 dimensions. This is not a valid 2D/3D image.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Controls")

    indices = []
    # For every dimension that is NOT part of the spatial Y/X display, create a slider
    for i in range(ndim - display_dims):
        axis_label = axes[i] if axes and i < len(axes) else f"Dimension {i}"
        max_val = shape[i] - 1
        
        if max_val > 0:
            val = st.sidebar.slider(f"{axis_label} Index", 0, max_val, 0)
            indices.append(val)
        else:
            indices.append(0)

    # Slice the multi-dimensional array down to a 2D (or 3D RGB) slice
    if ndim <= display_dims:
        preview_img = data
    else:
        preview_img = data[tuple(indices)]

    st.image(normalize_for_display(preview_img), use_container_width=True, clamp=True)

else:
    st.markdown("---")
    if st.button("← Cancel ROI Editing & Return to Raw Preview"):
        st.session_state.found_centers = None
        st.rerun()

# --- ROI EXTRACTION SECTION ---
st.markdown("---")
st.subheader("Automated ROI Extraction (Cellpose)")

import sys
# Ensure the script directory is accessible for import
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fillipo", "scripts")
if script_dir not in sys.path:
    sys.path.append(script_dir)
from cellpose_roi_extractor import CellposeRoiExtractor

if st.session_state.found_centers is None:
    if st.button("1. Find Cell ROIs", type="primary"):
        with st.spinner("Processing image through Cellpose (Apple Silicon GPU)... this might take a minute."):
            # If the user uploaded a file instead of using a local path, write it to a temp file
            if not isinstance(source, str) and not os.path.exists(process_path):
                with open(process_path, "wb") as f:
                    f.write(source.getvalue())
                
            base_name = os.path.splitext(os.path.basename(process_path))[0]
            out_plot = f"{base_name}_dbscan_plot.png"
            
            st.write("Initializing Cellpose model... (MPS GPU)")
            extractor = CellposeRoiExtractor(eps=60)
            
            st.write("Finding cell clusters and generating visualizations...")
            centers = extractor.visualize_clusters(process_path, output_plot=out_plot, data=data)
            st.session_state.found_centers = centers
            st.rerun()

if st.session_state.found_centers is not None and not st.session_state.get('extraction_complete', False):
    base_name = os.path.splitext(os.path.basename(process_path))[0]
    out_dir = f"{base_name}_extracted_crops"

    st.markdown("### Interactive ROI Editor")
    st.write("Click the **Red X** on the top right of a box to delete it. Click anywhere outside the boxes to **add** a new ROI.")
    
    import cv2
    from streamlit_image_coordinates import streamlit_image_coordinates
    
    # Extract middle Z slice
    if len(data.shape) >= 5: # T, Z, C, Y, X
        z_idx = data.shape[1] // 2
        raw_img_bf = data[0, z_idx, 0, :, :]
        other_channel = 1 if data.shape[2] > 1 else 0
        raw_img_other = data[0, z_idx, other_channel, :, :]
    else:
        raw_img_bf = data[0, 0] if data.ndim > 2 else data
        raw_img_other = raw_img_bf
        
    half = 128 // 2 # Fixed cell crop size
    btn_size = 30 # Size of the red close box
    padding = 10 # Extra invisible pad for the click area
        
    def prepare_interactive_img(raw_img):
        img_disp = normalize_for_display(raw_img)
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
        
        # Draw all boxes and close buttons on the original large image
        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            # Draw green bounding box
            cv2.rectangle(img_rgb, (x - half, y - half), (x + half, y + half), (0, 255, 0), 3)
            
            # Draw Red Close Button
            tr_x, tr_y = x + half - btn_size, y - half
            cv2.rectangle(img_rgb, (tr_x, tr_y), (tr_x + btn_size, tr_y + btn_size), (255, 0, 0), -1) 
            cv2.line(img_rgb, (tr_x + 5, tr_y + 5), (tr_x + btn_size - 5, tr_y + btn_size - 5), (255, 255, 255), 2)
            cv2.line(img_rgb, (tr_x + btn_size - 5, tr_y + 5), (tr_x + 5, tr_y + btn_size - 5), (255, 255, 255), 2)
            
            # Draw number
            cv2.putText(img_rgb, str(i + 1), (x - half + 5, y - half + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        # Shrink image natively to exactly 600px width so it fits columns intrinsically without CSS scaling breaking coordinates
        target_width = 600
        h, w = img_rgb.shape[:2]
        if w > target_width:
            scale = target_width / w
            target_height = int(h * scale)
            img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
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
        # Scale the coordinates back to the original image dimensions
        cx = int(clicked_value['x'] / scale)
        cy = int(clicked_value['y'] / scale)
        
        action_taken = False
        
        # Check if clicked a close button
        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            tr_x, tr_y = x + half - btn_size, y - half
            # Include invisible padding to keep it easy to click!
            if (tr_x - padding) <= cx <= (tr_x + btn_size + padding) and (tr_y - padding) <= cy <= (tr_y + btn_size + padding):
                st.session_state.found_centers.pop(i)
                action_taken = True
                st.rerun()
                break
                
        # If not close button, check if it's completely outside to add a new box
        if not action_taken:
            inside_any = False
            for y, x in st.session_state.found_centers:
                if x - half <= cx <= x + half and y - half <= cy <= y + half:
                    inside_any = True
                    break
                    
            if not inside_any:
                st.session_state.found_centers.append((cy, cx))
                st.rerun()
                
    st.markdown("---")
    if st.button(f"2. Extract & Save {len(st.session_state.found_centers)} Selected Crops", type="primary"):
        with st.spinner("Slicing hyperstack and saving TIF crops..."):
            extractor = CellposeRoiExtractor(eps=60)
            extractor.extract_and_save(process_path, out_dir, data=data, axes=axes, centers=st.session_state.found_centers)
            st.success(f"Successfully saved {len(st.session_state.found_centers)} crops to `{os.path.abspath(out_dir)}`!")
            
            # Cleanup temp file if uploaded
            if not isinstance(source, str) and os.path.exists("uploaded_temp.tif"):
                os.remove("uploaded_temp.tif")
                
            st.session_state.extraction_complete = True
            st.rerun()

elif st.session_state.get('extraction_complete', False):
    st.markdown("### 3. Filament Tracking Analysis")
    if st.button("← Go Back to Interactive ROI Editor"):
        st.session_state.extraction_complete = False
        st.rerun()
        
    base_name = os.path.splitext(os.path.basename(process_path))[0]
    out_dir = f"{base_name}_extracted_crops"
    
    # 1. Display Static Preview of the secondary channel with green boxes
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
        
    half = 128 // 2 # Match cellpose crop size
    import os
    for i, (y, x) in enumerate(st.session_state.found_centers):
        if i >= len(os.listdir(out_dir)): # Safeguard if crops are missing
            pass
        y, x = int(y), int(x)
        # Highlight selected box in thick yellow
        color = (255, 255, 0) if i == st.session_state.selected_roi_index else (0, 255, 0)
        thickness = 4 if i == st.session_state.selected_roi_index else 2

        cv2.rectangle(img_rgb, (x - half, y - half), (x + half, y + half), color, thickness)
        cv2.putText(img_rgb, str(i + 1), (x - half + 5, y - half + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    st.write("**Click on any green box in the image below to select it for tracking analysis.**")
    
    target_width = 800
    h, w = img_rgb.shape[:2]
    if w > target_width:
        scale_p3 = target_width / w
        target_height = int(h * scale_p3)
        img_resized_p3 = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        scale_p3 = 1.0
        img_resized_p3 = img_rgb

    # To ensure it resets click state when switching selections rapidly, add selection index to key
    val_p3 = streamlit_image_coordinates(img_resized_p3, key=f"roi_select_p3_{st.session_state.selected_roi_index}")
    
    if val_p3 is not None:
        cx_p3 = int(val_p3['x'] / scale_p3)
        cy_p3 = int(val_p3['y'] / scale_p3)
        
        for i, (y, x) in enumerate(st.session_state.found_centers):
            y, x = int(y), int(x)
            if x - half <= cx_p3 <= x + half and y - half <= cy_p3 <= y + half:
                if st.session_state.selected_roi_index != i:
                    st.session_state.selected_roi_index = i
                    st.rerun()
                break
    
    # 2. Import Ridge Enhancement
    import sys
    ridge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ridge_Enhancement")
    if ridge_dir not in sys.path:
        sys.path.append(ridge_dir)
        
    try:
        import Image as RImage
    except Exception as e:
        st.error(f"Could not load Ridge Enhancement module: {e}")
        st.stop()
        
    st.markdown("---")
    st.subheader("Analyze ROI Crop")
    
    import glob
    crops = sorted(glob.glob(os.path.join(out_dir, "*.tif")))
    
    if not crops:
        st.warning("No extracted crops found in the output directory.")
    elif st.session_state.selected_roi_index >= len(crops):
        st.warning("Selected ROI is out of bounds. Please re-extract.")
    else:
        st.info(f"**Currently Selected for Analysis:** ROI {st.session_state.selected_roi_index + 1}")
        selected_crop_path = crops[st.session_state.selected_roi_index]
        
        import tifffile
        crop_stack = tifffile.imread(selected_crop_path)
        # Convert to 2D time series (T, Y, X) for filament tracking
        if len(crop_stack.shape) == 5: # T, Z, C, Y, X
            other_channel = 1 if crop_stack.shape[2] > 1 else 0
            stack_2d_timeseries = np.max(crop_stack[:, :, other_channel, :, :], axis=1) 
        else:
            stack_2d_timeseries = crop_stack
            
        frames_count = stack_2d_timeseries.shape[0]
        
        with st.expander("Tracking Tuning Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                bright_filaments = st.checkbox("Bright Filaments", value=True)
                sigmas_input = st.text_input("Sigmas (comma-separated)", value="1, 1.5, 2, 2.5")
                low_q = st.slider("Low Quantile (low_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
                high_q = st.slider("High Quantile (high_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
            with col2:
                background_radius = st.slider("Background Radius", 1, 100, 15, 1)
                remove_objects_leq = st.number_input("Remove objects ≤ (px)", value=20, min_value=0)
                remove_holes_leq = st.number_input("Remove holes ≤ (px)", value=10, min_value=0)
                min_branch_length_um = st.number_input("Min Branch Length (μm)", value=0.200, min_value=0.0, format="%.3f", step=0.001)
            
            pixel_size_um = 0.184
            mins_per_frame = 15.0
            
            try:
                sigmas = tuple(float(x.strip()) for x in sigmas_input.split(","))
            except:
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
                pixel_size_um=pixel_size_um
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
                st.button("Pause II" if st.session_state.p3_playing else "Play ►", on_click=toggle_play_p3, key="btn_play_p3")
            else:
                st.write("**Single Frame**")
        with col_slide:
            if frames_count > 1:
                user_frame = st.slider("Frame Number", 1, frames_count, st.session_state.p3_frame)
                st.session_state.p3_frame = user_frame
            
        frame_idx = st.session_state.p3_frame - 1
        # Handle cases where user reduces frames and slider crashes out of bounds
        if frame_idx >= frames_count: 
            frame_idx = 0
            st.session_state.p3_frame = 1
            
        preview_frame = stack_2d_timeseries[frame_idx]
        pre = RImage.preprocess(preview_frame, params)
        ridge = RImage.ridge_enhance(pre, params)
        mask = RImage.segment_ridges(ridge, params)
        skeleton = RImage.morphology.skeletonize(mask)
        rows = RImage.extract_segments(skeleton, frame_idx, params)
        
        st.write(f"**Extracted segments in this frame:** {len(rows)}")
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame([{
                "Segment ID": r["segment_id"], 
                "Length (px)": round(r["length_px"], 2), 
                "Length (μm)": round(r["length_um"], 3)
            } for r in rows]), use_container_width=True)
        
        prev1, prev2, prev3, prev4 = st.columns(4)
        with prev1:
            st.write("**Original**")
            st.image(preview_frame, clamp=True, use_container_width=True)
        with prev2:
            st.write("**Preprocessed**")
            st.image(pre, clamp=True, use_container_width=True)
        with prev3:
            st.write("**Ridge Enhance**")
            st.image(ridge, clamp=True, use_container_width=True)
        with prev4:
            st.write("**Overlay**")
            rgb = np.stack([pre, pre, pre], axis=-1)  
            rgb[skeleton > 0] = [1.0, 0.0, 0.0]  # RED 
            st.image(rgb, clamp=True, use_container_width=True)
            
        st.markdown("---")
            
        if st.button("Run Full Tracking Analysis on Crop", type="primary"):
            with st.spinner("Processing hyperstack crop through Ridge Enhancement pipeline..."):
                try:
                    masks, skeletons, tracks = RImage.process_stack(stack_2d_timeseries, params)
                    
                    if not tracks.empty:
                        st.subheader("Tracking Results")
                        import pandas as pd
                        
                        active_frames = sorted(tracks["frame"].unique())
                        clusters = []
                        if active_frames:
                            start_f = active_frames[0]
                            prev_f = active_frames[0]
                            for f in active_frames[1:]:
                                if f == prev_f + 1:
                                    prev_f = f
                                else:
                                    clusters.append((start_f + 1, prev_f + 1))
                                    start_f = f
                                    prev_f = f
                            clusters.append((start_f + 1, prev_f + 1))
                            
                            cluster_df = pd.DataFrame(clusters, columns=["Start Frame", "End Frame"])
                            cluster_df["Duration (Frames)"] = cluster_df["End Frame"] - cluster_df["Start Frame"] + 1
                            cluster_df["Start Time (min)"] = (cluster_df["Start Frame"] - 1) * mins_per_frame
                            cluster_df["End Time (min)"] = (cluster_df["End Frame"] - 1) * mins_per_frame
                            cluster_df["Duration (min)"] = cluster_df["Duration (Frames)"] * mins_per_frame
                            
                            stable_clusters = cluster_df[cluster_df["Duration (Frames)"] > 2]
                            transient_clusters = cluster_df[cluster_df["Duration (Frames)"] <= 2]
                            
                            st.write("#### Stable Frame Clusters (> 2 frames)")
                            st.write(f"*Continuous periods where filaments persist stably. Total count: {len(stable_clusters)}*")
                            st.dataframe(stable_clusters, use_container_width=True)
                            
                            st.write("#### Transient Frame Clusters (1-2 frames)")
                            st.write(f"*Brief periods of activity that could be transient or noise. Total count: {len(transient_clusters)}*")
                            st.dataframe(transient_clusters, use_container_width=True)
                        
                    else:
                        st.warning("No filaments tracked across the stack. Try adjusting tuning parameters.")
                        
                except Exception as e:
                    st.error(f"Error during tracking: {e}")
                    
        # Placed at the very end of the script to prevent Streamlit React DOM duplication bugs on early abort
        if st.session_state.get('p3_playing', False) and frames_count > 1:
            import time
            time.sleep(0.3)
            st.session_state.p3_frame += 1
            if st.session_state.p3_frame > frames_count:
                st.session_state.p3_frame = 1
            st.rerun()
