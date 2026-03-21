import streamlit as st
import numpy as np
import Image

st.set_page_config(layout="wide", page_title="Filament Tracker Dashboard")

st.title("Filament Tracking Parameter Tuning")

# --- Sidebar Inputs ---
st.sidebar.header("Data Loading")
uploaded_file = st.sidebar.file_uploader("Upload TIF File", type=["tif", "tiff"])

if uploaded_file is None:
    st.info("Please upload a TIF file to begin.")
    st.stop()

@st.cache_data
def load_data(path):
    return Image.load_stack(path)

try:
    stack = load_data(uploaded_file)
    frames_count = stack.shape[0]
except Exception as e:
    st.error(f"Failed to load image: {e}")
    st.stop()

if "frame_number" not in st.session_state:
    st.session_state.frame_number = 1
if "playing" not in st.session_state:
    st.session_state.playing = False

def toggle_play():
    st.session_state.playing = not st.session_state.playing

st.sidebar.button("Pause II" if st.session_state.playing else "Play ►", on_click=toggle_play)

def sync_slider():
    st.session_state.frame_number = st.session_state.frame_slider

def sync_input():
    st.session_state.frame_number = st.session_state.frame_input

# Sync the visual widgets to our true frame_number BEFORE they render
st.session_state.frame_slider = st.session_state.frame_number
st.session_state.frame_input = st.session_state.frame_number

st.sidebar.slider("Frame Number (Slider)", 1, frames_count, key="frame_slider", on_change=sync_slider)
st.sidebar.number_input("Frame Number (Text)", 1, frames_count, key="frame_input", on_change=sync_input)

frame_number = st.session_state.frame_number
frame_idx = frame_number - 1

st.sidebar.header("Extraction Parameters")
bright_filaments = st.sidebar.checkbox("Bright Filaments", value=True)
sigmas_input = st.sidebar.text_input("Sigmas (comma-separated)", value="1, 1.5, 2, 2.5")
try:
    sigmas = tuple(float(x.strip()) for x in sigmas_input.split(","))
except:
    st.sidebar.error("Invalid sigmas format")
    sigmas = (1.5,)

low_q = st.sidebar.slider("Low Quantile (low_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
high_q = st.sidebar.slider("High Quantile (high_q)", 0.000, 1.000, 0.997, 0.001, format="%.3f")
background_radius = st.sidebar.slider("Background Radius", 1, 100, 15, 1)

remove_objects_leq = st.sidebar.number_input("Remove objects ≤ (px)", value=20, min_value=0)
remove_holes_leq = st.sidebar.number_input("Remove holes ≤ (px)", value=10, min_value=0)
min_branch_length_um = st.sidebar.number_input("Min Branch Length (μm)", value=0.200, min_value=0.0, format="%.3f", step=0.001)
pixel_size_um = 0.184
mins_per_frame = 15.0

min_branch_length_px = min_branch_length_um / pixel_size_um if pixel_size_um > 0 else 0.0

params = Image.Params(
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

# --- Process Frame ---
frame = stack[frame_idx]
pre = Image.preprocess(frame, params)
ridge = Image.ridge_enhance(pre, params)
mask = Image.segment_ridges(ridge, params)
skeleton = Image.morphology.skeletonize(mask)
rows = Image.extract_segments(skeleton, frame_idx, params)

# --- Visualization ---
st.write(f"**Extracted segments in this frame:** {len(rows)}")
if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame([{
        "Segment ID": r["segment_id"], 
        "Length (px)": round(r["length_px"], 2), 
        "Length (μm)": round(r["length_um"], 3)
    } for r in rows]))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Original (Raw)")
    st.image(frame, clamp=True, use_container_width=True)

with col2:
    st.subheader("Preprocessed")
    st.image(pre, clamp=True, use_container_width=True)

with col3:
    st.subheader("Ridge Enhancement")
    st.image(ridge, clamp=True, use_container_width=True)

with col4:
    st.subheader("Overlay")
    # Colorize the skeleton in red over the processed original
    rgb = np.stack([pre, pre, pre], axis=-1)  
    rgb[skeleton > 0] = [1.0, 0.0, 0.0]  # RED skeleton
    st.image(rgb, clamp=True, use_container_width=True)

st.write("---")
st.subheader("Full Stack Tracking Analysis")
if st.button("Run Tracking on All Frames"):
    with st.spinner("Processing entire stack... this may take a moment."):
        masks, skeletons, tracks = Image.process_stack(stack, params)
        if not tracks.empty:
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
                
                import pandas as pd
                cluster_df = pd.DataFrame(clusters, columns=["Start Frame", "End Frame"])
                cluster_df["Duration (Frames)"] = cluster_df["End Frame"] - cluster_df["Start Frame"] + 1
                cluster_df["Start Time (min)"] = (cluster_df["Start Frame"] - 1) * mins_per_frame
                cluster_df["End Time (min)"] = (cluster_df["End Frame"] - 1) * mins_per_frame
                cluster_df["Duration (min)"] = cluster_df["Duration (Frames)"] * mins_per_frame
                
                stable_clusters = cluster_df[cluster_df["Duration (Frames)"] > 2]
                transient_clusters = cluster_df[cluster_df["Duration (Frames)"] <= 2]
                
                st.write("#### Stable Frame Clusters (> 2 frames)")
                st.write(f"*Continuous periods where filaments persist stably. Total count: {len(stable_clusters)}*")
                st.dataframe(stable_clusters)
                
                st.write("#### Transient Frame Clusters (1-2 frames)")
                st.write(f"*Brief periods of activity that could be transient or noise. Total count: {len(transient_clusters)}*")
                st.dataframe(transient_clusters)
            
            summary = (
                tracks.groupby("track_id", as_index=False)
                .agg(
                    start_frame=("frame", "min"),
                    end_frame=("frame", "max"),
                    duration=("frame", "size"),
                    mean_length_px=("length_px", "mean"),
                    mean_length_um=("length_um", "mean"),
                )
            )
            summary["start_frame"] += 1
            summary["end_frame"] += 1
            summary["start_time_min"] = (summary["start_frame"] - 1) * mins_per_frame
            summary["end_time_min"] = (summary["end_frame"] - 1) * mins_per_frame
            summary["duration_min"] = summary["duration"] * mins_per_frame
            
            st.success(f"Tracked {len(summary)} unique filaments across {frames_count} frames.")
            
            stable_filaments = summary[summary["duration"] > 2].sort_values("duration", ascending=False)
            transient_filaments = summary[summary["duration"] <= 2].sort_values("duration", ascending=False)
            
            st.write("#### Stable Filaments (> 2 frames)")
            st.write(f"*Most likely real filaments. Total count: {len(stable_filaments)}*")
            st.dataframe(stable_filaments)
            
            st.write("#### Transient Filaments (1-2 frames)")
            st.write(f"*Could be transient or noise. Total count: {len(transient_filaments)}*")
            st.dataframe(transient_filaments)
        else:
            st.warning("No filaments tracked across the stack.")

st.write("---")
st.write("**Current Parameter Configuration:**")
st.code(f"""
params = Image.Params(
    bright_filaments={bright_filaments},
    sigmas={sigmas},
    low_q={low_q},
    high_q={high_q},
    background_radius={background_radius},
    remove_objects_leq={remove_objects_leq},
    remove_holes_leq={remove_holes_leq},
    min_branch_length_px={min_branch_length_px},
    pixel_size_um={pixel_size_um}
)
""", language="python")

if st.session_state.playing:
    import time
    time.sleep(0.5)
    if st.session_state.frame_number < frames_count:
        st.session_state.frame_number += 1
    else:
        st.session_state.frame_number = 1
        
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
