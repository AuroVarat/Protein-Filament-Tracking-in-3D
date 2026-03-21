import sys 
sys.path.append('.')
import Image
from dataclasses import replace
import time

params = Image.Params(
    bright_filaments=True,
    sigmas=(1.5,),
    low_q=0.99,
    high_q=0.99,
    remove_objects_leq=20,
    remove_holes_leq=10,
    background_radius=15,
    min_branch_length_px=2.0,
    max_centroid_move=20.0,
    max_cost=50.0,
    pixel_size_um=0.184
)

print(f"Running with params: {params}")
t0 = time.time()
stack = Image.load_stack('tiff/ch20_URA7_URA8_001-crop1.tif')
masks, skeletons, tracks = Image.process_stack(stack, params)
Image.save_outputs('output_tuned', masks, skeletons, tracks)
t1 = time.time()
print(f"Time taken: {t1-t0:.2f}s")
print(f"Frames: {stack.shape[0]}")
print(f"Tracked segments: {0 if tracks.empty else tracks['track_id'].nunique()}")
