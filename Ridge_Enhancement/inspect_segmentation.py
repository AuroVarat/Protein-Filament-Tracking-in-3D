import sys 
sys.path.append('.')
import Image

params = Image.Params(
    bright_filaments=True,
    sigmas=(2.0, 4.0, 6.0),
    low_q=0.50,          
    high_q=0.85,
    remove_objects_leq=10,
    remove_holes_leq=5,
    background_radius=30,
)

stack = Image.load_stack('tiff/ch20_URA7_URA8_001-crop1.tif')
frame = stack[50] # using middle frame

pre = Image.preprocess(frame, params)
print(f"Pre min/max: {pre.min():.3f}, {pre.max():.3f}")

ridge = Image.ridge_enhance(pre, params)
print(f"Ridge min/max: {ridge.min():.3f}, {ridge.max():.3f}")

mask = Image.segment_ridges(ridge, params)
print(f"Mask True count: {mask.sum()}")

skeleton = Image.morphology.skeletonize(mask)
print(f"Skeleton count: {skeleton.sum()}")

rows = Image.extract_segments(skeleton, 50, params)
print(f"Extracted segments in frame 50: {len(rows)}")
for r in rows:
    print(f" - Segment length: {r['length_px']:.2f}")
