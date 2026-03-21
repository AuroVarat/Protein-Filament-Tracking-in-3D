import numpy as np
import pandas as pd
import tifffile
import Image

def debug():
    p = Image.Params()
    p.low_q = 0.997
    p.high_q = 0.997
    p.remove_objects_leq = 20
    
    import glob
    files = glob.glob('tiff/*.tif')
    if not files:
        print("No tiff files found")
        return
    
    stack = Image.load_stack(files[0])
    print(f"Loaded {files[0]} shape {stack.shape}")
    
    if stack.shape[0] > 70:
        stack = stack[45:60]
    
    masks, skels, tracks = Image.process_stack(stack, p)
    print("Tracks found:")
    if tracks.empty:
        print("No tracks")
        return
    print(tracks.groupby('track_id')['frame'].agg(['min', 'max', 'count']))

if __name__ == '__main__':
    debug()
