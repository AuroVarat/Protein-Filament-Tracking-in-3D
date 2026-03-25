import tifffile
import numpy as np
import imageio
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python tif_to_mp4.py <input.tif> [fps]")
        print("Example: python tif_to_mp4.py tifs/ch20_URA7_URA8_002-crop4.tif 10")
        sys.exit(1)
        
    input_file = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
        
    output_file = os.path.splitext(input_file)[0] + ".mp4"
    
    print(f"Loading '{input_file}'...")
    image = tifffile.imread(input_file)
    
    print(f"Normalizing intensity for video encoding...")
    global_min = image.min()
    global_max = image.max()
    
    if global_max > global_min:
        normalized = (image - global_min) / (global_max - global_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)
        
    frames_8bit = (normalized * 255.0).astype(np.uint8)
    
    # Convert grayscale to RGB for maximum MP4 compatability across players
    frames_rgb = np.stack((frames_8bit,)*3, axis=-1)
    
    print(f"Writing to '{output_file}' at {fps} fps...")
    
    # imageio uses imageio-ffmpeg under the hood
    imageio.mimwrite(output_file, frames_rgb, fps=fps)
    
    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    main()
