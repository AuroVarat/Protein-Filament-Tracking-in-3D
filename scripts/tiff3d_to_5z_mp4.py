import tifffile
import numpy as np
import imageio
import sys
import os

def load_data(filepath, channel=0):
    print(f"Loading {filepath} (channel {channel})...")
    img = tifffile.imread(filepath).astype(np.float32)
    
    # Normalize to (T, Z, H, W)
    if img.ndim == 3:
        img = img[np.newaxis, ...]  # (1, Z, H, W)
    elif img.ndim == 4:
        # Check if it's (T, Z, H, W)
        pass
    elif img.ndim == 5:
        # Assume (T, Z, C, H, W), take specified channel
        if channel < img.shape[2]:
            img = img[:, :, channel, :, :]
        else:
            print(f"Warning: Channel {channel} not found. Defaulting to channel 0.")
            img = img[:, :, 0, :, :]
    
    T, Z, H, W = img.shape
    # Normalize per Z-slice for maximum per-frame visibility
    norm = np.zeros_like(img)
    for t in range(T):
        for z in range(Z):
            mn, mx = img[t, z].min(), img[t, z].max()
            if mx > mn:
                norm[t, z] = (img[t, z] - mn) / (mx - mn)
    return norm

def create_5z_frame(vol):
    Z, H, W = vol.shape
    # Concatenate all Z planes horizontally
    planes = [vol[z] for z in range(Z)]
    frame = np.concatenate(planes, axis=1) # (H, Z*W)
    
    # Convert to 8-bit RGB
    frame_8bit = (frame * 255.0).astype(np.uint8)
    return np.stack([frame_8bit] * 3, axis=-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tiff3d_to_5z_mp4.py <input.tif> [channel] [output_dir]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    channel = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "tiffs3d_mp4"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_ch{channel}_5z.mp4")
    
    data = load_data(input_file, channel=channel)
    T, Z, H, W = data.shape
    
    print(f"Generating {Z}-Z-plane view for {T} timepoints...")
    frames = []
    for t in range(T):
        frame = create_5z_frame(data[t])
        frames.append(frame)
        
    print(f"Saving to {output_path}...")
    imageio.mimwrite(output_path, frames, fps=10, quality=9)
    print("Done!")

if __name__ == "__main__":
    main()
