import tifffile
import numpy as np
import imageio
import sys
import os

def load_data(filepath):
    print(f"Loading {filepath}...")
    img = tifffile.imread(filepath).astype(np.float32)
    
    # Normalize to (T, Z, H, W)
    if img.ndim == 3:
        img = img[np.newaxis, ...]  # (1, Z, H, W)
    elif img.ndim == 4:
        # Check if it's (Z, C, H, W) or (T, Z, H, W)
        # Based on filament_3d_viewer.py it assumes (T, Z, H, W)
        pass
    elif img.ndim == 5:
        # Assume (T, Z, C, H, W), take first channel
        img = img[:, :, 0, :, :]
    
    T, Z, H, W = img.shape
    # Normalize per timepoint
    norm = np.zeros_like(img)
    for t in range(T):
        mn, mx = img[t].min(), img[t].max()
        if mx > mn:
            norm[t] = (img[t] - mn) / (mx - mn)
    return norm

def create_orthogonal_frame(vol):
    Z, H, W = vol.shape
    
    # Extract middle slices
    xy = vol[Z // 2, :, :]
    xz = vol[:, H // 2, :]
    yz = vol[:, :, W // 2]
    
    # Rescale Z dimension for XZ and YZ views so they match H/W height
    # Z is usually small (e.g. 5), so we stretch it to H
    if Z < H:
        # Simple nearest neighbor stretch using numpy.repeat
        repeat_factor = H // Z
        xz_stretched = np.repeat(xz, repeat_factor, axis=0)
        yz_stretched = np.repeat(yz, repeat_factor, axis=0)
        
        # If not exact multiple, pad or crop
        if xz_stretched.shape[0] < H:
            pad = H - xz_stretched.shape[0]
            xz_stretched = np.pad(xz_stretched, ((0, pad), (0, 0)), mode='edge')
            yz_stretched = np.pad(yz_stretched, ((0, pad), (0, 0)), mode='edge')
        elif xz_stretched.shape[0] > H:
            xz_stretched = xz_stretched[:H, :]
            yz_stretched = yz_stretched[:H, :]
    else:
        xz_stretched = xz
        yz_stretched = yz

    # Ensure yz_stretched width matches H if H != W, but usually H == W
    # Concatenate horizontally: XY | XZ | YZ
    # xy: (H, W), xz_stretched: (H, W), yz_stretched: (H, H)
    frame = np.concatenate([xy, xz_stretched, yz_stretched], axis=1)
    
    # Convert to 8-bit RGB
    frame_8bit = (frame * 255.0).astype(np.uint8)
    return np.stack([frame_8bit] * 3, axis=-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tiff3d_to_3view_mp4.py <input.tif> [output_dir]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "tiffs3d_mp4"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_3view.mp4")
    
    data = load_data(input_file)
    T, Z, H, W = data.shape
    
    print(f"Generating orthogonal views for {T} timepoints...")
    frames = []
    for t in range(T):
        frame = create_orthogonal_frame(data[t])
        frames.append(frame)
        
    print(f"Saving to {output_path}...")
    imageio.mimwrite(output_path, frames, fps=10, quality=9)
    print("Done!")

if __name__ == "__main__":
    main()
