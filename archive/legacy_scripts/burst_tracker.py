import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import sys

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    # avoid divide by zero gently
    with np.errstate(divide='ignore', invalid='ignore'):
        radialprofile = np.nan_to_num(tbin / nr)
    return radialprofile

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop4.tif"
    print(f"Loading {file_path} for Burst Tracking...")
    
    try:
        image = tifffile.imread(file_path).astype(np.float32)
        num_frames, h, w = image.shape
        
        # Physical constants
        pixel_size_um = 0.183
        
        center = (w//2, h//2)
        max_radius = min(center[0], center[1])
        q_um_inv = np.arange(0, max_radius) * (2 * np.pi) / (w * pixel_size_um)
        
        # We look exclusively at consecutive frame differences (dt=1)
        # For every single frame transition, diffs[t] = frame[t+1] - frame[t]
        print("Extracting spatial bursting frequencies across the timeline...")
        diffs = image[1:] - image[:-1]
        
        # Array to store the instantaneous Power Spectrum at every time point T
        # timeline_power[T, q] tells us how much activity happened at size q between Frame T and Frame T+1
        timeline_power = np.zeros((num_frames - 1, max_radius))
        
        for t in range(num_frames - 1):
            diff = diffs[t]
            # Hanning window prevents boundary glitches from mimicking high-frequency noise
            window = np.outer(np.hanning(h), np.hanning(w))
            f_transform = fftpack.fftshift(fftpack.fft2(diff * window))
            power_spectrum = np.abs(f_transform)**2
            
            radial_avg = radial_profile(power_spectrum, center)
            timeline_power[t, :] = radial_avg[:max_radius]
            
        print("Done! Launching Time-Resolved Spectrogram...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top Plot: Heatmap of all sizes vs Absolute Time
        extent = [0, num_frames - 1, q_um_inv[0], q_um_inv[-1]]
        im = axes[0].imshow(np.log10(timeline_power.T + 1e-5), aspect='auto', cmap='magma', origin='lower', extent=extent)
        axes[0].set_title("Time-Resolved Spectrogram of Filament Bursting Activity (dt=1)")
        axes[0].set_xlabel("Time (Frame Transition $T \\rightarrow T+1$)")
        axes[0].set_ylabel("Wavevector $q$ ($\mu m^{-1}$) [Higher = Smaller objects]")
        fig.colorbar(im, ax=axes[0], label="$\log_{10}$ Activity Intensity")
        
        # Lower plot: Timeline of specific size categories
        # The user mentioned filaments are 3-8 microns long. 
        # We target q-indices corresponding roughly to 8um, 4.7um, 3.3um, and 2.3um to capture both length and width.
        q_indices = [3, 5, 7, 10] 
        for q_idx in q_indices:
            if q_idx < max_radius:
                real_q = q_um_inv[q_idx]
                real_length_um = (2 * np.pi) / real_q if real_q > 0 else float('inf')
                axes[1].plot(range(num_frames - 1), timeline_power[:, q_idx], 
                             label=f"Size ~ {real_length_um:.1f} $\mu m$ (q={real_q:.2f})")
                
        axes[1].set_title("Activity Timeline for Specific Object Sizes\n(Sharp peaks indicate exact frames where objects of that size burst/moved)")
        axes[1].set_xlabel("Time (Frame Transition $T \\rightarrow T+1$)")
        axes[1].set_ylabel("Activity Intensity (Arbitrary Units)")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
