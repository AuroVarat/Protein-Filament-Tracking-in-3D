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
    radialprofile = tbin / nr
    return radialprofile 

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop4.tif"
    print(f"Loading {file_path} for DDM analysis...")
    
    try:
        image = tifffile.imread(file_path).astype(np.float32)
        num_frames, h, w = image.shape
        
        print(f"Image shape: {num_frames} frames, {h}x{w} pixels")

        # Range of lag times (delta t) to analyze
        # We'll use all possible delays up to max_dt
        max_dt = num_frames // 2
        dts = np.arange(1, max_dt)
        
        # Determine the maximum radius for radial average
        center = (w//2, h//2)
        max_radius = min(center[0], center[1])
        
        # Array to store the Image Structure Function (ISF) I(q, dt)
        I_q_dt = np.zeros((len(dts), max_radius))
        
        print("\nCalculating Differential Dynamic Microscopy (DDM) Structure Functions...")
        
        for idx, dt in enumerate(dts):
            print(f"\rProcessing lag time dt = {dt}/{max_dt-1}", end="")
            
            # Calculate all differences for this dt: I(x, t+dt) - I(x, t)
            diffs = image[dt:] - image[:-dt]
            
            # Array to accumulate the power spectra for this dt
            power_spectra = np.zeros((h, w))
            
            for diff in diffs:
                # 2D Fast Fourier Transform of the difference image
                # Apply a Hanning window to reduce edge artifacts
                window = np.outer(np.hanning(h), np.hanning(w))
                f_transform = fftpack.fftshift(fftpack.fft2(diff * window))
                
                # Power spectrum is the squared absolute value
                power_spectrum = np.abs(f_transform)**2
                power_spectra += power_spectrum
            
            # Average power spectrum over all frame pairs separated by dt
            avg_power_spectrum = power_spectra / len(diffs)
            
            # Compute the radial average (azimuthal average) to get I(q, dt)
            radial_avg = radial_profile(avg_power_spectrum, center)
            I_q_dt[idx, :] = radial_avg[:max_radius]
        
        print("\nDone calculating! Plotting results with physical size / frame units...")
        
        # Physical constants
        pixel_size_um = 0.183
        
        # Convert X-axis to physical units
        q_um_inv = np.arange(0, max_radius) * (2 * np.pi) / (w * pixel_size_um)
        
        # Plotting the Image Structure Function I(q, dt)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Heatmap contour of I(q, dt)
        im = axes[0].imshow(np.log10(I_q_dt + 1e-5), aspect='auto', cmap='viridis', origin='lower',
                            extent=[q_um_inv[0], q_um_inv[-1], dts[0], dts[-1]])
        axes[0].set_title("Image Structure Function $\log_{10}(I(q, \Delta t))$")
        axes[0].set_xlabel("Wavevector $q$ ($\mu m^{-1}$)")
        axes[0].set_ylabel("Lag Time $\Delta t$ (frames)")
        fig.colorbar(im, ax=axes[0], label="$\log_{10}$ Intensity")
        
        # 2. Extract dynamics for a specific length scale (wavevector q)
        q_indices = [2, 10, 25, 40]
        for q_idx in q_indices:
            if q_idx < max_radius:
                real_q = q_um_inv[q_idx]
                real_length_um = (2 * np.pi) / real_q if real_q > 0 else float('inf')
                axes[1].plot(dts, I_q_dt[:, q_idx], label=f"$q={real_q:.2f}$ (scale ~ {real_length_um:.1f} $\mu m$)")
                
        axes[1].set_title("Dynamics at Specific Spatial Scales")
        axes[1].set_xlabel("Lag Time $\Delta t$ (frames)")
        axes[1].set_ylabel("$I(q, \Delta t)$")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
