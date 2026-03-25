import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import fftpack
import sys

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop4.tif"
    print(f"Loading {file_path} for Anisotropic Filament Detection...")
    
    try:
        # Load the image
        image = tifffile.imread(file_path).astype(np.float32)
        num_frames, h, w = image.shape
        pixel_size_um = 0.183
        
        print("Per-frame normalizing the video before analysis...")
        normalized_video = np.zeros_like(image)
        for i in range(num_frames):
            f_min = image[i].min()
            f_max = image[i].max()
            if f_max > f_min:
                normalized_video[i] = (image[i] - f_min) / (f_max - f_min)
            else:
                normalized_video[i] = 0.0
                
        print("Calculating Anisotropic Signatures across the normalized video timeline...")
        diffs = normalized_video[1:] - normalized_video[:-1]
        
        # We want to isolate the 3-8 µm structures specifically.
        # Create a 2D mapping of exactly what physical size every pixel in the 2D FFT represents.
        y, x = np.indices((h, w))
        r_2d = np.sqrt((x - w/2)**2 + (y - h/2)**2)
        r_2d_safe = np.where(r_2d == 0, 1e-10, r_2d)
        q_2d = r_2d_safe * (2 * np.pi) / (w * pixel_size_um)
        
        size_2d = (2 * np.pi) / q_2d
        size_2d[r_2d == 0] = float('inf') # Center DC component is infinity
        
        # Create a boolean mask strictly selecting the 2D donut/ring for sizes 3.0 to 8.0 µm
        filament_mask_2d = (size_2d >= 3.0) & (size_2d <= 8.0)
        
        anisotropy_signature = np.zeros(num_frames - 1)
        
        for t in range(num_frames - 1):
            diff = diffs[t]
            # Hanning window prevents sharp edges from mimicking high-frequency noise
            window = np.outer(np.hanning(h), np.hanning(w))
            f_transform = fftpack.fftshift(fftpack.fft2(diff * window))
            power_spectrum = np.abs(f_transform)**2
            
            # Extract ONLY the power spectrum pixels lying precisely inside the 3-8µm ring.
            ring_pixels = power_spectrum[filament_mask_2d]
            
            # The Anisotropy Math:
            # - If structures are round/isotropic (like a cell), the power is scattered equally in all directions (low StdDev).
            # - If structures are straight lines/anisotropic (like a filament), the power heavily concentrates into a stark streak in one direction (massive StdDev).
            # We take the ratio of StdDev over the Mean to get a scale-independent "Lineness" score.
            mean_power = np.mean(ring_pixels)
            if mean_power > 0:
                anisotropy_score = np.std(ring_pixels) / mean_power
            else:
                anisotropy_score = 0.0
                
            anisotropy_signature[t] = anisotropy_score
        
        # Smooth the signature slightly since filaments persist over a few consecutive frames
        smoothed_signature = np.convolve(anisotropy_signature, np.ones(3)/3, mode='same')
        
        # Statistical Threshold (Mean + 1.5 * StdDev)
        baseline_mean = np.mean(smoothed_signature)
        baseline_std = np.std(smoothed_signature)
        threshold = baseline_mean + 1.5 * baseline_std
        
        is_bursting = smoothed_signature > threshold
        bursting_frames = np.where(is_bursting)[0] + 1
        
        print(f"\n======== AUTOMATED DETECTION RESULTS ========")
        if len(bursting_frames) > 0:
            print(f"Highly anisotropic structures localized specifically to frames: {bursting_frames}")
        else:
            print("No strongly anisotropic filament bursts detected.")
        print("=============================================\n")
            
        print("Plotting the interactive analytical result. Close the window to exit.")
        fig, (ax_plot, ax_img) = plt.subplots(1, 2, figsize=(16, 7))
        plt.subplots_adjust(bottom=0.20)
        fig.suptitle(f"Anisotropic Filament Detection: {file_path}", fontsize=14)
        
        time_axis = np.arange(1, num_frames)
        
        # LEFT SIDE: The Detection Signature
        ax_plot.plot(time_axis, smoothed_signature, label="Anisotropy Signature ('Lineness' Score)", color='#2ca02c', linewidth=2)
        ax_plot.axhline(threshold, color='red', linestyle='--', label=f"Detection Threshold (+1.5 StdDev)")
        ax_plot.fill_between(time_axis, 0, smoothed_signature, where=is_bursting, color='red', alpha=0.3, label="Detected Filament Activity")
        
        playhead_line = ax_plot.axvline(1, color='magenta', linewidth=2, label="Current Frame")
        
        ax_plot.set_title("Anisotropic Detection Algorithm (2D FFT Variance at 3-8 µm band)")
        ax_plot.set_xlabel("Time (Frame Number)")
        ax_plot.set_ylabel("Anisotropy Score\n(StdDev / Mean of Ring Power)")
        ax_plot.legend(loc='upper right')
        ax_plot.grid(True)
        
        # RIGHT SIDE: The Video Viewer
        img_display = ax_img.imshow(normalized_video[1], cmap='gray', vmin=0, vmax=1)
        ax_img.set_title("Video Frame: 1")
        ax_img.axis('off')
        
        # BOTTOM SLIDER
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
        frame_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=1,
            valmax=num_frames - 1,
            valinit=1,
            valstep=1,
            initcolor='none'
        )
        
        def update(val):
            frame_idx = int(frame_slider.val)
            img_display.set_data(normalized_video[frame_idx])
            ax_img.set_title(f"Video Frame: {frame_idx}")
            playhead_line.set_xdata([frame_idx, frame_idx])
            fig.canvas.draw_idle()
            
        frame_slider.on_changed(update)
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
