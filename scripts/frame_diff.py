import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def main():
    file_path = "ch20_URA7_URA8_002-crop1.tif"
    print(f"Loading {file_path}...")
    
    try:
        # Load the image as float to prevent underflow/overflow when subtracting
        image = tifffile.imread(file_path).astype(np.float32)
        print(f"Original shape: {image.shape}")
        
        if len(image.shape) <= 2:
            print("Image does not have multiple frames to subtract.")
            return

        # Per-frame min-max normalization down to 0-1 ranges
        print("Performing per-frame normalization (0-1)...")
        normalized_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            frame_min = image[i].min()
            frame_max = image[i].max()
            if frame_max > frame_min:
                normalized_image[i] = (image[i] - frame_min) / (frame_max - frame_min)
            else:
                normalized_image[i] = 0.0
                
        # Calculate absolute difference between consecutive normalized frames
        diffs = np.abs(np.diff(normalized_image, axis=0))
        num_diff_frames = diffs.shape[0]
        print(f"Calculated differences shape: {diffs.shape}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # We cap the vmax at the 99th percentile of all differences
        # to prevent extreme outliers from washing out the color scale
        vmax = np.percentile(diffs, 99)
        
        # Display the first difference frame as a heatmap
        img_display = ax.imshow(diffs[0], cmap='hot', vmin=0, vmax=vmax)
        ax.set_title("Frame Difference: 1 - 0")
        ax.axis('off')
        
        # Add a colorbar for reference
        cbar = fig.colorbar(img_display, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Intensity Difference')
        
        # Add a slider to scrub through difference frames
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        frame_slider = Slider(
            ax=ax_slider,
            label='Diff Frame',
            valmin=0,
            valmax=num_diff_frames - 1,
            valinit=0,
            valstep=1,
            initcolor='none'
        )
        
        def update(val):
            idx = int(frame_slider.val)
            img_display.set_data(diffs[idx])
            ax.set_title(f"Frame Difference: {idx+1} - {idx}")
            fig.canvas.draw_idle()
            
        frame_slider.on_changed(update)
        
        print("Showing difference heatmap window. Close it to exit.")
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
