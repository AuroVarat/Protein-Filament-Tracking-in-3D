import tifffile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import sys

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop5.tif"
    print(f"Loading {file_path}...")
    
    try:
        # Load the image
        image = tifffile.imread(file_path).astype(np.float32)
        print(f"Image shape: {image.shape}")
        
        if len(image.shape) <= 2:
            print("Image does not have multiple frames to scrub through.")
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            ax.set_title(file_path)
            ax.axis('off')
            plt.show()
            return
            
        num_frames = image.shape[0]

        print("Calculating Normalizations...")
        
        # 1. Global Normalization (Min-Max over the entire video sequence)
        global_min = image.min()
        global_max = image.max()
        global_norm = image.copy()
        if global_max > global_min:
            global_norm = (image - global_min) / (global_max - global_min)
            
        # 2. Per-Frame Normalization (Min-Max for each individual frame)
        frame_norm = np.zeros_like(image)
        for i in range(num_frames):
            f_min = image[i].min()
            f_max = image[i].max()
            if f_max > f_min:
                frame_norm[i] = (image[i] - f_min) / (f_max - f_min)
            else:
                frame_norm[i] = 0.0

        print("Opening triple-view player...")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25)
        fig.suptitle(f"TIFF Viewer: {file_path}", fontsize=14)
        
        # Initialize left axis: Unmodified Original
        # We explicitly lock the vmin/vmax to Frame 0 to replicate the original behavior
        orig_min = image[0].min()
        orig_max = image[0].max()
        img_display1 = ax1.imshow(image[0], cmap='gray', vmin=orig_min, vmax=orig_max)
        ax1.set_title("Original (Raw)\n(Locked to Frame 0's brightness)")
        ax1.axis('off')

        # Initialize middle axis: Global Normalization
        img_display2 = ax2.imshow(global_norm[0], cmap='gray', vmin=0, vmax=1)
        ax2.set_title("Global Normalization\n(True brightness scaled 0-1)")
        ax2.axis('off')
        
        # Initialize right axis: Per-Frame Normalization
        img_display3 = ax3.imshow(frame_norm[0], cmap='gray', vmin=0, vmax=1)
        ax3.set_title("Per-Frame Normalization\n(Contrast boosted every frame)")
        ax3.axis('off')
        
        # Add a unified slider for frames
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        frame_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=num_frames - 1,
            valinit=0,
            valstep=1,
            initcolor='none'
        )
        
        def update(val):
            frame_idx = int(frame_slider.val)
            img_display1.set_data(image[frame_idx])
            img_display2.set_data(global_norm[frame_idx])
            img_display3.set_data(frame_norm[frame_idx])
            fig.canvas.draw_idle()
            
        frame_slider.on_changed(update)
        
        print("Showing image window. Close the window to exit.")
        plt.show() # Keeps window open until user closes it
            
    except Exception as e:
        print(f"Error loading or showing image: {e}")

if __name__ == "__main__":
    main()
