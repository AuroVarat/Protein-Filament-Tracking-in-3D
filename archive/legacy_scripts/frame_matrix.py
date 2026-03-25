import tifffile
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop1.tif"
    print(f"Loading {file_path}...")
    
    try:
        image = tifffile.imread(file_path).astype(np.float32)
        print(f"Original shape: {image.shape}")
        
        if len(image.shape) <= 2:
            print("Image does not have multiple frames to subtract.")
            return

        num_frames = image.shape[0]
        
        print("Performing per-frame normalization (0-1)...")
        normalized_image = np.zeros_like(image)
        for i in range(num_frames):
            frame_min = image[i].min()
            frame_max = image[i].max()
            if frame_max > frame_min:
                normalized_image[i] = (image[i] - frame_min) / (frame_max - frame_min)
            else:
                normalized_image[i] = 0.0
                
        print("Calculating frame-by-frame difference matrix...")
        diff_matrix = np.zeros((num_frames, num_frames), dtype=np.float32)
        
        # Calculate the mean absolute difference between every pair of frames
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                diff_val = np.mean(np.abs(normalized_image[i] - normalized_image[j]))
                diff_matrix[i, j] = diff_val
                diff_matrix[j, i] = diff_val

        # Setup figure with 2 subplots (Matrix on top, Line plot on bottom)
        fig, (ax_matrix, ax_line) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot the 2D Difference Matrix
        img_display = ax_matrix.imshow(diff_matrix, cmap='magma', origin='lower')
        ax_matrix.set_title("N x N Frame Difference Matrix\n(Click anywhere on the matrix to see the 1D profile)")
        ax_matrix.set_xlabel("Frame Index (X)")
        ax_matrix.set_ylabel("Frame Index (Y)")
        
        cbar = fig.colorbar(img_display, ax=ax_matrix)
        cbar.set_label('Mean Absolute Difference')
        
        # Setup the 1D line plot (Default starts by showing Frame 0's profile)
        line_plot, = ax_line.plot(range(num_frames), diff_matrix[0], color='blue')
        ax_line.set_xlim(0, num_frames - 1)
        ax_line.set_ylim(0, np.max(diff_matrix) * 1.05)
        ax_line.set_xlabel("Frame Index")
        ax_line.set_ylabel("Difference")
        ax_line.set_title("Difference Profile for Frame 0")
        ax_line.grid(True)
        
        # Add visual markers (crosshairs) on the matrix to show which frame is active
        hline = ax_matrix.axhline(0, color='cyan', linestyle='--', alpha=0.8)
        vline = ax_matrix.axvline(0, color='cyan', linestyle='--', alpha=0.8)

        def on_click(event):
            # If the user clicks inside the Matrix plot
            if event.inaxes == ax_matrix:
                # Get the frame they clicked on the Y axis
                selected_frame = int(round(event.ydata))
                
                # Check bounds
                if 0 <= selected_frame < num_frames:
                    # Update the line plot
                    line_plot.set_ydata(diff_matrix[selected_frame])
                    ax_line.set_title(f"Difference Profile for Frame {selected_frame} vs All Other Frames")
                    
                    # Move the crosshairs
                    hline.set_ydata(selected_frame)
                    vline.set_xdata(selected_frame)
                    
                    fig.canvas.draw_idle()

        # Wire up the click event
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        plt.tight_layout()
        print("Showing interactive difference matrix. Close it to exit.")
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
