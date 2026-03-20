import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def main():
    file_path = "ch20_URA7_URA8_002-crop1.tiff"
    print(f"Loading {file_path}...")
    
    try:
        # Load the image
        image = tifffile.imread(file_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        
        # Display the image
        if len(image.shape) > 2:
            num_frames = image.shape[0]
            img_display = ax.imshow(image[0], cmap='gray')
            ax.set_title(f"{file_path}")
            ax.axis('off')
            
            # Add a slider for frames
            ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
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
                img_display.set_data(image[frame_idx])
                fig.canvas.draw_idle()
                
            frame_slider.on_changed(update)
            
            print("Showing image window. Close the window to exit.")
            plt.show() # Keeps window open until user closes it
        else:
            ax.imshow(image, cmap='gray')
            ax.set_title(file_path)
            ax.axis('off')
            print("Showing image window. Close the window to exit.")
            plt.show()
            
    except Exception as e:
        print(f"Error loading or showing image: {e}")

if __name__ == "__main__":
    main()
