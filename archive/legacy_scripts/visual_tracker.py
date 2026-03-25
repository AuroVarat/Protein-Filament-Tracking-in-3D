import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
import sys

def ridge_filter_single(img, sigma=1.5, beta=0.5, c_thresh=0.02):
    Hxx = ndimage.gaussian_filter(img, sigma, order=[2, 0])
    Hyy = ndimage.gaussian_filter(img, sigma, order=[0, 2])
    Hxy = ndimage.gaussian_filter(img, sigma, order=[1, 1])
    
    trace = Hxx + Hyy
    det_diff = np.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)
    
    l1 = 0.5 * (trace + det_diff)
    l2 = 0.5 * (trace - det_diff)
    
    mag_l1 = np.abs(l1)
    mag_l2 = np.abs(l2)
    
    lambda1 = np.where(mag_l1 > mag_l2, l1, l2) 
    lambda2 = np.where(mag_l1 > mag_l2, l2, l1) 
    
    l1_safe = np.where(lambda1 == 0, 1e-10, lambda1)
    Rb = np.abs(lambda2) / np.abs(l1_safe)
    
    S = np.sqrt(lambda1**2 + lambda2**2)
    frame_max_S = np.max(S)
    
    c = max(frame_max_S / 3.0, c_thresh)
    
    vesselness = np.exp(-(Rb**2) / (2 * beta**2)) * (1 - np.exp(-(S**2) / (2 * c**2)))
    vesselness[lambda1 > 0] = 0
    
    return vesselness

def filter_by_size(vesselness_map, length_target=22, width_target=3, tolerance=0.5):
    """
    Physically measures the dimensions of every glowing structure in the mask
    and destroys anything that doesn't match the target lengths natively.
    """
    # Binarize the map to extract blobs
    thresh = np.max(vesselness_map) * 0.2
    if thresh <= 0: return np.zeros_like(vesselness_map)
    binary_mask = vesselness_map > thresh
    
    # Label all distinct features that survived the ridge filter
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0: return np.zeros_like(vesselness_map)
    
    filtered_map = np.zeros_like(vesselness_map)
    
    for i in range(1, num_features + 1):
        # Extract pixel coordinates of this specific object
        y, x = np.where(labeled_mask == i)
        
        if len(y) < 3: # Ignore microscopic specks mathematically
            continue
            
        # Calculate 2D covariance of spatial coordinates to mathematically fit an ellipse bounds
        cov = np.cov(y, x)
        
        # Guard against zero variance elements
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)): continue
            
        eigenvalues, _ = np.linalg.eigh(cov)
        
        # Rule of thumb for the bounds of a spatial covariance distribution:
        # Length = 2 * (2 * std_dev) = 4 * sqrt(Variance)
        minor_axis = 4 * np.sqrt(np.max([np.min(eigenvalues), 0]) + 1e-10)
        major_axis = 4 * np.sqrt(np.max([np.max(eigenvalues), 0]) + 1e-10)
        
        # Strict matching rules
        length_match = (major_axis >= length_target * (1 - tolerance)) and (major_axis <= length_target * (1 + tolerance))
        width_match = (minor_axis >= width_target * (1 - tolerance)) and (minor_axis <= width_target * (1 + tolerance))
        
        # If it fits the size profile perfectly, paint it back onto the output map!
        if length_match and width_match:
            filtered_map[labeled_mask == i] = 1.0 # Or use vesselness intensity
            
    return filtered_map

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop4.tif"
    print(f"Loading {file_path} for Interactive Parameter Tuning...")
    
    try:
        image = tifffile.imread(file_path).astype(np.float32)
        num_frames = image.shape[0]
        
        print("Per-frame normalizing the video sequence...")
        normalized_video = np.zeros_like(image)
        for i in range(num_frames):
            f_min = image[i].min()
            f_max = image[i].max()
            if f_max > f_min:
                normalized_video[i] = (image[i] - f_min) / (f_max - f_min)
            else:
                normalized_video[i] = 0.0
                
        diffs = normalized_video[1:] - normalized_video[:-1]
        
        print("Launching Interactive Target Overlay Dashboard...")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))
        plt.subplots_adjust(bottom=0.35)
        fig.suptitle(f"Interactive Filament Size-Target Overlay: {file_path}", fontsize=14)
        
        # Left Panel (Col 1)
        img1 = ax1.imshow(image[1], cmap='gray', vmin=image[1].min(), vmax=image[1].max())
        ax1.set_title("Raw Video")
        ax1.axis('off')
        
        # Middle Left (Col 2)
        diff_max = np.percentile(np.abs(diffs), 99)
        if diff_max == 0: diff_max = 1
        img2 = ax2.imshow(np.abs(diffs[0]), cmap='magma', vmin=0, vmax=diff_max)
        ax2.set_title("Unfiltered Difference")
        ax2.axis('off')
        
        # Middle Right (Col 3)
        init_sigma = 1.5
        init_beta = 0.5
        init_c = 0.02
        
        initial_filtered = ridge_filter_single(np.abs(diffs[0]), init_sigma, init_beta, init_c)
        img3 = ax3.imshow(initial_filtered, cmap='inferno')
        img3.set_clim(vmin=0, vmax=np.max(initial_filtered) if np.max(initial_filtered) > 0 else 1.0)
        ax3.set_title("General Anisotropy")
        ax3.axis('off')
        
        # Right (Col 4) - Target Object Overlay specifically on Raw Frame
        img4_raw = ax4.imshow(image[1], cmap='gray', vmin=image[1].min(), vmax=image[1].max())
        
        # Find exact 22x3 objects structurally in the filtered array
        init_targets = filter_by_size(initial_filtered, length_target=22, width_target=3, tolerance=0.5)
        
        # Mask the target map to only display where objects exist (so the raw displays fine underneath)
        masked_targets = np.ma.masked_where(init_targets == 0, init_targets)
        img4_overlay = ax4.imshow(masked_targets, cmap='spring', vmin=0, vmax=1, alpha=0.6)
        
        ax4.set_title("Overlay Targeting Exactly 22x3px")
        ax4.axis('off')
        
        # Setup tuning controls
        ax_frame = plt.axes([0.15, 0.20, 0.7, 0.03])
        ax_sigma = plt.axes([0.15, 0.15, 0.7, 0.03])
        ax_beta  = plt.axes([0.15, 0.10, 0.7, 0.03])
        ax_c     = plt.axes([0.15, 0.05, 0.7, 0.03])
        
        slider_frame = Slider(ax_frame, 'Frame Time', 1, num_frames - 1, valinit=5, valstep=1, initcolor='none')
        slider_sigma = Slider(ax_sigma, 'Thickness Filt. (Sigma)', 0.1, 5.0, valinit=init_sigma, initcolor='none')
        slider_beta  = Slider(ax_beta,  'Roundness Rej. (Beta)', 0.01, 2.0, valinit=init_beta, initcolor='none')
        slider_c     = Slider(ax_c,     'Noise Floor (c_thresh)', 0.001, 0.1, valinit=init_c, initcolor='none')
        
        def update(val):
            t = int(slider_frame.val)
            s = slider_sigma.val
            b = slider_beta.val
            c = slider_c.val
            
            # 1. Update Raw view
            img1.set_data(image[t])
            img1.set_clim(vmin=image[t].min(), vmax=image[t].max())
            ax1.set_title(f"Raw Video Frame: {t}")
            
            # 2. Update Unfiltered diff
            img2.set_data(np.abs(diffs[t-1]))
            
            # 3. Update Isolated tracking (General straight-lines)
            filtered = ridge_filter_single(np.abs(diffs[t-1]), sigma=s, beta=b, c_thresh=c)
            f_max = np.max(filtered)
            if f_max == 0: f_max = 1.0
            img3.set_data(filtered)
            img3.set_clim(vmin=0, vmax=f_max)
            
            # 4. Update the Exact 22x3 Object Overlay
            img4_raw.set_data(image[t])
            img4_raw.set_clim(vmin=image[t].min(), vmax=image[t].max())
            
            targets = filter_by_size(filtered, length_target=22, width_target=3, tolerance=0.5)
            
            # Update opacity map using pure transparency (0 or 1) so it doesn't wash out raw img
            masked_targets = np.ma.masked_where(targets == 0, targets)
            img4_overlay.set_data(masked_targets)
            
            fig.canvas.draw_idle()
            
        slider_frame.on_changed(update)
        slider_sigma.on_changed(update)
        slider_beta.on_changed(update)
        slider_c.on_changed(update)
        
        plt.show()

    except Exception as e:
        print(f"Error processing visual tracking: {e}")

if __name__ == "__main__":
    main()
