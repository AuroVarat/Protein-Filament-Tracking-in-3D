import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
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

def filter_by_size(vesselness_map, length_target=22, width_target=3, tolerance=0.5, conf_thresh=0.2):
    thresh = np.max(vesselness_map) * conf_thresh
    if thresh <= 0: return np.zeros_like(vesselness_map), []
    binary_mask = vesselness_map > thresh
    
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0: return np.zeros_like(vesselness_map), []
    
    filtered_map = np.zeros_like(vesselness_map)
    detected_objects = []
    
    object_id = 1
    for i in range(1, num_features + 1):
        y, x = np.where(labeled_mask == i)
        if len(y) < 3: continue
            
        cov = np.cov(y, x)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)): continue
            
        eigenvalues, _ = np.linalg.eigh(cov)
        
        minor_axis = 4 * np.sqrt(np.max([np.min(eigenvalues), 0]) + 1e-10)
        major_axis = 4 * np.sqrt(np.max([np.max(eigenvalues), 0]) + 1e-10)
        
        length_match = (major_axis >= length_target * (1 - tolerance)) and (major_axis <= length_target * (1 + tolerance))
        width_match = (minor_axis >= width_target * (1 - tolerance)) and (minor_axis <= width_target * (1 + tolerance))
        
        if length_match and width_match:
            filtered_map[labeled_mask == i] = object_id 
            detected_objects.append({
                'id': object_id,
                'length': major_axis,
                'width': minor_axis
            })
            object_id += 1
            
    return filtered_map, detected_objects

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "tifs/ch20_URA7_URA8_002-crop4.tif"
    print(f"Loading {file_path} for Direct Frame Tracking...")
    
    try:
        image = tifffile.imread(file_path).astype(np.float32)
        num_frames = image.shape[0]
        
        print("Globally normalizing the video sequence...")
        global_min = image.min()
        global_max = image.max()
        if global_max > global_min:
            normalized_video = (image - global_min) / (global_max - global_min)
        else:
            normalized_video = np.zeros_like(image)
                
        print("Launching Interactive Target Overlay Dashboard...")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        plt.subplots_adjust(bottom=0.60, right=0.85)
        fig.suptitle(f"Interactive Direct Frame Tracking: {file_path}", fontsize=14)
        
        # Left Panel (Col 1)
        img1 = ax1.imshow(normalized_video[0], cmap='gray', vmin=0, vmax=1)
        ax1.set_title("Globally Normalized Video")
        ax1.axis('off')
        
        # Middle Panel (Col 2)
        init_sigma = 1.5
        init_beta = 0.5
        init_c = 0.02
        init_vmin = 0.0
        init_vmax = 1.0
        init_blur = 0.0  # Default to no extra pre-blur
        
        filtered_img_init = np.copy(normalized_video[0])
        if init_blur > 0.01:
            filtered_img_init = ndimage.gaussian_filter(filtered_img_init, sigma=init_blur)
        filtered_img_init[filtered_img_init < init_vmin] = 0
        filtered_img_init[filtered_img_init > init_vmax] = 0
        initial_filtered = ridge_filter_single(filtered_img_init, init_sigma, init_beta, init_c)
        img2 = ax2.imshow(initial_filtered, cmap='inferno')
        img2.set_clim(vmin=0, vmax=np.max(initial_filtered) if np.max(initial_filtered) > 0 else 1.0)
        ax2.set_title("General Anisotropy\n(Calculated Direct on Frame)")
        ax2.axis('off')
        
        # Right (Col 3) - Target Object Overlay
        img3_raw = ax3.imshow(normalized_video[0], cmap='gray', vmin=0, vmax=1)
        init_targets, init_objects = filter_by_size(initial_filtered, length_target=22, width_target=3, tolerance=0.5)
        masked_targets = np.ma.masked_where(init_targets == 0, init_targets)
        
        cmap = plt.get_cmap('tab20')
        img3_overlay = ax3.imshow(masked_targets, cmap=cmap, vmin=1, vmax=20, alpha=0.9)
        
        # Initial Legend
        legend_handles = []
        for obj in init_objects:
            color = cmap((obj['id'] - 1) % 20)
            patch = mpatches.Patch(color=color, label=f"#{obj['id']} (L:{obj['length']:.1f}, W:{obj['width']:.1f})")
            legend_handles.append(patch)
        ax3.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Measured Dimensions")
        
        ax3.set_title("Overlay Targeting Exactly 22x3px\n(Direct on Frame)")
        ax3.axis('off')
        
        # Setup tuning controls
        ax_frame = plt.axes([0.15, 0.50, 0.7, 0.03])
        ax_blur  = plt.axes([0.15, 0.45, 0.7, 0.03])
        ax_sigma = plt.axes([0.15, 0.40, 0.7, 0.03])
        ax_beta  = plt.axes([0.15, 0.35, 0.7, 0.03])
        ax_c     = plt.axes([0.15, 0.30, 0.7, 0.03])
        ax_vmin  = plt.axes([0.15, 0.25, 0.7, 0.03])
        ax_vmax  = plt.axes([0.15, 0.20, 0.7, 0.03])
        ax_tol   = plt.axes([0.15, 0.15, 0.7, 0.03])
        ax_conf  = plt.axes([0.15, 0.10, 0.7, 0.03])
        
        slider_frame = Slider(ax_frame, 'Frame Time', 0, num_frames - 1, valinit=5, valstep=1, initcolor='none')
        slider_blur  = Slider(ax_blur,  'Pre-Blur Denoise', 0.0, 3.0, valinit=init_blur, initcolor='none')
        slider_sigma = Slider(ax_sigma, 'Thickness Filt. (Sigma)', 0.1, 5.0, valinit=init_sigma, initcolor='none')
        slider_beta  = Slider(ax_beta,  'Roundness Rej. (Beta)', 0.01, 2.0, valinit=init_beta, initcolor='none')
        slider_c     = Slider(ax_c,     'Noise Floor (c_thresh)', 0.001, 0.1, valinit=init_c, initcolor='none')
        slider_vmin  = Slider(ax_vmin,  'Intensity Min', 0.0, 1.0, valinit=init_vmin, initcolor='none')
        slider_vmax  = Slider(ax_vmax,  'Intensity Max', 0.0, 1.0, valinit=init_vmax, initcolor='none')
        slider_tol   = Slider(ax_tol,   'Size Tolerance', 0.1, 1.5, valinit=0.5, initcolor='none')
        slider_conf  = Slider(ax_conf,  'Confidence Thresh', 0.01, 0.99, valinit=0.2, initcolor='none')
        
        def update(val):
            t = int(slider_frame.val)
            blur_val = slider_blur.val
            s = slider_sigma.val
            b = slider_beta.val
            c = slider_c.val
            vmin_thresh = slider_vmin.val
            vmax_thresh = slider_vmax.val
            tol = slider_tol.val
            conf = slider_conf.val
            
            # 1. Update Globally Normalized view
            img1.set_data(normalized_video[t])
            img1.set_clim(vmin=0, vmax=1)
            ax1.set_title(f"Globally Normalized Video Frame: {t}")
            
            # 2. Update Isolated tracking 
            frame_data = np.copy(normalized_video[t])
            if blur_val > 0.01:
                frame_data = ndimage.gaussian_filter(frame_data, sigma=blur_val)
            
            frame_data[frame_data < vmin_thresh] = 0
            frame_data[frame_data > vmax_thresh] = 0
            
            filtered = ridge_filter_single(frame_data, sigma=s, beta=b, c_thresh=c)
            f_max = np.max(filtered)
            if f_max == 0: f_max = 1.0
            img2.set_data(filtered)
            img2.set_clim(vmin=0, vmax=f_max)
            
            # 3. Update the Exact 22x3 Object Overlay
            img3_raw.set_data(normalized_video[t])
            img3_raw.set_clim(vmin=0, vmax=1)
            
            targets, objects = filter_by_size(filtered, length_target=22, width_target=3, tolerance=tol, conf_thresh=conf)
            masked_targets = np.ma.masked_where(targets == 0, targets)
            img3_overlay.set_data(masked_targets)
            
            # Update the side legend
            legend_handles = []
            for obj in objects:
                color = cmap((obj['id'] - 1) % 20)
                patch = mpatches.Patch(color=color, label=f"#{obj['id']} (L:{obj['length']:.1f}, W:{obj['width']:.1f})")
                legend_handles.append(patch)
            
            ax3.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Measured Dimensions")
            
            fig.canvas.draw_idle()
            
        slider_frame.on_changed(update)
        slider_blur.on_changed(update)
        slider_sigma.on_changed(update)
        slider_beta.on_changed(update)
        slider_c.on_changed(update)
        slider_vmin.on_changed(update)
        slider_vmax.on_changed(update)
        slider_tol.on_changed(update)
        slider_conf.on_changed(update)
        
        plt.show()

    except Exception as e:
        print(f"Error processing visual tracking: {e}")

if __name__ == "__main__":
    main()
