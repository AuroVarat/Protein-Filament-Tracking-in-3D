import os
import cv2
import numpy as np
import tifffile as tiff
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cellpose import models
from sklearn.cluster import DBSCAN

# Check for hardware acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    use_gpu = True
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu = True
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    use_gpu = False
    print("Using CPU")

class CellposeRoiExtractor:
    def __init__(self, crop_size=128, cluster_area=170, min_cells=3, max_cells=10, min_distance=100, eps=20):
        """
        crop_size: Size of output patches (e.g. 128x128).
        cluster_area: Max width/height of the bounding box of a cluster.
        min_cells: Minimum cells required in the cluster.
        max_cells: Maximum cells allowed in the cluster.
        min_distance: Min distance between different cluster centers.
        eps: DBSCAN distance threshold (max distance between cells in a cluster).
        """
        self.crop_size = crop_size
        self.cluster_area = cluster_area
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.min_distance = min_distance
        self.eps = eps
        
        self.min_diam = 15
        self.max_diam = 35
        
        # Initialize Cellpose model
        self.model = models.CellposeModel(model_type='cyto', gpu=use_gpu, device=device)

    def find_cell_rich_centers(self, hyperstack_5d, bf_channel=0, return_all=False):
        """
        Uses Cellpose to find cells and DBSCAN to find clusters.
        Returns ROI centers (y, x).
        """
        z_idx = hyperstack_5d.shape[1] // 2
        ref_2d = hyperstack_5d[0, z_idx, bf_channel, :, :]
        
        print("  Running Cellpose segmentation...")
        masks, flows, styles = self.model.eval(ref_2d, diameter=25, channels=[0,0])
        
        n_labels = np.max(masks)
        valid_cell_centers = []
        
        for i in range(1, n_labels + 1):
            mask_i = (masks == i)
            coords_y, coords_x = np.where(mask_i)
            if len(coords_y) == 0: continue
            
            area = len(coords_y)
            diam = 2 * np.sqrt(area / np.pi)
            
            if self.min_diam <= diam <= self.max_diam:
                center_y = np.mean(coords_y)
                center_x = np.mean(coords_x)
                valid_cell_centers.append((center_y, center_x))

        print(f"  Detected {len(valid_cell_centers)} cells matching size filter ({self.min_diam}-{self.max_diam}px).")
        
        if len(valid_cell_centers) < self.min_cells:
            if return_all: return [], [], [], ref_2d, masks
            return []

        # --- DBSCAN CLUSTERING ---
        coords = np.array(valid_cell_centers)
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_cells)
        labels = clusterer.fit_predict(coords)
        
        final_centers = []
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        for label in unique_labels:
            cluster_points = coords[labels == label]
            count = len(cluster_points)
            
            if not (self.min_cells <= count <= self.max_cells):
                continue
            
            min_y, min_x = np.min(cluster_points, axis=0)
            max_y, max_x = np.max(cluster_points, axis=0)
            if (max_y - min_y) > self.cluster_area or (max_x - min_x) > self.cluster_area:
                continue
                
            centroid_y, centroid_x = np.mean(cluster_points, axis=0)
            
            # Clamp to stay in image bounds for 128x128 crop
            half = self.crop_size // 2
            h, w = ref_2d.shape
            clamped_y = max(half, min(centroid_y, h - half))
            clamped_x = max(half, min(centroid_x, w - half))
            
            final_centers.append((clamped_y, clamped_x))

        if return_all:
            return final_centers, coords, labels, ref_2d, masks
        return final_centers

    def extract_crops(self, hyperstack_5d, bf_channel=0):
        """
        Finds clusters and returns a list of cropped 5D hyperstacks.
        This is intended for use in an automated pipeline.
        """
        centers = self.find_cell_rich_centers(hyperstack_5d, bf_channel=bf_channel)
        crops = []
        half = self.crop_size // 2

        for (y, x) in centers:
            y, x = int(round(y)), int(round(x))
            # Slice 5D: (T, Z, C, Y, X)
            crop = hyperstack_5d[:, :, :, y-half : y+half, x-half : x+half]
            crops.append(crop)
        
        return crops, centers

    def visualize_clusters(self, input_path, output_plot="cluster_visualization_dbscan.png"):
        print(f"Visualizing: {input_path}")
        with tiff.TiffFile(input_path) as tif:
            data = tif.asarray()
        
        z_idx = data.shape[1] // 2
        ref_bf = data[0, z_idx, 0, :, :]
        other_channel = 1 if data.shape[2] > 1 else 0
        ref_other = data[0, z_idx, other_channel, :, :]
        
        final_centers, coords, labels, _, masks = self.find_cell_rich_centers(data, return_all=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        
        for ax, img, title in zip([ax1, ax2], [ref_bf, ref_other], ["Brightfield", f"Channel {other_channel}"]):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=20)
            
            unique_m = np.unique(masks)
            for val in unique_m:
                if val == 0: continue
                m_i = (masks == val).astype(np.uint8)
                cont, _ = cv2.findContours(m_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cont:
                    ax.plot(c[:, 0, 0], c[:, 0, 1], 'y-', linewidth=0.5, alpha=0.3)

            if len(coords) > 0:
                ax.scatter(coords[labels == -1, 1], coords[labels == -1, 0], c='white', s=15, alpha=0.4, label='Noise')
                ax.scatter(coords[labels != -1, 1], coords[labels != -1, 0], c=labels[labels != -1], cmap='tab20', s=50, alpha=0.8, edgecolors='black', label='Clusters')
            
            if final_centers:
                f_centers = np.array(final_centers)
                ax.scatter(f_centers[:, 1], f_centers[:, 0], c='red', marker='x', s=100, label='ROI Center')
                
                half = self.crop_size // 2
                for y, x in final_centers:
                    rect = patches.Rectangle((x-half, y-half), self.crop_size, self.crop_size, 
                                             linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.4)
                    ax.add_patch(rect)
            
            ax.axis('off')

        plt.suptitle(f"DBSCAN Analysis (eps={self.eps}) - {len(final_centers)} ROIs - {os.path.basename(input_path)}", fontsize=24)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot saved to: {output_plot}")

    def extract_and_save(self, input_path, output_dir):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        with tiff.TiffFile(input_path) as tif:
            data = tif.asarray()
            axes = tif.series[0].axes
        
        print(f"Processing: {input_path}")
        crops, centers = self.extract_crops(data)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        for i, crop in enumerate(crops):
            out_path = os.path.join(output_dir, f"{base_name}_dbscan_crop_{i+1:02d}.tif")
            tiff.imwrite(out_path, crop, imagej=True, metadata={'axes': axes})
            
        print(f"  Saved {len(crops)} crops to {output_dir}")

if __name__ == "__main__":
    extractor = CellposeRoiExtractor(eps=60) 
    input_file = "Biohackathon_hyperstacks/ch20_URA7_URA8_001_hyperstack.tif"
    if os.path.exists(input_file):
        extractor.visualize_clusters(input_file)
        extractor.extract_and_save(input_file, "dbscan_crops_gpu")
