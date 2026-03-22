import numpy as np
import os
import glob

results = glob.glob("results/*_analysis.npz")
if not results:
    print("No analysis files found.")
else:
    for f in results:
        print(f"\nChecking {f}...")
        try:
            data = np.load(f)
            labels = data['filament_labels']
            print(f"  Shape: {labels.shape}")
            unique = np.unique(labels)
            print(f"  Unique Labels: {unique}")
            if len(unique) <= 1:
                print("  WARNING: No tracked filaments found in labels!")
        except Exception as e:
            print(f"  Error: {e}")
