import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# File definition
FILE_PATH = "/home/hls/.cache/huggingface/hub/datasets--rocochallenge2025--rocochallenge2025/snapshots/76a5691156397e249cf9b8f568d37407c302d724/gearbox_assembly_demos_updated/100.hdf5"
OUTPUT_DIR = "/home/hls/codes/gearboxAssembly/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "depth_visualization.png")

def visualize_depth():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    try:
        with h5py.File(FILE_PATH, 'r') as f:
            # Get a frame from the middle of the episode
            num_frames = f['observations/head_rgb'].shape[0]
            idx = num_frames // 2
            print(f"Visualizing frame {idx}/{num_frames}")

            # Define keys to visualize
            cameras = [
                ("Head", "head_rgb", "head_depth"),
                ("Left Hand", "left_hand_rgb", "left_hand_depth"),
                ("Right Hand", "right_hand_rgb", "right_hand_depth")
            ]

            fig, axes = plt.subplots(len(cameras), 2, figsize=(12, 12))
            plt.subplots_adjust(hspace=0.3)

            for i, (name, rgb_key, depth_key) in enumerate(cameras):
                # RGB
                rgb = f[f'observations/{rgb_key}'][idx]
                
                # Depth
                depth = f[f'observations/{depth_key}'][idx]
                
                # Plot Depth - FOCUSED RANGE (0 to 1.5m)
                # This reveals the details on the table by ignoring the "far" background
                im = axes[i, 1].imshow(depth, cmap='magma', vmin=0, vmax=1.5)
                axes[i, 1].set_title(f"{name} Depth (0-1.5m range)")
                axes[i, 1].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

            plt.savefig(OUTPUT_FILE)
            print(f"Visualization saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    visualize_depth()
