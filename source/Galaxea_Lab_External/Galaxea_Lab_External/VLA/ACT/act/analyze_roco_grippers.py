
import os
import glob
import h5py
import numpy as np
import argparse
from tqdm import tqdm

def analyze_grippers():
    parser = argparse.ArgumentParser(description="Analyze ROCO HDF5 gripper values.")
    parser.add_argument("--src", type=str, required=True, help="Root directory of source dataset")
    args = parser.parse_args()

    print(f"Scanning for HDF5 files in {args.src}...")
    files = glob.glob(os.path.join(args.src, "**", "*.hdf5"), recursive=True)
    files.extend(glob.glob(os.path.join(args.src, "**", "*.h5"), recursive=True))
    files.sort()
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files. Analyzing...")

    min_l_vals = []
    min_r_vals = []
    closed_count_l = 0
    closed_count_r = 0
    threshold = 0.01  # Consider closed if < 1cm

    for file_path in tqdm(files):
        try:
            with h5py.File(file_path, 'r') as f:
                l_grip = f['observations/left_gripper_joint_pos'][:]
                r_grip = f['observations/right_gripper_joint_pos'][:]
                
                l_min = l_grip.min()
                r_min = r_grip.min()
                
                min_l_vals.append(l_min)
                min_r_vals.append(r_min)
                
                if l_min < threshold:
                    closed_count_l += 1
                if r_min < threshold:
                    closed_count_r += 1
                    
                # print(f"{os.path.basename(file_path)}: L_Min={l_min:.4f}, R_Min={r_min:.4f}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print("\n--- Summary ---")
    print(f"Total Files: {len(files)}")
    print(f"Files with Left Gripper Closing (< {threshold}): {closed_count_l}")
    print(f"Files with Right Gripper Closing (< {threshold}): {closed_count_r}")
    print(f"Overall Min Left Gripper: {min(min_l_vals):.4f}")
    print(f"Overall Min Right Gripper: {min(min_r_vals):.4f}")
    print(f"Average Min Left Gripper: {np.mean(min_l_vals):.4f}")
    print(f"Average Min Right Gripper: {np.mean(min_r_vals):.4f}")

if __name__ == "__main__":
    analyze_grippers()
