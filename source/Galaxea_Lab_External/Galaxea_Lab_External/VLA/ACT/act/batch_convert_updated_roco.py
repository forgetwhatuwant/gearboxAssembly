
import h5py
import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse
from tqdm import tqdm
import glob
import os

def load_roco_updated_episode(file_path):
    """
    Reads a single ROCO HDF5 file (UPDATED version with valid actions).
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # --- OBSERVATIONS ---
            l_arm_pos = f['observations/left_arm_joint_pos'][:]
            l_grip_pos = f['observations/left_gripper_joint_pos'][:]
            r_arm_pos = f['observations/right_arm_joint_pos'][:]
            r_grip_pos = f['observations/right_gripper_joint_pos'][:]

            if l_grip_pos.ndim == 1: l_grip_pos = l_grip_pos[:, None]
            if r_grip_pos.ndim == 1: r_grip_pos = r_grip_pos[:, None]
            
            # Concatenate observations (L=14)
            states = np.concatenate([l_arm_pos, r_arm_pos, l_grip_pos, r_grip_pos], axis=1)

            # --- ACTIONS (REAL) ---
            # Reading from the 'actions' group - NO OFFSET applied
            act_l_arm = f['actions/left_arm_action'][:]
            act_l_grip = f['actions/left_gripper_action'][:]
            act_r_arm = f['actions/right_arm_action'][:]
            act_r_grip = f['actions/right_gripper_action'][:]

            if act_l_grip.ndim == 1: act_l_grip = act_l_grip[:, None]
            if act_r_grip.ndim == 1: act_r_grip = act_r_grip[:, None]

            # Concatenate actions (L=14)
            actions = np.concatenate([act_l_arm, act_r_arm, act_l_grip, act_r_grip], axis=1)

            # --- IMAGES ---
            head_rgb = f['observations/head_rgb'][:]
            left_hand_rgb = f['observations/left_hand_rgb'][:]
            right_hand_rgb = f['observations/right_hand_rgb'][:]

            length = actions.shape[0]
            # Simple validation
            if not (length == states.shape[0] == head_rgb.shape[0]):
                print(f"Skipping {file_path}: Shape mismatch {length} vs {states.shape[0]}")
                return None

            return {
                "action": actions,
                "observation.state": states,
                "observation.images.head": head_rgb,
                "observation.images.left_wrist": left_hand_rgb,
                "observation.images.right_wrist": right_hand_rgb
            }

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def batch_convert():
    parser = argparse.ArgumentParser(description="Batch Convert Updated ROCO Dataset")
    parser.add_argument("--raw_dir", type=str, 
                        default="/home/hls/.cache/huggingface/hub/datasets--rocochallenge2025--rocochallenge2025/snapshots/76a5691156397e249cf9b8f568d37407c302d724/gearbox_assembly_demos_updated",
                        help="Path to source HDF5 files")
    parser.add_argument("--out_dir", type=str, 
                        default="/media/hls/MK-ssd-mini/isaac_sim_data/lerobot_datasets/roco_gearbox_updated_no_offset",
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="galaxea/roco_gearbox_updated_no_offset", help="Repo ID")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    # Find files (support both .hdf5 and .h5 extensions)
    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.hdf5")))
    files.extend(sorted(glob.glob(os.path.join(args.raw_dir, "*.h5"))))
    print(f"Found {len(files)} HDF5 files in {args.raw_dir}")
    if len(files) == 0:
        print("No files found! Check path.")
        return

    # Clean start
    save_path = Path(args.out_dir)
    if save_path.exists():
        import shutil
        shutil.rmtree(save_path)
        print(f"Cleaned existing directory: {save_path}")

    # Define Features
    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.images.head": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
    }

    # Initialize Dataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=save_path,
        features=features,
        use_videos=True
    )

    count = 0
    for file_path in tqdm(files, desc="Converting Episodes"):
        data_dict = load_roco_updated_episode(file_path)
        
        if data_dict:
            num_frames = len(data_dict["action"])
            for i in range(num_frames):
                frame = {key: value[i] for key, value in data_dict.items()}
                frame["task"] = "Assemble gearbox" # Required field
                dataset.add_frame(frame)
            
            dataset.save_episode()
            count += 1
    
    # Finalize
    # Note: No consolidate() call needed for LeRobotDataset v2.0+
    print(f"Conversion Complete! Processed {count}/{len(files)} episodes.")
    print(f"Saved to: {args.out_dir}")

if __name__ == "__main__":
    batch_convert()
