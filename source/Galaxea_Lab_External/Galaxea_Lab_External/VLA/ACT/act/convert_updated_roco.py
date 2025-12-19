
import h5py
import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import argparse

# --- CONFIGURATION ---
# The source "Updated" dataset path
RAW_DIR_DEFAULT = "/home/hls/.cache/huggingface/hub/datasets--rocochallenge2025--rocochallenge2025/snapshots/76a5691156397e249cf9b8f568d37407c302d724/gearbox_assembly_demos_updated"
# Where to save the converted LeRobot dataset
OUTPUT_DIR_DEFAULT = "/media/hls/HIKSEMI/isaac_sim_data/lerobot_datasets/roco_updated_debug"

VIDEO_KEYS = ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"]

def load_roco_updated_episode(file_path, fps):
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
            # Now reading from the 'actions' group which is valid in the 'updated' dataset
            act_l_arm = f['actions/left_arm_action'][:]
            act_l_grip = f['actions/left_gripper_action'][:]
            act_r_arm = f['actions/right_arm_action'][:]
            act_r_grip = f['actions/right_gripper_action'][:]

            if act_l_grip.ndim == 1: act_l_grip = act_l_grip[:, None]
            if act_r_grip.ndim == 1: act_r_grip = act_r_grip[:, None]

            # --- GRIPPER OFFSET LOGIC ---
            # Real actions are ~0.03 (weak). We apply -0.01 to force them towards 0.0 (strong grasp).
            # Logic: New_Act = Real_Act - 0.01
            OFFSET = 0.01
            act_l_grip = act_l_grip - OFFSET
            act_r_grip = act_r_grip - OFFSET
            
            # Clip to valid range [0.0, 0.04]
            act_l_grip = np.clip(act_l_grip, 0.0, 0.04)
            act_r_grip = np.clip(act_r_grip, 0.0, 0.04)

            # Concatenate actions (L=14)
            actions = np.concatenate([act_l_arm, act_r_arm, act_l_grip, act_r_grip], axis=1)

            # --- IMAGES ---
            head_rgb = f['observations/head_rgb'][:]
            left_hand_rgb = f['observations/left_hand_rgb'][:]
            right_hand_rgb = f['observations/right_hand_rgb'][:]

            length = actions.shape[0]
            assert length == states.shape[0] == head_rgb.shape[0], "Length Mismatch!"

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

def convert_single_episode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="100.hdf5", help="File name in updated dir")
    args = parser.parse_args()

    raw_path = Path(RAW_DIR_DEFAULT) / args.file
    local_dir = Path(OUTPUT_DIR_DEFAULT)
    
    print(f"Converting: {raw_path}")
    print(f"Output: {local_dir}")

    # Initialize LeRobot Dataset
    if local_dir.exists():
        import shutil
        shutil.rmtree(local_dir) # CLEAN START for debug
    
    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.images.head": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channels"]},
    }

    dataset = LeRobotDataset.create(
        repo_id="galaxea/roco_updated_debug",
        fps=20,
        root=local_dir,
        features=features,
        use_videos=True
    )

    data_dict = load_roco_updated_episode(raw_path, fps=20)
    
    if data_dict:
        num_frames = len(data_dict["action"])
        for i in range(num_frames):
            frame = {key: value[i] for key, value in data_dict.items()}
            frame["task"] = "Assemble gearbox"
            dataset.add_frame(frame)
        
        dataset.save_episode()
        print(f"Successfully converted {num_frames} frames.")
    
    print("Dataset created!")

if __name__ == "__main__":
    convert_single_episode()
