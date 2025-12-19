
import os
import h5py
import numpy as np
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
import torch

# Add lerobot to path if necessary
import sys
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Feature definition for Galaxea R1 (Bimanual)
# Based on RoCo Challenge description
ROCO_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (14,), # 7 left + 7 right
        "names": [
            # Left Arm (7)
            "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
            # Right Arm (7)
            "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,), # 6 joint + 1 gripper per arm
        "names": [
            # Left Arm (7)
            "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
            # Right Arm (7)
            "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
        ],
    },
    "observation.images.head": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 240,
            "video.width": 320,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 240,
            "video.width": 320,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 240,
            "video.width": 320,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

def load_roco_episode(file_path, fps=20):
    """
    Reads a single ROCO HDF5 file and returns a dictionary of arrays 
    ready for LeRobotDataset addition.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # --- STATES (QPOS) ---
            l_arm_pos = f['observations/left_arm_joint_pos'][:]
            l_grip_pos = f['observations/left_gripper_joint_pos'][:]
            r_arm_pos = f['observations/right_arm_joint_pos'][:]
            r_grip_pos = f['observations/right_gripper_joint_pos'][:]

            if l_grip_pos.ndim == 1: l_grip_pos = l_grip_pos[:, None]
            if r_grip_pos.ndim == 1: r_grip_pos = r_grip_pos[:, None]

            # Debug: Check gripper value range
            print(f"DEBUG: {file_path}")
            print(f"  Left Gripper: Min={l_grip_pos.min():.4f}, Max={l_grip_pos.max():.4f}, Mean={l_grip_pos.mean():.4f}")
            print(f"  Right Gripper: Min={r_grip_pos.min():.4f}, Max={r_grip_pos.max():.4f}, Mean={r_grip_pos.mean():.4f}")

            # Concatenate to form 14-dim state vector
            # ORDER MUST MATCH Environment: Left Arm, Right Arm, Left Grip, Right Grip
            states = np.concatenate([l_arm_pos, r_arm_pos, l_grip_pos, r_grip_pos], axis=1) # (L, 14)

            # --- ACTIONS (Synthesized from States) ---
            # The source HDF5 has empty/zero 'actions', so we use State(t+1) as the Action(t).
            # This is standard for Behavior Cloning from valid trajectories.
            actions = np.zeros_like(states)
            actions[:-1] = states[1:]
            actions[-1] = states[-1] # Repeat last state for the final action

            # CONTINUOUS GRIPPER LOGIC (Leader-Follower Simulation):
            # To train ACT, we need actions that "lead" the observation to generate force.
            # We apply a negative offset (-0.01m) to the observed position.
            # This simulates the operator squeezing the leader arm "deeper" than the object allows.
            # Result: Continuous signal, smooth dynamics, but strong grasping force.
            
            OFFSET = 0.01
            
            # Left Gripper (Index 12)
            # Clip to [0.0, 1.0] or [0.0, 0.04]? Gripper range is 0.0 to 0.04.
            l_grip_acts = states[:, 12] - OFFSET
            l_grip_acts = np.clip(l_grip_acts, 0.0, 0.04)
            actions[:, 12] = l_grip_acts
            
            # Right Gripper (Index 13)
            r_grip_acts = states[:, 13] - OFFSET
            r_grip_acts = np.clip(r_grip_acts, 0.0, 0.04)
            actions[:, 13] = r_grip_acts

            # --- IMAGES ---
            # Keys in ROCO: head_rgb, left_hand_rgb, right_hand_rgb
            # Target Keys in Features: head, left_wrist, right_wrist
            
            head_rgb = f['observations/head_rgb'][:]
            # MAPPING: left_hand -> left_wrist
            left_hand_rgb = f['observations/left_hand_rgb'][:]
            # MAPPING: right_hand -> right_wrist
            right_hand_rgb = f['observations/right_hand_rgb'][:]

            length = actions.shape[0]
            
            # Ensure all lengths match
            assert length == states.shape[0] == head_rgb.shape[0]
            
            return {
                "action": actions,
                "observation.state": states,
                "observation.images.head": head_rgb,
                "observation.images.left_wrist": left_hand_rgb,
                "observation.images.right_wrist": right_hand_rgb
            }, length

    except Exception as e:
        # Only print error if it's NOT a missing file (which is expected for broken symlinks in this dataset)
        if "No such file or directory" not in str(e):
             print(f"Error reading {file_path}: {e}")
        return None, 0

def convert_roco_to_lerobot():
    parser = argparse.ArgumentParser(description="Convert ROCO HDF5 dataset to LeRobot format.")
    parser.add_argument("--src", type=str, default="/mnt/nas/isaac_sim_data/datasets--rocochallenge2025--rocochallenge2025",
                        help="Root directory of source dataset")
    parser.add_argument("--repo_id", type=str, default="galaxea/roco_gearbox", help="Repository ID for LeRobot Dataset")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory to save the LeRobot dataset")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to convert")
    parser.add_argument("--task", type=str, default="Assemble gearbox", help="Task description")
    
    args = parser.parse_args()

    # 1. Find Files
    print(f"Scanning for HDF5 files in {args.src}...")
    files = glob.glob(os.path.join(args.src, "**", "*.hdf5"), recursive=True)
    files.extend(glob.glob(os.path.join(args.src, "**", "*.h5"), recursive=True))
    files.sort()
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} potential files.")

    # 2. Initialize LeRobot Dataset
    print(f"Initializing LeRobotDataset at {args.local_dir}...")
    
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=ROCO_FEATURES,
        root=args.local_dir,
        robot_type="galaxea_r1"
    )

    count = 0
    pbar = tqdm(files)
    
    for file_path in pbar:
        if args.max_episodes and count >= args.max_episodes:
            break
            
        episode_data, length = load_roco_episode(file_path, fps=args.fps)
        
        if episode_data is not None:
            # Add frames
            for i in range(length):
                frame = {
                    key: episode_data[key][i] for key in episode_data
                }
                # Add task description to the frame dict, NOT as a kwarg to add_frame
                frame["task"] = args.task
                
                dataset.add_frame(frame)
                
            dataset.save_episode()
            count += 1
            pbar.set_description(f"Converted {count} eps")

    print(f"\nConversion complete. Saved {count} episodes to {args.local_dir}")
    
    # Stats calculation removed as it requires specific version match.
    # Users can compute stats separately if needed using lerobot tools.
    print("Conversion complete.")

if __name__ == "__main__":
    convert_roco_to_lerobot()
