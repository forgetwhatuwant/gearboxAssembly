#!/usr/bin/env python3
"""
Convert a single HDF5 episode to LeRobot dataset format.
Usage: python scripts/convert_single_h5_to_lerobot.py --src <file.h5> --output_dir <dir>
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path

# Add lerobot to path
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Feature definition - matching sun_gear_3 data structure
def get_features(height=240, width=320):
    """Get feature definition based on image resolution."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
                "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
                "left_gripper", "right_gripper",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
                "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
                "left_gripper", "right_gripper",
            ],
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
    }


def load_h5_episode(file_path: str, fps: int = 20):
    """Load a sun_gear_3 HDF5 episode and return data for LeRobot."""
    with h5py.File(file_path, 'r') as f:
        # Joint positions (observations)
        l_arm_pos = f['observations/left_arm_joint_pos'][:]
        l_grip_pos = f['observations/left_gripper_joint_pos'][:]
        r_arm_pos = f['observations/right_arm_joint_pos'][:]
        r_grip_pos = f['observations/right_gripper_joint_pos'][:]
        
        # Ensure grippers are 2D
        if l_grip_pos.ndim == 1:
            l_grip_pos = l_grip_pos[:, None]
        if r_grip_pos.ndim == 1:
            r_grip_pos = r_grip_pos[:, None]
        
        # Concatenate to 14-dim state: [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
        states = np.concatenate([l_arm_pos, r_arm_pos, l_grip_pos, r_grip_pos], axis=1)
        
        # Actions - use actual recorded actions if available
        l_arm_act = f['actions/left_arm_action'][:]
        l_grip_act = f['actions/left_gripper_action'][:]
        r_arm_act = f['actions/right_arm_action'][:]
        r_grip_act = f['actions/right_gripper_action'][:]
        
        if l_grip_act.ndim == 1:
            l_grip_act = l_grip_act[:, None]
        if r_grip_act.ndim == 1:
            r_grip_act = r_grip_act[:, None]
        
        actions = np.concatenate([l_arm_act, r_arm_act, l_grip_act, r_grip_act], axis=1)
        
        # Images
        head_rgb = f['observations/head_rgb'][:]
        left_hand_rgb = f['observations/left_hand_rgb'][:]
        right_hand_rgb = f['observations/right_hand_rgb'][:]
        
        # Get resolution
        height, width = head_rgb.shape[1], head_rgb.shape[2]
        
        length = actions.shape[0]
        
        print(f"Loaded episode: {length} frames, resolution: {height}x{width}")
        print(f"  State shape: {states.shape}")
        print(f"  Action shape: {actions.shape}")
        
        return {
            "action": actions.astype(np.float32),
            "observation.state": states.astype(np.float32),
            "observation.images.head": head_rgb,
            "observation.images.left_wrist": left_hand_rgb,
            "observation.images.right_wrist": right_hand_rgb,
        }, length, height, width


def main():
    parser = argparse.ArgumentParser(description="Convert single H5 to LeRobot format")
    parser.add_argument("--src", type=str, required=True, help="Source HDF5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="galaxea/sun_gear_3", help="Repository ID")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate")
    parser.add_argument("--task", type=str, default="Pick and place sun gears onto the planetary carrier pins", help="Task description")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.src):
        print(f"Error: Source file not found: {args.src}")
        return
    
    # Load episode data
    print(f"Loading: {args.src}")
    episode_data, length, height, width = load_h5_episode(args.src, args.fps)
    
    if length == 0:
        print("Error: No data in episode")
        return
    
    # Handle existing directory
    if os.path.exists(args.output_dir):
        if args.overwrite:
            import shutil
            print(f"Removing existing directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            print(f"Error: Output directory exists: {args.output_dir}")
            print("Use --overwrite to replace")
            return
    
    # Get features based on detected resolution
    features = get_features(height, width)
    
    # Create LeRobot dataset
    print(f"Creating LeRobot dataset at: {args.output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=args.fps,
        features=features,
        use_videos=True,
    )
    
    # Add episode
    print(f"Adding episode with {length} frames...")
    for frame_idx in range(length):
        frame = {
            "action": episode_data["action"][frame_idx],
            "observation.state": episode_data["observation.state"][frame_idx],
            "observation.images.head": episode_data["observation.images.head"][frame_idx],
            "observation.images.left_wrist": episode_data["observation.images.left_wrist"][frame_idx],
            "observation.images.right_wrist": episode_data["observation.images.right_wrist"][frame_idx],
            "task": args.task,  # Required by LeRobot
        }
        dataset.add_frame(frame)
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{length}")
    
    # Save episode
    dataset.save_episode()
    
    print(f"\n✅ Conversion complete!")
    print(f"   Dataset saved to: {args.output_dir}")
    print(f"   Episodes: {dataset.num_episodes}")
    print(f"   Total frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
