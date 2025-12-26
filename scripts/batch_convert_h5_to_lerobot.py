#!/usr/bin/env python3
"""
Batch convert multiple HDF5 episodes to a single LeRobot dataset.
Usage: python scripts/batch_convert_h5_to_lerobot.py --src_dir <dir> --output_dir <dir>
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from glob import glob

# Add lerobot to path
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_features(height=480, width=640):
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


def load_h5_episode(file_path: str):
    """Load a sun_gear HDF5 episode and return data for LeRobot."""
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
        
        # Concatenate to 14-dim state
        states = np.concatenate([l_arm_pos, r_arm_pos, l_grip_pos, r_grip_pos], axis=1)
        
        # Actions
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
        
        return {
            "action": actions.astype(np.float32),
            "observation.state": states.astype(np.float32),
            "observation.images.head": head_rgb,
            "observation.images.left_wrist": left_hand_rgb,
            "observation.images.right_wrist": right_hand_rgb,
        }, length, height, width


def main():
    parser = argparse.ArgumentParser(description="Batch convert H5 files to LeRobot format")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="galaxea/sun_gear_3", help="Repository ID")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate")
    parser.add_argument("--task", type=str, default="Pick and place sun gears onto the planetary carrier pins", 
                        help="Task description")
    parser.add_argument("--pattern", type=str, default="*.h5", help="File pattern to match (default: *.h5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")
    args = parser.parse_args()
    
    # Find all H5 files
    h5_files = sorted(glob(os.path.join(args.src_dir, args.pattern)))
    
    if not h5_files:
        print(f"Error: No H5 files found in {args.src_dir} with pattern {args.pattern}")
        return
    
    print(f"Found {len(h5_files)} H5 files to convert")
    
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
    
    # Detect resolution from first file
    print(f"Detecting resolution from first file...")
    _, _, height, width = load_h5_episode(h5_files[0])
    print(f"Detected resolution: {height}x{width}")
    
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
    
    total_frames = 0
    successful_episodes = 0
    
    # Process each H5 file
    for idx, h5_file in enumerate(h5_files):
        print(f"\n[{idx+1}/{len(h5_files)}] Processing: {os.path.basename(h5_file)}")
        
        try:
            episode_data, length, h, w = load_h5_episode(h5_file)
            
            if length == 0:
                print(f"  ⚠️ Skipping: No data in episode")
                continue
            
            if h != height or w != width:
                print(f"  ⚠️ Skipping: Resolution mismatch ({h}x{w} vs {height}x{width})")
                continue
            
            # Add frames
            for frame_idx in range(length):
                frame = {
                    "action": episode_data["action"][frame_idx],
                    "observation.state": episode_data["observation.state"][frame_idx],
                    "observation.images.head": episode_data["observation.images.head"][frame_idx],
                    "observation.images.left_wrist": episode_data["observation.images.left_wrist"][frame_idx],
                    "observation.images.right_wrist": episode_data["observation.images.right_wrist"][frame_idx],
                    "task": args.task,
                }
                dataset.add_frame(frame)
            
            # Save episode
            dataset.save_episode()
            total_frames += length
            successful_episodes += 1
            print(f"  ✅ Added episode: {length} frames")
            
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ Batch conversion complete!")
    print(f"   Dataset saved to: {args.output_dir}")
    print(f"   Episodes: {successful_episodes}/{len(h5_files)}")
    print(f"   Total frames: {total_frames}")
    print(f"   Task: {args.task}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
