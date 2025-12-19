#!/usr/bin/env python3
"""
LeRobot Dataset Verification Script

This script replays a converted LeRobot dataset in the Isaac Lab environment
to verify coordinate system alignment, action feasibility, and data integrity.

Usage:
    python verify_lerobot_dataset.py --dataset_dir /path/to/dataset --num_episodes 1
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="LeRobot Dataset Verification")
parser.add_argument("--dataset_dir", type=str, default="/media/hls/MK-ssd-mini/isaac_sim_data/lerobot_datasets/roco_gearbox", help="Path to LeRobot dataset")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-External-Direct-v0", help="Task name")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to replay")
parser.add_argument("--fps", type=int, default=20, help="Playback FPS (determines wait time)")
parser.add_argument("--random", action="store_true", help="Pick a random episode to replay")
parser.add_argument("--episode_idx", type=int, default=None, help="Replay a specific episode index (overrides num_episodes)")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# Add project root to path to allow importing Galaxea_Lab_External
# Path: source/Galaxea_Lab_External
start_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if start_path not in sys.path:
    sys.path.append(start_path)

import Galaxea_Lab_External.tasks

# Add lerobot to path
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print(f"Error: Could not import LeRobotDataset from {lerobot_path}")
    sys.exit(1)

def verify_dataset():
    print(f"Loading dataset from: {args_cli.dataset_dir}")
    dataset = LeRobotDataset(root=args_cli.dataset_dir, repo_id="galaxea/roco_gearbox")
    print(f"Dataset loaded. Total episodes: {dataset.num_episodes}")
    
    # Create Environment
    # Create Environment
    # parse_env_cfg signature: (task_name: str, device: str = 'cuda:0', num_envs: int = None, use_fabric: bool = True)
    # AppLauncher handles global device settings, but we might need to pass it here.
    # Trying with just task and num_envs, or check if device arg is available from args or AppLauncher
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    
    env = gym.make(args_cli.task, cfg=env_cfg)

    # --- DIAGNOSTICS ---
    print("\n--- DIAGNOSTICS ---")
    try:
        unwrapped_env = env.unwrapped
        joint_indices = unwrapped_env._joint_idx
        print(f"Joint Indices Length: {len(joint_indices)}")
        print(f"Joint Indices: {joint_indices}")
        
        # Print Mapping
        robot = unwrapped_env.robot
        all_joint_names = robot.data.joint_names
        print("Joint Index -> Name Mapping:")
        for idx in joint_indices:
             print(f"  {idx} -> {all_joint_names[idx]}")

        # Verify Stiffness
        gripper_stiff = env_cfg.robot_cfg.actuators["r1_grippers"].stiffness
        print(f"Gripper Stiffness Object: {gripper_stiff}")
        
    except Exception as e:
        print(f"Diagnostics Error: {e}")
    print("-------------------\n")

    
    # Determine which episodes to replay
    episode_indices = []
    if args_cli.random:
        idx = np.random.randint(0, dataset.num_episodes)
        print(f"Randomly selected Episode: {idx}")
        episode_indices.append(idx)
    elif args_cli.episode_idx is not None:
        if args_cli.episode_idx >= dataset.num_episodes:
             print(f"Error: Episode index {args_cli.episode_idx} out of range (Total: {dataset.num_episodes})")
             return
        print(f"Selected specific Episode: {args_cli.episode_idx}")
        episode_indices.append(args_cli.episode_idx)
    else:
        # Default: First N episodes
        count = min(args_cli.num_episodes, dataset.num_episodes)
        episode_indices = list(range(count))
        
    # Verify loop
    for episode_idx in episode_indices:
        print(f"\nReplaying Episode {episode_idx}...")
        
        episode_meta = dataset.meta.episodes[episode_idx]
        length = episode_meta["length"].item() if hasattr(episode_meta["length"], "item") else episode_meta["length"]
        
        from_idx = 0
        for i in range(episode_idx):
            prev_len = dataset.meta.episodes[i]["length"]
            prev_len = prev_len.item() if hasattr(prev_len, "item") else prev_len
            from_idx += prev_len
            
        to_idx = from_idx + length
        print(f"  Frame range: {from_idx} -> {to_idx} ({length} frames)")
        
        # history for plotting
        l_grip_acts_hist = []
        l_grip_obs_hist = []
        r_grip_acts_hist = []
        r_grip_obs_hist = []

        obs, _ = env.reset()
        
        if length > 0:
            first_frame = dataset[from_idx]
            start_state = first_frame["observation.state"]
            start_tensor = start_state.unsqueeze(0).to(env.unwrapped.device)
            env.reset()
            for _ in range(20): 
                env.step(start_tensor)
        
        # Replay loop (Limit to 100 frames for debug)
        limit = min(length, 100)
        for i in range(limit):
            frame_idx = from_idx + i
            frame = dataset[frame_idx]
            action = frame["action"]
            
            # Send to env
            action_tensor = action.unsqueeze(0).to(env.unwrapped.device)
            
            # Step
            obs, rew, terminated, truncated, info = env.step(action_tensor)
            
            if i % 10 == 0:
                print(f"Frame {i}:")
                # Side-by-side comparison
                act_l_grip = action[12].item()
                obs_l_grip = 0.0
                debug_obs = obs.get('policy', obs)
                if 'left_gripper_joint_pos' in debug_obs:
                     obs_l_grip = debug_obs['left_gripper_joint_pos'].item() if isinstance(debug_obs['left_gripper_joint_pos'], torch.Tensor) else debug_obs['left_gripper_joint_pos']
                     # Handle single-element tensor
                     if hasattr(obs_l_grip, 'item'): obs_l_grip = obs_l_grip.item() 
                     elif isinstance(obs_l_grip, list) or (isinstance(obs_l_grip, np.ndarray) and obs_l_grip.size==1): obs_l_grip = float(obs_l_grip)

                print(f"  Left Grip: Act={act_l_grip:.4f} vs Obs={obs_l_grip:.4f}")

            # Collect data for plotting
            # Action
            l_grip_acts_hist.append(action[12].item())
            r_grip_acts_hist.append(action[13].item())
            
            # Obs
            debug_obs = obs.get('policy', obs)
            
            # Left
            val_l = 0.0
            if 'left_gripper_joint_pos' in debug_obs:
                t = debug_obs['left_gripper_joint_pos'] 
                val_l = t.item() if hasattr(t, 'item') else float(t)
            l_grip_obs_hist.append(val_l)
            
            # Right
            val_r = 0.0
            if 'right_gripper_joint_pos' in debug_obs:
                t = debug_obs['right_gripper_joint_pos']
                val_r = t.item() if hasattr(t, 'item') else float(t)
            r_grip_obs_hist.append(val_r)

        # End of episode loop - PLOT
        print(f"Generating plot for Episode {episode_idx}...")
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(l_grip_acts_hist, label='Action', linestyle='--', color='blue')
        plt.plot(l_grip_obs_hist, label='Observation', color='cyan')
        plt.title(f'Left Gripper: Act vs Obs (Ep {episode_idx})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(r_grip_acts_hist, label='Action', linestyle='--', color='red')
        plt.plot(r_grip_obs_hist, label='Observation', color='orange')
        plt.title(f'Right Gripper: Act vs Obs (Ep {episode_idx})')
        plt.legend()
        plt.grid(True)
        
        out_path = f"gripper_plot_ep{episode_idx}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved gripper plot to: {out_path}")

        # Save raw data to TXT
        txt_path = f"gripper_data_ep{episode_idx}.txt"
        with open(txt_path, "w") as f:
            f.write("Frame,L_Act,L_Obs,R_Act,R_Obs\n")
            for t in range(len(l_grip_acts_hist)):
                f.write(f"{t},{l_grip_acts_hist[t]:.6f},{l_grip_obs_hist[t]:.6f},{r_grip_acts_hist[t]:.6f},{r_grip_obs_hist[t]:.6f}\n")
        print(f"Saved gripper data to: {txt_path}")


            


        print(f"Episode {episode_idx} complete.")

    env.close()
    print("\nVerification complete.")

if __name__ == "__main__":
    verify_dataset()
