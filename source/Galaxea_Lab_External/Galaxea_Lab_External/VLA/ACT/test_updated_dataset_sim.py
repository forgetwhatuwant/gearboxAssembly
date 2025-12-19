#!/usr/bin/env python3
"""
Test Updated LeRobot Dataset in Simulator
Replays 'roco_updated_debug' to verify the new 'Real-Action + Offset' strategy.
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
parser = argparse.ArgumentParser(description="Test Updated Dataset")
parser.add_argument("--dataset_dir", type=str, default="/media/hls/HIKSEMI/isaac_sim_data/lerobot_datasets/roco_updated_debug", help="Path to LeRobot dataset")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-External-Direct-v0", help="Task name")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to replay")
parser.add_argument("--random", action="store_true", help="Pick a random episode")
parser.add_argument("--episode_idx", type=int, default=None, help="Replay specific episode")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks

# Add lerobot to path
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def verify_dataset():
    print(f"Loading dataset from: {args_cli.dataset_dir}")
    # Use generic repo_id as we only care about local loading
    dataset = LeRobotDataset(root=args_cli.dataset_dir, repo_id="galaxea/roco_updated_debug")
    print(f"Dataset loaded. Total episodes: {dataset.num_episodes}")
    
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Determine episode
    episode_idx = 0
    if args_cli.random:
        episode_idx = np.random.randint(0, dataset.num_episodes)
    elif args_cli.episode_idx is not None:
        episode_idx = args_cli.episode_idx

    print(f"\nReplaying Episode {episode_idx}...")
    
    # Get frame range
    val_from = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    from_idx = val_from.item() if hasattr(val_from, "item") else val_from
    
    val_to = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    to_idx = val_to.item() if hasattr(val_to, "item") else val_to
    length = to_idx - from_idx
    print(f"  Frame range: {from_idx} -> {to_idx} ({length} frames)")
    
    # Plotting History
    l_grip_acts_hist = []
    l_grip_obs_hist = []
    
    env.reset()
    
    if length > 0:
        first_frame = dataset[from_idx]
        start_state = first_frame["observation.state"]
        start_tensor = start_state.unsqueeze(0).to(env.unwrapped.device)
        env.reset()
        # Stabilize
        for _ in range(20): 
            env.step(start_tensor)

    # Replay Loop
    # We replay the WHOLE episode to catch the grasping moment
    for i in range(length):
        frame_idx = from_idx + i
        frame = dataset[frame_idx]
        action = frame["action"]
        
        action_tensor = action.unsqueeze(0).to(env.unwrapped.device)
        obs, _, _, _, _ = env.step(action_tensor)
        
        # Log data
        act_l_grip = action[12].item()
        
        # Extract Obs safely
        debug_obs = obs.get('policy', obs)
        obs_l_grip = 0.0
        if 'left_gripper_joint_pos' in debug_obs:
            t = debug_obs['left_gripper_joint_pos']
            obs_l_grip = t.item() if hasattr(t, 'item') else float(t)
            
        l_grip_acts_hist.append(act_l_grip)
        l_grip_obs_hist.append(obs_l_grip)

        if i % 50 == 0:
            print(f"Frame {i}: Act={act_l_grip:.4f} | Obs={obs_l_grip:.4f}")

    # PLOT
    plt.figure(figsize=(10, 5))
    plt.plot(l_grip_acts_hist, label='Action', linestyle='--', color='blue')
    plt.plot(l_grip_obs_hist, label='Observation', color='orange')
    plt.title(f'Left Gripper Response (Real Actions + Offset) - Ep {episode_idx}')
    plt.legend()
    plt.grid(True)
    out_path = f"plot_updated_ep{episode_idx}.png"
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")
    
    env.close()

if __name__ == "__main__":
    verify_dataset()
