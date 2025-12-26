# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay recorded HDF5 demonstrations in Isaac Lab environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded demonstrations.")
parser.add_argument("--task", type=str, default="Template-Galaxea-Sun-Gear-3-v0", help="Name of the task.")
parser.add_argument("--hdf5_file", type=str, required=True, help="Path to HDF5 file to replay.")
parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--loop", action="store_true", help="Loop the replay continuously.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import numpy as np
import torch
import time

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import Galaxea_Lab_External.tasks


def load_hdf5_data(filepath: str) -> dict:
    """Load demonstration data from HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Print metadata
        print("\n" + "="*60)
        print("LOADING DEMONSTRATION")
        print("="*60)
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        # Load observations
        data['left_arm_joint_pos'] = f['observations/left_arm_joint_pos'][:]
        data['right_arm_joint_pos'] = f['observations/right_arm_joint_pos'][:]
        data['left_gripper_joint_pos'] = f['observations/left_gripper_joint_pos'][:]
        data['right_gripper_joint_pos'] = f['observations/right_gripper_joint_pos'][:]
        
        # Load actions
        data['left_arm_action'] = f['actions/left_arm_action'][:]
        data['right_arm_action'] = f['actions/right_arm_action'][:]
        data['left_gripper_action'] = f['actions/left_gripper_action'][:]
        data['right_gripper_action'] = f['actions/right_gripper_action'][:]
        
        # Load timestamps
        data['current_time'] = f['current_time'][:]
        data['score'] = f['score'][:]
        
        print(f"  Frames loaded: {len(data['current_time'])}")
        print(f"  Duration: {data['current_time'][-1]:.2f}s")
        print("="*60 + "\n")
    
    return data


def main():
    """Replay recorded demonstration."""
    # Load HDF5 data
    demo_data = load_hdf5_data(args_cli.hdf5_file)
    num_frames = len(demo_data['current_time'])
    
    # Create environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.record_data = False  # Disable recording during replay
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # Get joint indices (same as in environment)
    left_arm_idx, _ = env.robot.find_joints(env.cfg.left_arm_joint_dof_name)
    right_arm_idx, _ = env.robot.find_joints(env.cfg.right_arm_joint_dof_name)
    left_gripper_idx, _ = env.robot.find_joints(env.cfg.left_gripper_dof_name)
    right_gripper_idx, _ = env.robot.find_joints(env.cfg.right_gripper_dof_name)
    
    all_joint_idx = left_arm_idx + right_arm_idx + left_gripper_idx + right_gripper_idx
    
    # Reset environment
    env.reset()
    
    # Calculate frame timing
    if num_frames > 1:
        dt = (demo_data['current_time'][-1] - demo_data['current_time'][0]) / (num_frames - 1)
    else:
        dt = 0.05  # Default 20Hz
    
    replay_dt = dt / args_cli.playback_speed
    
    print(f"Replaying at {1/replay_dt:.1f} Hz (speed: {args_cli.playback_speed}x)")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            for frame_idx in range(num_frames):
                start_time = time.time()
                
                # Get actions for this frame
                left_arm_act = torch.tensor(demo_data['left_arm_action'][frame_idx], device=env.device)
                right_arm_act = torch.tensor(demo_data['right_arm_action'][frame_idx], device=env.device)
                left_gripper_act = torch.tensor([demo_data['left_gripper_action'][frame_idx]], device=env.device)
                right_gripper_act = torch.tensor([demo_data['right_gripper_action'][frame_idx]], device=env.device)
                
                # Combine actions
                combined_action = torch.cat([left_arm_act, right_arm_act, left_gripper_act, right_gripper_act])
                
                # Apply joint position targets and step physics (with decimation)
                for _ in range(env.cfg.decimation):
                    env.robot.set_joint_position_target(combined_action, joint_ids=all_joint_idx)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                
                # Render
                env.sim.render()
                
                # Print progress
                score = demo_data['score'][frame_idx]
                if frame_idx % 20 == 0:
                    print(f"Frame {frame_idx}/{num_frames} | Time: {demo_data['current_time'][frame_idx]:.2f}s | Score: {score}")
                
                # Maintain playback timing
                elapsed = time.time() - start_time
                if elapsed < replay_dt:
                    time.sleep(replay_dt - elapsed)
            
            if not args_cli.loop:
                print("\nReplay complete!")
                break
            else:
                print("\n--- Looping replay ---\n")
                env.reset()
                
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
