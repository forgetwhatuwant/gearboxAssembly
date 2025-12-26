# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--no_action", action="store_true", default=False, help="Do not apply actions to the robot.")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save recorded data.")
parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to record (0=infinite).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

print(f"args_cli: {args_cli}")
print(f"Python path: {sys.path}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # Override data directory in config before creating environment
    if hasattr(env_cfg, 'data_dir'):
        env_cfg.data_dir = args_cli.data_dir
        print(f"[INFO]: Data will be saved to: {args_cli.data_dir}")
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, use_action=not args_cli.no_action)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # Episode tracking
    episode_count = 0
    target_episodes = args_cli.num_episodes
    if target_episodes > 0:
        print(f"[INFO]: Will record {target_episodes} episodes then exit")
    else:
        print(f"[INFO]: Running indefinitely (Ctrl+C to stop)")
    
    # reset environment
    env.reset()
    
    # Set viewport camera to head_cam perspective
    try:
        import omni.kit.viewport.utility as viewport_utils
        # Get the active viewport and set camera to head_cam prim
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            head_cam_prim_path = "/World/envs/env_0/Robot/zed_link/head_cam/head_cam"
            viewport.set_active_camera(head_cam_prim_path)
            print(f"[INFO]: Viewport camera set to head_cam: {head_cam_prim_path}")
    except Exception as e:
        print(f"[WARN]: Could not set viewport to head_cam: {e}")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # The environment handles the rule-based policy internally.
            # We just need to pass a dummy action to step().
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)

            # Check if episode ended
            if terminated.any() or truncated.any():
                episode_count += 1
                print(f"\n[INFO]: Episode {episode_count} completed")
                
                # Check if we've reached target episodes
                if target_episodes > 0 and episode_count >= target_episodes:
                    print(f"\n[SUCCESS]: Completed {episode_count} episodes. Exiting...")
                    break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

