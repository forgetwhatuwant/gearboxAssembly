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
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, use_action=not args_cli.no_action)

    # sample_every_n_steps = max(int(sample_period / env.step_dt), 1)
    print("env type: ", type(env))

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
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

            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            # env.step(actions)
            # if terminated or truncated:
            #     env.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
