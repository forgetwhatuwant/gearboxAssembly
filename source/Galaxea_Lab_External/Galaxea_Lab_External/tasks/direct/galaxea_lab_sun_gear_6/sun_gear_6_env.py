# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import numpy as np
import time
from datetime import datetime
# from torchvision.utils import save_image
from PIL import Image

from collections.abc import Sequence
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, euler_xyz_from_quat

from .sun_gear_6_env_cfg import SunGear6EnvCfg

from pxr import Usd, Sdf, UsdPhysics, UsdGeom, Gf
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

import isaacsim.core.utils.torch as torch_utils

from Galaxea_Lab_External.robots import SunGear6RulePolicy
from isaaclab.sensors import Camera

import h5py

class SunGear6Env(DirectRLEnv):
    """Stage 2 curriculum: gears 1-3 pre-assembled, assemble gears 4-6."""
    cfg: SunGear6EnvCfg

    def __init__(self, cfg: SunGear6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print(f"--------------------------------INIT--------------------------------")

        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)

        print(f"DEBUG: Robot Joint Names: {self.robot.joint_names}")
        print(f"DEBUG: Robot Num Dof: {self.robot.num_joints}")

        self._left_arm_action = torch.zeros(self._left_arm_joint_idx, device=self.device)
        self._right_arm_action = torch.zeros(self._right_arm_joint_idx, device=self.device)
        self._left_gripper_action = torch.zeros(1, device=self.device)
        self._right_gripper_action = torch.zeros(1, device=self.device)

        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        print(f"_torso_joint_idx: {self._torso_joint_idx}")

        self._torso_joint1_idx, _ = self.robot.find_joints(self.cfg.torso_joint1_dof_name)
        self._torso_joint2_idx, _ = self.robot.find_joints(self.cfg.torso_joint2_dof_name)
        self._torso_joint3_idx, _ = self.robot.find_joints(self.cfg.torso_joint3_dof_name)

        print(f"_left_arm_joint_idx: {self._left_arm_joint_idx}")
        print(f"_right_arm_joint_idx: {self._right_arm_joint_idx}")
        print(f"_left_gripper_dof_idx: {self._left_gripper_dof_idx}")
        print(f"_right_gripper_dof_idx: {self._right_gripper_dof_idx}")

        self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + self._left_gripper_dof_idx + self._right_gripper_dof_idx

        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]

        self.left_arm_joint_vel = self.robot.data.joint_vel[:, self._left_arm_joint_idx]
        self.right_arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_idx]
        self.left_gripper_joint_vel = self.robot.data.joint_vel[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_vel = self.robot.data.joint_vel[:, self._right_gripper_dof_idx]
        
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_idx]

        self.data_dict = {
            '/observations/head_rgb': [],
            '/observations/left_hand_rgb': [],
            '/observations/right_hand_rgb': [],
            '/observations/left_arm_joint_pos': [],
            '/observations/right_arm_joint_pos': [],
            '/observations/left_gripper_joint_pos': [],
            '/observations/right_gripper_joint_pos': [],
            '/actions/left_arm_action': [],
            '/actions/right_arm_action': [],
            '/actions/left_gripper_action': [],
            '/actions/right_gripper_action': [],
            '/score': [],
            '/current_time': [],
        }
        
        # Counter to wait 5 seconds after success before terminating
        self.success_wait_steps = 40  # 5 seconds at 20Hz
        self.success_wait_counter = 0
        self.task_completed = False

    def _setup_scene(self):

        print(f"--------------------------------SETUP SCENE--------------------------------")

        self.robot = Articulation(self.cfg.robot_cfg)
        
        self.head_camera = Camera(self.cfg.head_camera_cfg)
        self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
        self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

        self.table = RigidObject(self.cfg.table_cfg)

        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0471, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0471, 0.0268, 0.0], device=self.device),  # pin_2
        ]


        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.obj_dict = {"ring_gear": self.ring_gear,
                        "planetary_carrier": self.planetary_carrier,
                        "sun_planetary_gear_1": self.sun_planetary_gear_1,
                        "sun_planetary_gear_2": self.sun_planetary_gear_2,
                        "sun_planetary_gear_3": self.sun_planetary_gear_3,
                        "sun_planetary_gear_4": self.sun_planetary_gear_4,
                        "planetary_reducer": self.planetary_reducer}

        self._initialize_scene()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        print(f"--------------------------------PRE PHYSICS STEP at {mdp.observations.current_time_s(self).item()} seconds--------------------------------")
        pass

    def _apply_action(self) -> None:
        start_time = time.time()
        current_time_s = mdp.observations.current_time_s(self)
        print(f"Apply action: {current_time_s.item()} seconds")
        action = self.env_step_action
        joint_ids = self.env_step_joint_ids

        if joint_ids is not None:
            self.robot.set_joint_position_target(action, joint_ids=joint_ids)

        self.rule_policy.count += 1
        sim_dt = self.sim.get_physics_dt()

        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)

        for cam in [self.head_camera, self.left_hand_camera, self.right_hand_camera]:
            cam.update(dt=sim_dt)

        end_time = time.time()

    def _get_observations(self) -> dict:
        current_time_s = mdp.observations.current_time_s(self)
        
        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]

        self.left_arm_joint_vel = self.robot.data.joint_vel[:, self._left_arm_joint_idx]
        self.right_arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_idx]
        self.left_gripper_joint_vel = self.robot.data.joint_vel[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_vel = self.robot.data.joint_vel[:, self._right_gripper_dof_idx]

        self.obs = dict(
            head_rgb=self.head_camera.data.output['rgb'],
            head_depth=self.head_camera.data.output['distance_to_image_plane'],
            left_hand_rgb=self.left_hand_camera.data.output['rgb'],
            left_hand_depth=self.left_hand_camera.data.output['distance_to_image_plane'],
            right_hand_rgb=self.right_hand_camera.data.output['rgb'],
            right_hand_depth=self.right_hand_camera.data.output['distance_to_image_plane'],
            left_arm_joint_pos=self.left_arm_joint_pos,
            right_arm_joint_pos=self.right_arm_joint_pos,
            left_gripper_joint_pos=self.left_gripper_joint_pos,
            right_gripper_joint_pos=self.right_gripper_joint_pos,
            left_arm_joint_vel=self.left_arm_joint_vel,
            right_arm_joint_vel=self.right_arm_joint_vel,
            left_gripper_joint_vel=self.left_gripper_joint_vel,
            right_gripper_joint_vel=self.right_gripper_joint_vel,
        )

        return {"policy": self.obs}

    def _get_rewards(self) -> torch.Tensor:
        score = self.evaluate_score()
        self.score = score
        return torch.tensor([score], device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if task is completed (score >= 6 for Stage 2)
        if self.score >= 6 and not self.task_completed:
            self.task_completed = True
            self.success_wait_counter = 0
            print(f"Task completed! Waiting {self.success_wait_steps / 20.0:.1f}s before reset...")
        
        # Wait 1 second after success before terminating
        if self.task_completed:
            self.success_wait_counter += 1
            if self.success_wait_counter >= self.success_wait_steps:
                terminated = torch.tensor([True], device=self.device)
            else:
                terminated = torch.tensor([False], device=self.device)
        else:
            terminated = torch.tensor([False], device=self.device)
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print(f"--------------------------------RESET--------------------------------")
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.rule_policy = SunGear6RulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
        self.initial_root_state = None

        self.env_step_action = None
        self.env_step_joint_ids = None

        self.act = dict()
        self.obs = dict()

        self.score = 0
        self.task_completed = False
        self.success_wait_counter = 0

        # Reset Table
        root_state = self.table.data.default_root_state.clone()
        self.table.write_root_state_to_sim(root_state)

        self.save_hdf5_file_name = f"data_sun_gear_6_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"

        # For Stage 2: Pre-assemble gears 1-3 on pins, randomize remaining parts
        # First randomize carrier and parts that go on table
        self.initial_root_state = self._randomize_object_positions(
            [self.planetary_carrier, self.ring_gear, 
             self.sun_planetary_gear_4, self.planetary_reducer], 
            ['planetary_carrier', 'ring_gear', 
             'sun_planetary_gear_4', 'planetary_reducer'])
        
        # Pre-position gears 1-3 on pins (mounted on carrier)
        self._pre_assemble_gears_1_3()
        
        for obj_name, obj in self.obj_dict.items():
            obj.update(self.sim.get_physics_dt())

        self.rule_policy.set_initial_root_state(self.initial_root_state)
        self.rule_policy.prepare_mounting_plan()

        joint_pos = self.robot.data.default_joint_pos[env_ids, self._joint_idx]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)

        # Write the default torso joint position to simulation
        self.robot.write_joint_position_to_sim(
            torch.tensor([self.cfg.initial_torso_joint1_pos, self.cfg.initial_torso_joint2_pos, self.cfg.initial_torso_joint3_pos], device=self.device), 
            self._torso_joint_idx, env_ids)

        # Set torso joint position limit
        self.robot.write_joint_position_limit_to_sim(
            torch.tensor([self.cfg.initial_torso_joint1_pos, self.cfg.initial_torso_joint1_pos], device=self.device), 
            self._torso_joint1_idx, env_ids)
        self.robot.write_joint_position_limit_to_sim(
            torch.tensor([self.cfg.initial_torso_joint2_pos, self.cfg.initial_torso_joint2_pos], device=self.device), 
            self._torso_joint2_idx, env_ids)
        self.robot.write_joint_position_limit_to_sim(
            torch.tensor([self.cfg.initial_torso_joint3_pos, self.cfg.initial_torso_joint3_pos], device=self.device), 
            self._torso_joint3_idx, env_ids)

        # Clean buffers
        for key in self.data_dict:
            self.data_dict[key] = []


    def step(self, action: torch.Tensor):
        current_time_s = mdp.observations.current_time_s(self)
        print(f"--------------------------------RL step at {current_time_s.item()} seconds--------------------------------")
        
        action = action.to(self.device)
        self._pre_physics_step(action)
        
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        print(f"Generate action at {current_time_s.item()} seconds")
        
        # Use internal rule policy
        self.env_step_action, self.env_step_joint_ids = self.rule_policy.get_action()
        print(f"Action: {self.env_step_action}")

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Only save successful episodes (terminated = score >= 6)
            if self.reset_terminated.any():
                print(f"Episode successful! Score: {self.score}. Writing data to hdf5 file")
                self._write_hdf5()
            else:
                print(f"Episode timeout/failed. Score: {self.score}. Discarding data.")
                # Clear the data buffer without saving
                for key in self.data_dict:
                    self.data_dict[key] = []
            
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        self.obs_buf = self._get_observations()
        
        # Post step data calc
        current_pos = self.robot.data.joint_pos
        self._left_arm_action = current_pos[:, self._left_arm_joint_idx]
        self._right_arm_action = current_pos[:, self._right_arm_joint_idx]
        self._left_gripper_action = current_pos[:, self._left_gripper_dof_idx[0]]
        self._right_gripper_action = current_pos[:, self._right_gripper_dof_idx[0]]
        
        if self.env_step_joint_ids == self._left_arm_joint_idx:
            self._left_arm_action = self.env_step_action.clone()
        elif self.env_step_joint_ids == self._right_arm_joint_idx:
            self._right_arm_action = self.env_step_action.clone()
        elif self.env_step_joint_ids == self._left_arm_joint_idx + self._right_arm_joint_idx:
            self._left_arm_action = self.env_step_action.clone()[:, :6]
            self._right_arm_action = self.env_step_action.clone()[:, 6:12]
        elif self.env_step_joint_ids == self._left_gripper_dof_idx:
            self._left_gripper_action = self.env_step_action[0].clone()
        elif self.env_step_joint_ids == self._right_gripper_dof_idx:
            self._right_gripper_action = self.env_step_action[0].clone()

        self.act = dict(left_arm_action=self._left_arm_action, right_arm_action=self._right_arm_action,
            left_gripper_action=self._left_gripper_action, right_gripper_action=self._right_gripper_action)

        if self.cfg.record_data and (self.rule_policy.count % self.cfg.record_freq == 0):
            self._record_data()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def evaluate_score(self):
        """Evaluate score for Stage 2: check all 6 assembly conditions."""
        score = 0
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()
        num_envs = planetary_carrier_quat.shape[0]
        
        # Score 1-3: Gears 1-3 should be on pins (pre-assembled, should always be true)
        for gear_idx, gear_name in enumerate(["sun_planetary_gear_1", "sun_planetary_gear_2", "sun_planetary_gear_3"]):
            gear = self.obj_dict[gear_name]
            pin_local_pos = self.pin_local_positions[gear_idx]
            
            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, -1)
            pin_local_pos_batched = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
            
            pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, 
                identity_quat, pin_local_pos_batched
            )[1]

            gear_pos = gear.data.root_state_w[:, :3]
            distance = torch.norm(gear_pos - pin_world_pos)
            
            if distance < 0.05:
                score += 1
        
        # Score 4: Gear 4 should be on top of planetary carrier (center)
        gear4_pos = self.sun_planetary_gear_4.data.root_state_w[:, :3]
        carrier_center = planetary_carrier_pos.clone()
        carrier_center[:, 2] += 0.02  # Gear 4 sits slightly above carrier
        distance_gear4 = torch.norm(gear4_pos[:, :2] - carrier_center[:, :2])  # XY distance
        if distance_gear4 < 0.05:
            score += 1
        
        # Score 5: Ring gear should be around the carrier assembly
        ring_pos = self.ring_gear.data.root_state_w[:, :3]
        distance_ring = torch.norm(ring_pos[:, :2] - planetary_carrier_pos[:, :2])  # XY distance
        if distance_ring < 0.05:
            score += 1
        
        # Score 6: Reducer should be on top of gear 4
        reducer_pos = self.planetary_reducer.data.root_state_w[:, :3]
        gear4_top = gear4_pos.clone()
        gear4_top[:, 2] += 0.03  # Reducer sits on top of gear 4
        distance_reducer = torch.norm(reducer_pos[:, :2] - gear4_top[:, :2])  # XY distance
        if distance_reducer < 0.05:
            score += 1
                
        return score

    def _initialize_scene(self):
        """Initialize physics materials for scene objects."""
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gripper_friction_coefficient,
            dynamic_friction=self.cfg.gripper_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)

        num_envs = self.scene.num_envs
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", "/World/Materials/gripper_material")  
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link2/collisions", "/World/Materials/gripper_material")

        gear_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gears_friction_coefficient,
            dynamic_friction=self.cfg.gears_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gear_material", gear_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/ring_gear/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_1/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_2/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_3/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_4/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_carrier/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_reducer/node_/mesh_", "/World/Materials/gear_material")
        
        table_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.table_friction_coefficient,
            dynamic_friction=self.cfg.table_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/table_material", table_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Table/table/body_whiteLarge", "/World/Materials/table_material")

    def _pre_assemble_gears_1_3(self):
        """Pre-position gears 1-3 on pins for Stage 2 curriculum."""
        # Get carrier position from initial_root_state
        carrier_state = self.initial_root_state["planetary_carrier"]
        carrier_pos = carrier_state[:, :3].clone()
        carrier_quat = carrier_state[:, 3:7].clone()
        
        num_envs = self.scene.num_envs
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, -1)
        
        gear_objects = [self.sun_planetary_gear_1, self.sun_planetary_gear_2, self.sun_planetary_gear_3]
        gear_names = ["sun_planetary_gear_1", "sun_planetary_gear_2", "sun_planetary_gear_3"]
        
        for gear_idx, (gear_obj, gear_name) in enumerate(zip(gear_objects, gear_names)):
            # Get the pin local position for this gear
            pin_local_pos = self.pin_local_positions[gear_idx]
            pin_local_pos_batched = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
            
            # Transform pin local position to world position
            _, pin_world_pos = torch_utils.tf_combine(
                carrier_quat, carrier_pos, 
                identity_quat, pin_local_pos_batched
            )
            
            # Create root state for this gear (position on pin)
            root_state = gear_obj.data.default_root_state.clone()
            root_state[:, :3] = pin_world_pos
            root_state[:, 3:7] = identity_quat  # Upright orientation
            
            # Write to simulation
            gear_obj.write_root_state_to_sim(root_state)
            
            # Store in initial_root_state
            self.initial_root_state[gear_name] = root_state.clone()
            print(f"[INFO] Pre-assembled {gear_name} on pin_{gear_idx} at {pin_world_pos[0]}")

    def _randomize_object_positions(self, object_list: list, object_names: list,
                              safety_margin: float = 0.02, max_attempts: int = 1000):
        """Randomize positions of objects on table without overlapping."""
        OBJECT_RADII = {
            'ring_gear': 0.1,
            'sun_planetary_gear_1': 0.035,
            'sun_planetary_gear_2': 0.035,
            'sun_planetary_gear_3': 0.035,
            'sun_planetary_gear_4': 0.035,
            'planetary_carrier': 0.07,
            'planetary_reducer': 0.04,
        }

        initial_root_state = {obj_name: torch.zeros((self.scene.num_envs, 7), device=self.device) for obj_name in object_names}
        num_envs = self.scene.num_envs
        placed_objects = [[] for _ in range(num_envs)]

        for obj_idx, obj in enumerate(object_list):
            obj_name = object_names[obj_idx]
            root_state = obj.data.default_root_state.clone()
            current_radius = OBJECT_RADII.get(obj_name, 0.05)

            for env_idx in range(num_envs):
                position_found = False

                for attempt in range(max_attempts):
                    x = torch.rand(1, device=self.device).item() * 0.2 + 0.3 + self.cfg.x_offset
                    y = torch.rand(1, device=self.device).item() * 0.8 - 0.4
                    z = 0.92

                    if obj_name == "planetary_carrier":
                        x = 0.4 + self.cfg.x_offset 
                        y = 0.0
                    elif obj_name == "sun_planetary_gear_1":
                        y = torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_2":
                        y = torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_3":
                        y = -torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_4":
                        y = -torch.rand(1, device=self.device).item() * 0.4

                    pos = torch.tensor([x, y, z], device=self.device)

                    is_valid = True
                    for placed_pos, placed_obj_name in placed_objects[env_idx]:
                        placed_radius = OBJECT_RADII.get(placed_obj_name, 0.05)
                        min_distance = current_radius + placed_radius + safety_margin
                        distance = torch.norm(pos[:2] - placed_pos[:2]).item()
                        if distance < min_distance:
                            is_valid = False
                            break

                    if is_valid:
                        root_state[env_idx, :3] = pos
                        placed_objects[env_idx].append((pos, obj_name))
                        position_found = True
                        break

                if not position_found:
                    print(f"[WARN] Could not find non-overlapping position for {obj_name} in env {env_idx}")
                    root_state[env_idx, :3] = pos
                    placed_objects[env_idx].append((pos, obj_name))

            obj.write_root_state_to_sim(root_state)
            initial_root_state[obj_name] = root_state.clone()

        return initial_root_state

    def _record_data(self):
        self.data_dict['/observations/head_rgb'].append(self.obs['head_rgb'].cpu().numpy().squeeze(0))
        self.data_dict['/observations/left_hand_rgb'].append(self.obs['left_hand_rgb'].cpu().numpy().squeeze(0))
        self.data_dict['/observations/right_hand_rgb'].append(self.obs['right_hand_rgb'].cpu().numpy().squeeze(0))
        
        self.data_dict['/observations/left_arm_joint_pos'].append(self.obs['left_arm_joint_pos'].cpu().numpy().squeeze(0))
        self.data_dict['/observations/right_arm_joint_pos'].append(self.obs['right_arm_joint_pos'].cpu().numpy().squeeze(0))
        self.data_dict['/observations/left_gripper_joint_pos'].append(self.obs['left_gripper_joint_pos'].cpu().numpy()[0].squeeze(0))
        self.data_dict['/observations/right_gripper_joint_pos'].append(self.obs['right_gripper_joint_pos'].cpu().numpy()[0].squeeze(0))
        
        self.data_dict['/actions/left_arm_action'].append(self.act['left_arm_action'].cpu().numpy().squeeze(0))
        self.data_dict['/actions/right_arm_action'].append(self.act['right_arm_action'].cpu().numpy().squeeze(0))
        self.data_dict['/actions/left_gripper_action'].append(self.act['left_gripper_action'].cpu().numpy()[0].squeeze(0))
        self.data_dict['/actions/right_gripper_action'].append(self.act['right_gripper_action'].cpu().numpy()[0].squeeze(0))

        self.data_dict['/score'].append(self.score)
        self.data_dict['/current_time'].append(self.rule_policy.count * self.sim.get_physics_dt())

    def _write_hdf5(self):
        with h5py.File(self.save_hdf5_file_name, 'w') as f:
            f.attrs['sim'] = True
            obs = f.create_group('observations')
            act = f.create_group('actions')
            num_items = len(self.data_dict['/observations/head_rgb'])
            obs.create_dataset('head_rgb', shape=(num_items, 240, 320, 3), dtype='uint8')
            obs.create_dataset('left_hand_rgb', shape=(num_items, 240, 320, 3), dtype='uint8')
            obs.create_dataset('right_hand_rgb', shape=(num_items, 240, 320, 3), dtype='uint8')
            obs.create_dataset('left_arm_joint_pos', shape=(num_items, 6), dtype='float32')
            obs.create_dataset('right_arm_joint_pos', shape=(num_items, 6), dtype='float32')
            obs.create_dataset('left_gripper_joint_pos', shape=(num_items, ), dtype='float32')
            obs.create_dataset('right_gripper_joint_pos', shape=(num_items, ), dtype='float32')
            act.create_dataset('left_arm_action', shape=(num_items, 6), dtype='float32')
            act.create_dataset('right_arm_action', shape=(num_items, 6), dtype='float32')
            act.create_dataset('left_gripper_action', shape=(num_items, ), dtype='float32')
            act.create_dataset('right_gripper_action', shape=(num_items, ), dtype='float32')
            
            f.create_dataset('score', shape=(num_items,), dtype='int32')
            f.create_dataset('current_time', shape=(num_items,), dtype='float32')

            for name, value in self.data_dict.items():
                f[name][...] = value
        
        self.data_dict = {
            '/observations/head_rgb': [],
            '/observations/left_hand_rgb': [],
            '/observations/right_hand_rgb': [],
            '/observations/left_arm_joint_pos': [],
            '/observations/right_arm_joint_pos': [],
            '/observations/left_gripper_joint_pos': [],
            '/observations/right_gripper_joint_pos': [],
            '/actions/left_arm_action': [],
            '/actions/right_arm_action': [],
            '/actions/left_gripper_action': [],
            '/actions/right_gripper_action': [],
            '/score': [],
            '/current_time': [],
        }

