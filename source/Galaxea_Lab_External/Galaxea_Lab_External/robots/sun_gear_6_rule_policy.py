"""Rule-based policy for Sun Gear 6 environment - Stage 2 (gears 4, ring_gear, reducer)."""

import torch
import math
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import carb.input
from carb.input import KeyboardEventType
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils


from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material

from isaaclab.sim import SimulationContext

class SunGear6RulePolicy:
    """Rule policy for Stage 2: Assemble gear 4, ring gear, and planetary reducer."""
    
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene,
    obj_dict: dict):
        self.sim = sim
        self.scene = scene
        self.device = sim.device

        self.obj_dict = obj_dict

        self.planetary_carrier = obj_dict["planetary_carrier"]
        self.ring_gear = obj_dict["ring_gear"]
        # Gears 1-3 are pre-assembled (on pins)
        self.sun_planetary_gear_1 = obj_dict["sun_planetary_gear_1"]
        self.sun_planetary_gear_2 = obj_dict["sun_planetary_gear_2"]
        self.sun_planetary_gear_3 = obj_dict["sun_planetary_gear_3"]
        # Gear 4 needs to be assembled
        self.sun_planetary_gear_4 = obj_dict["sun_planetary_gear_4"]
        self.planetary_reducer = obj_dict["planetary_reducer"]

        # Define pin positions (pins 0,1,2 occupied by gears 1-3)
        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0 - gear 1
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1 - gear 2
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2 - gear 3
        ]

        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003
        self.lifting_height = 0.2

        self.diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.get_config("left")
        self.diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.get_config("right")

        self.right_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids
        self.left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids

        self.initial_pos_left = torch.tensor([-20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
                                         -24.0 / 180.0 * math.pi, 17.8 / 180.0 * math.pi,
                                         38.7 / 180.0 * math.pi, 20.1 / 180.0 * math.pi], device=self.device)
        self.initial_pos_right = torch.tensor([-20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
                                         -22.0 / 180.0 * math.pi, -40.0 / 180.0 * math.pi,
                                         -67.6 / 180.0 * math.pi, 18.1 / 180.0 * math.pi], device=self.device)

        self.num_gripper_joints = None

        self.gear_to_pin_map = None
        
        self.current_target_position = None
        self.current_target_orientation = None
        
        self.current_target_joint_pos = None
        self.step_initial_joint_pos = None


        self.sim_dt = sim.get_physics_dt()
        print(f"sim_dt: {self.sim_dt}")
        self.count = 0

        # Time for initial stabilization
        self.time_step_0 = 0.2
        self.count_step_0 = int(self.time_step_0 / self.sim_dt)
        print(f"count_step_0: {self.count_step_0}")

        # Pick up gear 4
        self.time_step_1 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_1 = torch.cumsum(self.time_step_1, dim=0) + self.time_step_0
        self.count_step_1 = self.time_step_1 / self.sim_dt
        self.count_step_1 = self.count_step_1.int()
        print(f"count_step_1: {self.count_step_1}")

        # Mount gear 4 to the planetary_carrier (with rotation)
        self.time_step_2 = torch.tensor([0.0, 1.0, 1.0, 5.0, 0.5, 0.5], device=sim.device)
        self.time_step_2 = torch.cumsum(self.time_step_2, dim=0) + self.time_step_1[-1]
        self.count_step_2 = self.time_step_2 / self.sim_dt
        self.count_step_2 = self.count_step_2.int()
        print(f"count_step_2: {self.count_step_2}")

        # Reset arm
        self.time_step_3 = torch.tensor([0.0, 0.5], device=sim.device)
        self.time_step_3 = torch.cumsum(self.time_step_3, dim=0) + self.time_step_2[-1]
        self.count_step_3 = self.time_step_3 / self.sim_dt
        self.count_step_3 = self.count_step_3.int()
        print(f"count_step_3: {self.count_step_3}")

        # Pick up ring gear
        self.time_step_4 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_4 = torch.cumsum(self.time_step_4, dim=0) + self.time_step_3[-1]
        self.count_step_4 = self.time_step_4 / self.sim_dt
        self.count_step_4 = self.count_step_4.int()
        print(f"count_step_4: {self.count_step_4}")

        # Mount ring gear to the carrier (with rotation)
        self.time_step_5 = torch.tensor([0.0, 1.0, 1.0, 3.0, 0.5, 0.5], device=sim.device)
        self.time_step_5 = torch.cumsum(self.time_step_5, dim=0) + self.time_step_4[-1]
        self.count_step_5 = self.time_step_5 / self.sim_dt
        self.count_step_5 = self.count_step_5.int()
        print(f"count_step_5: {self.count_step_5}")

        # Pick up reducer
        self.time_step_6 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_6 = torch.cumsum(self.time_step_6, dim=0) + self.time_step_5[-1]
        self.count_step_6 = self.time_step_6 / self.sim_dt
        self.count_step_6 = self.count_step_6.int()
        print(f"count_step_6: {self.count_step_6}")

        # Mount reducer to gear 4
        self.time_step_7 = torch.tensor([0.0, 1.0, 1.0, 0.5, 0.5], device=sim.device)
        self.time_step_7 = torch.cumsum(self.time_step_7, dim=0) + self.time_step_6[-1]
        self.count_step_7 = self.time_step_7 / self.sim_dt
        self.count_step_7 = self.count_step_7.int()
        print(f"count_step_7: {self.count_step_7}")

        self.total_time_steps = self.count_step_7[-1]
        self.initial_root_state = None


    def set_initial_root_state(self, initial_root_state: dict):
        self.initial_root_state = initial_root_state.copy()


    def get_config(self, arm_name: str):
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )

        arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_arm_joint.*"], body_names=[f"{arm_name}_arm_link6"]
        )
        gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_gripper_axis1"]
        )

        arm_entity_cfg.resolve(self.scene)
        gripper_entity_cfg.resolve(self.scene)
        
        return diff_ik_controller, arm_entity_cfg, gripper_entity_cfg


    def move_robot_to_position(self,
                            arm_entity_cfg: SceneEntityCfg,
                            gripper_entity_cfg: SceneEntityCfg,
                            diff_ik_controller: DifferentialIKController,
                            target_position: torch.Tensor, target_orientation: torch.Tensor,
                            target_marker: VisualizationMarkers):
        robot = self.scene["robot"]

        arm_joint_ids = arm_entity_cfg.joint_ids
        arm_body_ids = arm_entity_cfg.body_ids

        gripper_joint_ids = gripper_entity_cfg.joint_ids
        self.num_gripper_joints = len(gripper_joint_ids)

        if robot.is_fixed_base:
            ee_jacobi_idx = arm_body_ids[0] - 1
        else:
            ee_jacobi_idx = arm_body_ids[0]

        ik_commands = torch.cat([target_position, target_orientation], dim=-1)
        diff_ik_controller.set_command(ik_commands)

        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, arm_entity_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_state_w[
            :, arm_body_ids[0], 0:7
        ]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        return joint_pos_des, arm_entity_cfg.joint_ids


    def prepare_mounting_plan(self):
        """Prepare mounting plan for Stage 2 parts (gear 4, ring gear, reducer)."""
        
        gear_names = ['sun_planetary_gear_4', 'ring_gear', 'planetary_reducer']

        left_ee_pos = self.scene["robot"].data.body_state_w[:, self.left_arm_entity_cfg.body_ids[0], 0:3]
        right_ee_pos = self.scene["robot"].data.body_state_w[:, self.right_arm_entity_cfg.body_ids[0], 0:3]

        self.gear_to_pin_map = {}

        for gear_name in gear_names:
            if gear_name not in self.initial_root_state:
                print(f"[WARN] {gear_name} not found in initial_root_state, skipping")
                continue

            gear_pos = self.initial_root_state[gear_name][:, :3].clone()
            env_idx = 0
            gear_pos_env = gear_pos[env_idx]

            if gear_pos_env[1] > 0.0:
                chosen_arm = 'left'
            else:
                chosen_arm = 'right'

            self.gear_to_pin_map[gear_name] = {
                'arm': chosen_arm,
                'pin': None,
                'pin_local_pos': None,
                'pin_world_pos': None,
                'pin_world_quat': None,
            }

            print(f"[INFO] {gear_name} -> {chosen_arm} arm")
            print(f"       Gear pos: {gear_pos_env}")

        return self.gear_to_pin_map


    def pick_up_target_gear(self,
                                    gear_name: str,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg,
                                    diff_ik_controller: DifferentialIKController):
        
        obj_height_offset = 0.0

        if gear_name == "sun_planetary_gear_4":
            root_state = self.initial_root_state["sun_planetary_gear_4"]
            obj_height_offset = 0.01

        elif gear_name == "ring_gear":
            ring_gear_pos = self.initial_root_state["ring_gear"][:, :3].clone()
            ring_gear_quat = self.initial_root_state["ring_gear"][:, 3:7].clone()
            local_pos = torch.tensor([0.0, 0.0, 0.0], device=self.sim.device).unsqueeze(0)
            target_orientation, target_position = torch_utils.tf_combine(
                ring_gear_quat, ring_gear_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
            )
            root_state = torch.cat([target_position, target_orientation], dim=-1)
            obj_height_offset = 0.030

        elif gear_name == "planetary_reducer":
            root_state = self.initial_root_state["planetary_reducer"]
            obj_height_offset = 0.05

        else:
            root_state = self.initial_root_state[gear_name]

        target_position = root_state[:, :3].clone()
        target_position[:, 2] = self.table_height + self.grasping_height + obj_height_offset
        target_position = target_position + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)
        
        target_orientation = root_state[:, 3:7].clone()
        target_orientation, target_position = torch_utils.tf_combine(
            target_orientation, target_position, 
            torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=self.sim.device)
        )

        action = None
        joint_ids = None

        if self.count >= count_step[0] and self.count < count_step[1]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position_h, target_orientation, None)
        
        gripper_joint_ids = gripper_entity_cfg.joint_ids

        if self.count >= count_step[1] and self.count < count_step[2]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position, target_orientation, None)

        if self.count >= count_step[2] and self.count < count_step[3]:
            action = torch.tensor([[0.0]], device=self.sim.device)
            joint_ids = gripper_joint_ids

        if self.count >= count_step[3] and self.count < count_step[4]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        return action, joint_ids


    def mount_gear_to_target(self,
                                    gear_name: str,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):

        obj_height_offset = 0.0
        mount_height_offset = 0.023

        if gear_name == "sun_planetary_gear_4":
            root_state = self.planetary_carrier.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
            obj_height_offset = 0.01
            mount_height_offset = 0.03

        elif gear_name == "planetary_reducer":
            root_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
                self.current_target_orientation = root_state[:, 3:7].clone()
            obj_height_offset = 0.023 + 0.02
            mount_height_offset = 0.025

        target_position = self.current_target_position.clone()
        target_position[:, 2] = self.table_height + self.grasping_height
        target_position[:, 2] += obj_height_offset
        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)
        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device)

        if gear_name == "planetary_reducer":
            target_orientation, target_position = torch_utils.tf_combine(
                self.current_target_orientation, target_position, 
                torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=self.sim.device)
            )

        target_position_h_down = target_position + torch.tensor([0.0, 0.0, mount_height_offset], device=self.sim.device)

        action = None
        joint_ids = None

        if self.count >= count_step[0] and self.count < count_step[1]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        if self.count >= count_step[1] and self.count < count_step[2]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h_down, target_orientation, None)

        if self.count >= count_step[2] and self.count < count_step[3]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            num_gripper_joints = len(gripper_joint_ids)
            gripper_joint_pos_des = torch.full((num_gripper_joints,), 0.04, device=self.device)
            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids
            
        if self.count >= count_step[3] and self.count < count_step[4]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        return action, joint_ids


    def mount_gear_to_target_and_rotate(self,
                                    gear_name: str,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):
        """Mount gear with rotation to fit."""

        if gear_name == "sun_planetary_gear_4":
            root_state = self.planetary_carrier.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
            obj_height_offset = 0.01
            mount_height_offset = 0.040
            rot_deg = 60

        elif gear_name == "ring_gear":
            planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()
            local_pos = torch.tensor([0.0, 0.0, 0.0], device=self.sim.device).unsqueeze(0)
            if self.count == count_step[0]:
                self.current_target_orientation, self.current_target_position = torch_utils.tf_combine(
                    planetary_carrier_quat, planetary_carrier_pos, 
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
                )
            obj_height_offset = 0.01
            mount_height_offset = 0.025 + 0.028
            rot_deg = 30

        target_position = self.current_target_position.clone()
        target_position[:, 2] = self.table_height + self.grasping_height + obj_height_offset
        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)
        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device)
        target_position_h_down = target_position + torch.tensor([0.0, 0.0, mount_height_offset], device=self.sim.device)

        action = None
        joint_ids = None

        if self.count >= count_step[0] and self.count < count_step[1]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        if self.count >= count_step[1] and self.count < count_step[2]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h_down, target_orientation, None)

        # Rotate to fit
        if self.count >= count_step[2] and self.count < count_step[3]:
            joint_ids = arm_entity_cfg.joint_ids
            delta_rot_rad = rot_deg / (count_step[3] - count_step[2]) * torch.pi / 180.0
            if self.count == count_step[2]:
                joint_pos = self.scene["robot"].data.joint_pos.clone()
                self.step_initial_joint_pos = joint_pos[:, joint_ids].clone()
            
            self.current_target_joint_pos = self.step_initial_joint_pos.clone()
            self.current_target_joint_pos[:, 5] += delta_rot_rad * (self.count - count_step[2] + 5)
            action = self.current_target_joint_pos

        if self.count >= count_step[3] and self.count < count_step[4]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            num_gripper_joints = len(gripper_joint_ids)
            gripper_joint_pos_des = torch.full((num_gripper_joints,), 0.04, device=self.device)
            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids
            
        if self.count >= count_step[4] and self.count < count_step[5]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        return action, joint_ids


    def get_action(self):
        action = None
        joint_ids = None

        # Initial stabilization
        if self.count < self.count_step_0:
            action = torch.cat([self.initial_pos_left, self.initial_pos_right], dim=0).unsqueeze(0)
            joint_ids = self.left_arm_entity_cfg.joint_ids + self.right_arm_entity_cfg.joint_ids

        # Pick up gear 4
        if self.count >= self.count_step_1[0] and self.count < self.count_step_1[-1]:
            gear_name = "sun_planetary_gear_4"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_name, self.count_step_1, current_arm, current_gripper, self.diff_ik_controller)

        # Mount gear 4 to carrier (with rotation)
        if self.count >= self.count_step_2[0] and self.count < self.count_step_2[-1]:
            gear_name = "sun_planetary_gear_4"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_name, self.count_step_2, current_arm, current_gripper)

        # Reset arm
        if self.count >= self.count_step_3[0] and self.count < self.count_step_3[-1]:
            gear_name = "sun_planetary_gear_4"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                action = self.initial_pos_left.unsqueeze(0)
                joint_ids = self.left_arm_entity_cfg.joint_ids
            else:
                action = self.initial_pos_right.unsqueeze(0)
                joint_ids = self.right_arm_entity_cfg.joint_ids

        # Pick up ring gear
        if self.count >= self.count_step_4[0] and self.count < self.count_step_4[-1]:
            gear_name = "ring_gear"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_name, self.count_step_4, current_arm, current_gripper, self.diff_ik_controller)

        # Mount ring gear (with rotation)
        if self.count >= self.count_step_5[0] and self.count < self.count_step_5[-1]:
            gear_name = "ring_gear"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_name, self.count_step_5, current_arm, current_gripper)

        # Pick up reducer
        if self.count >= self.count_step_6[0] and self.count < self.count_step_6[-1]:
            gear_name = "planetary_reducer"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_name, self.count_step_6, current_arm, current_gripper, self.diff_ik_controller)

        # Mount reducer to gear 4
        if self.count >= self.count_step_7[0] and self.count < self.count_step_7[-1]:
            gear_name = "planetary_reducer"
            current_arm_str = self.gear_to_pin_map[gear_name]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target(gear_name, self.count_step_7, current_arm, current_gripper)

        return action, joint_ids
