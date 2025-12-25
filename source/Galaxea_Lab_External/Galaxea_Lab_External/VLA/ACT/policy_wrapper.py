#!/usr/bin/env python3
"""
Standard Policy Wrapper Interface for Competition

This module provides a standardized interface for deploying different policy types
in the Isaac Lab environment. All competition participants should implement this
interface for their policies.

Author: Competition Organizers
Date: 2025-12-10
"""

import torch
import numpy as np
import h5py
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import sys
import os
import time
import pickle
import threading
import io
import grpc
from queue import Queue, Empty
from dataclasses import dataclass
from types import ModuleType

# Add LeRobot transport to path to import generated protobufs
lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

try:
    from lerobot.transport import services_pb2
    from lerobot.transport import services_pb2_grpc
except ImportError:
    # Try relative path backup
    lerobot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../lerobot/src'))
    if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
        sys.path.append(lerobot_path)
    
    try:
        from lerobot.transport import services_pb2
        from lerobot.transport import services_pb2_grpc
    except ImportError:
        print("⚠ Warning: Could not import LeRobot transport. RemotePolicyWrapper will not work.")

# --- Local Transport Utilities (to avoid heavy LeRobot deps) ---
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB

def send_bytes_in_chunks(buffer: bytes, message_class: Any):
    """Generator to split bytes into gRPC messages."""
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0, io.SEEK_END)
    size_in_bytes = bytes_buffer.tell()
    bytes_buffer.seek(0)
    
    sent_bytes = 0
    while sent_bytes < size_in_bytes:
        transfer_state = services_pb2.TransferState.TRANSFER_MIDDLE
        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = services_pb2.TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = services_pb2.TransferState.TRANSFER_BEGIN
            
        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)
        
        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read

def receive_bytes_in_chunks(iterator) -> bytes:
    """Reassemble bytes from gRPC message stream."""
    bytes_buffer = io.BytesIO()
    
    for item in iterator:
        if item.transfer_state == services_pb2.TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
        elif item.transfer_state == services_pb2.TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
        elif item.transfer_state == services_pb2.TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            return bytes_buffer.getvalue()
    
    return bytes_buffer.getvalue()

# --- Pickle Compatibility Mocking ---
# We mock the module structure so pickle finds our local classes when looking up 
# 'lerobot.async_inference.helpers.<ClassName>'
mock_module_name = "lerobot.async_inference.helpers"

def _mock_module(name):
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)
    return sys.modules[name]

# Ensure parent modules exist
_mock_module("lerobot")
_mock_module("lerobot.async_inference")
fake_helpers = _mock_module(mock_module_name)

@dataclass
class RemotePolicyConfig:
    """Minimal config to send to server."""
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict | None = None

RemotePolicyConfig.__module__ = mock_module_name
fake_helpers.RemotePolicyConfig = RemotePolicyConfig

@dataclass
class TimedData:
    timestamp: float
    timestep: int
    
    def get_timestamp(self): return self.timestamp
    def get_timestep(self): return self.timestep

TimedData.__module__ = mock_module_name
fake_helpers.TimedData = TimedData

@dataclass
class TimedObservation(TimedData):
    observation: dict
    must_go: bool = False
    
    def get_observation(self): return self.observation

TimedObservation.__module__ = mock_module_name
fake_helpers.TimedObservation = TimedObservation

@dataclass
class TimedAction(TimedData):
    action: torch.Tensor
    
    def get_action(self): return self.action

TimedAction.__module__ = mock_module_name
fake_helpers.TimedAction = TimedAction



class PolicyWrapper(ABC):
    """
    Abstract base class for policy wrappers.
    
    All policies must implement this interface to be compatible with the
    standardized deployment and evaluation pipeline.
    
    Key Features:
    - Unified predict() interface
    - Optional control frequency specification
    - Automatic device management
    - Clean separation from environment code
    """
    
    @abstractmethod
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Predict actions based on current state and images.
        
        Args:
            qpos: Joint positions, shape (batch_size, state_dim)
                  For dual-arm robot: state_dim = 14 (7 joints per arm)
            images: Camera images, shape (batch_size, num_cameras, C, H, W)
                    Default: (1, 3, 3, 240, 320) for 3 RGB cameras
        
        Returns:
            actions: Predicted actions, shape (batch_size, action_dim)
                    For dual-arm: action_dim = 14
        
        Note:
            - All tensors should be on the same device (cuda/cpu)
            - Images are expected to be normalized to [0, 1]
            - Implementation should handle batch processing
        """
        pass
    
    @property
    def required_control_frequency(self) -> Optional[float]:
        """
        Specify the required control frequency in Hz.
        
        Returns:
            float: Required frequency in Hz (e.g., 50.0 for ACT)
            None: Use environment default (20 Hz, decimation=5)
        
        The deployment script will automatically adjust environment
        decimation to match this frequency.
        """
        return None
    
    @property
    def camera_names(self) -> list:
        """
        Specify required camera names in order.
        
        Returns:
            List of camera names, e.g., ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
        """
        return ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
    
    @property
    def device(self) -> torch.device:
        """Get the device (cuda/cpu) used by the policy."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACTPolicyWrapper(PolicyWrapper):
    """
    Wrapper for ACT (Action Chunking with Transformers) policy.
    
    This wrapper demonstrates how to integrate ACT with the standardized interface.
    It handles:
    - ACT model loading with modified argparse
    - Temporal aggregation for smooth action execution
    - Proper observation format conversion
    """
    
    def __init__(self, checkpoint_path: str, temporal_agg: bool = True):
        """
        Initialize ACT policy wrapper.
        
        Args:
            checkpoint_path: Path to ACT checkpoint file (.ckpt)
            temporal_agg: Whether to use temporal aggregation (recommended)
        """
        self.checkpoint_path = checkpoint_path
        self.temporal_agg = temporal_agg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Temporal aggregation state
        self.all_time_actions = None
        self.timestep = 0
        self.num_queries = 100  # ACT chunk size
        
        # Normalization statistics (will be loaded from dataset_stats.pkl)
        self.action_mean = None
        self.action_std = None
        self.qpos_mean = None
        self.qpos_std = None
        
        # Delayed import to avoid argparse conflicts
        self._load_act_policy()
        
    def _load_act_policy(self):
        """Load ACT policy with sys.argv isolation."""
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # Clear sys.argv to prevent argparse conflicts
            sys.argv = ['policy_wrapper.py']
            
            # Add ACT to path
            act_dir = os.path.join(os.path.dirname(__file__), '..', 'act')
            if act_dir not in sys.path:
                sys.path.insert(0, act_dir)
                sys.path.insert(0, os.path.join(act_dir, 'detr'))
            
            # Import ACT modules
            from .act.policy import ACTPolicy
            from .act.constants import DT
            
            self.act_dt = DT  # Should be 0.02s (50Hz)
            
            # ACT configuration matching training
            policy_config = {
                'num_queries': 100,
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': 1e-5,
                'backbone': 'resnet18',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': ['head_rgb', 'left_hand_rgb', 'right_hand_rgb'],
            }
            
            # Create and load policy
            self.policy = ACTPolicy(policy_config)
            ckpt = torch.load(self.checkpoint_path, map_location=self._device)
            self.policy.load_state_dict(ckpt)
            self.policy.to(self._device)
            self.policy.eval()
            
            # Load normalization statistics
            import pickle
            stats_path = os.path.join(os.path.dirname(self.checkpoint_path), 'dataset_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                self.action_mean = torch.from_numpy(stats['action_mean']).float().to(self._device)
                self.action_std = torch.from_numpy(stats['action_std']).float().to(self._device)
                
                # Also load qpos stats if available
                if 'qpos_mean' in stats:
                    self.qpos_mean = torch.from_numpy(stats['qpos_mean']).float().to(self._device)
                    self.qpos_std = torch.from_numpy(stats['qpos_std']).float().to(self._device)
                    print(f"  - Qpos mean range: [{self.qpos_mean.min():.3f}, {self.qpos_mean.max():.3f}]")
                    print(f"  - Qpos std range: [{self.qpos_std.min():.3f}, {self.qpos_std.max():.3f}]")
                else:
                    self.qpos_mean = None
                    self.qpos_std = None
                    print(f"⚠ Warning: No qpos stats found in dataset_stats.pkl")

                print(f"✓ Loaded normalization stats from {stats_path}")
                print(f"  - Action mean range: [{self.action_mean.min():.3f}, {self.action_mean.max():.3f}]")
                print(f"  - Action std range: [{self.action_std.min():.3f}, {self.action_std.max():.3f}]")
            else:
                print(f"⚠ Warning: No dataset_stats.pkl found at {stats_path}")
                print(f"  Actions will NOT be denormalized - this may cause issues!")
            
            print(f"✓ ACT policy loaded successfully")
            print(f"  - Checkpoint: {self.checkpoint_path}")
            print(f"  - Parameters: {sum(p.numel() for p in self.policy.parameters()) / 1e6:.2f}M")
            print(f"  - Chunk size: {self.num_queries}")
            print(f"  - Temporal aggregation: {self.temporal_agg}")
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
    
    @property
    def required_control_frequency(self) -> float:
        """ACT requires 20 Hz control frequency (modified for user setup)."""
        return 20.0
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Predict action using ACT policy.
        
        Args:
            qpos: Joint positions (batch_size, 14)
            images: Camera images (batch_size, 3, 3, 240, 320)
        
        Returns:
            actions: Predicted action (batch_size, 14) - DENORMALIZED to real joint positions
        """
        with torch.no_grad():
            # Normalize qpos
            if self.qpos_mean is not None and self.qpos_std is not None:
                qpos = (qpos - self.qpos_mean) / self.qpos_std
                
            # Get action chunk from policy (normalized outputs)
            all_actions = self.policy(qpos, images)  # (batch_size, num_queries, 14)
            
            if self.temporal_agg:
                # Use temporal aggregation for smooth execution
                action = self._temporal_aggregation(all_actions)
            else:
                # Just use first action from chunk
                action = all_actions[:, 0, :]
            
            # Denormalize action to real joint positions
            if self.action_mean is not None and self.action_std is not None:
                action = action * self.action_std + self.action_mean
            
            self.timestep += 1
            
        return action
    
    def _temporal_aggregation(self, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal aggregation with exponential weighting.
        
        Args:
            all_actions: Action chunk (batch_size, num_queries, action_dim)
        
        Returns:
            Aggregated action (batch_size, action_dim)
        """
        batch_size = all_actions.shape[0]
        action_dim = all_actions.shape[2]
        
        # Initialize buffer if needed
        if self.all_time_actions is None:
            max_timesteps = 3000
            self.all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + self.num_queries, action_dim],
                device=self._device
            )
        
        # Store current action chunk
        self.all_time_actions[self.timestep, self.timestep:self.timestep + self.num_queries] = all_actions[0]
        
        # Exponential weighting: more recent predictions have higher weight
        actions_for_curr_step = self.all_time_actions[:self.timestep + 1, self.timestep]
        weights = torch.exp(-0.1 * torch.arange(self.timestep + 1, device=self._device).flip(0))
        weights = weights / weights.sum()
        
        # Weighted average
        action = (actions_for_curr_step.T @ weights).unsqueeze(0)
        
        return action
    
    def reset(self):
        """Reset temporal aggregation buffers for new episode."""
        self.all_time_actions = None
        self.timestep = 0


# Example: Placeholder for other policy types
class DiffusionPolicyWrapper(PolicyWrapper):
    """Placeholder for Diffusion Policy wrapper."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError("Diffusion Policy wrapper not yet implemented")
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BCPolicyWrapper(PolicyWrapper):
    """Placeholder for Behavior Cloning policy wrapper."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError("BC Policy wrapper not yet implemented")
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    """Test ACT policy wrapper loading and inference."""
    
    print("Testing ACT Policy Wrapper...")
    print("=" * 60)
    
    # Test checkpoint path
    checkpoint_path = "act/ckpt/policy_best.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please provide correct checkpoint path")
        exit(1)
    
    # Load policy
    print("\nLoading ACT policy...")
    wrapper = ACTPolicyWrapper(checkpoint_path, temporal_agg=True)
    
    # Test inference
    print("\nTesting inference...")
    batch_size = 1
    state_dim = 14
    num_cameras = 3
    
    # Create dummy inputs
    qpos = torch.randn(batch_size, state_dim).to(wrapper.device)
    images = torch.rand(batch_size, num_cameras, 3, 240, 320).to(wrapper.device)
    
    print(f"Input shapes:")
    print(f"  - qpos: {qpos.shape}")
    print(f"  - images: {images.shape}")
    
    # Predict
    action = wrapper.predict(qpos, images)
    
    print(f"\nOutput:")
    print(f"  - action shape: {action.shape}")
    print(f"  - action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    
    # Test properties
    print(f"\nPolicy properties:")
    print(f"  - Required frequency: {wrapper.required_control_frequency} Hz")
    print(f"  - Camera names: {wrapper.camera_names}")
    print(f"  - Device: {wrapper.device}")
    
    print("\n✓ All tests passed!")


class DataReplayPolicyWrapper(PolicyWrapper):
    """
    Data Replay Policy - replays actions from recorded demonstration data.
    
    This policy wrapper reads actions from an HDF5 file and replays them step by step.
    Useful for:
    1. Debugging deployment pipeline (excludes model performance issues)
    2. Verifying environment behavior matches data collection
    3. Testing action application without model inference
    
    The replay uses a simple step counter to index into the action array.
    """
    
    def __init__(
        self,
        data_path: str,
        device: str = "cuda",
        camera_names: Optional[list] = None,
    ):
        """
        Initialize data replay policy.
        
        Args:
            data_path: Path to HDF5 file containing recorded actions
            device: Device to use for tensors
            camera_names: Camera names (for compatibility with interface)
        """
        self.device = device
        self._camera_names = camera_names or ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
        
        print(f"\n{'='*80}")
        print("Initializing Data Replay Policy")
        print(f"{'='*80}")
        print(f"Data path: {data_path}")
        
        # Load action data from HDF5
        with h5py.File(data_path, 'r') as f:
            # Read all action components
            left_arm = f['/actions/left_arm_action'][:]  # (N, 6)
            right_arm = f['/actions/right_arm_action'][:]  # (N, 6)
            left_gripper = f['/actions/left_gripper_action'][:]  # (N,)
            right_gripper = f['/actions/right_gripper_action'][:]  # (N,)
            
            # Combine into 14-dim actions: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
            self.actions = np.concatenate([
                left_arm,  # (N, 6)
                right_arm,  # (N, 6)
                left_gripper[:, np.newaxis],  # (N, 1)
                right_gripper[:, np.newaxis],  # (N, 1)
            ], axis=1)  # (N, 14)
            
            # Also load observations for reference
            self.qpos_data = f['/observations/left_arm_joint_pos'][:] if '/observations/left_arm_joint_pos' in f else None
            
            print(f"\nLoaded data:")
            print(f"  - Total timesteps: {len(self.actions)}")
            print(f"  - Action shape: {self.actions.shape}")
            print(f"  - Action range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")
            
            # Statistics
            print(f"\nAction statistics per joint:")
            for i in range(14):
                joint_name = self._get_joint_name(i)
                mean_val = self.actions[:, i].mean()
                std_val = self.actions[:, i].std()
                min_val = self.actions[:, i].min()
                max_val = self.actions[:, i].max()
                print(f"  {joint_name:20s}: mean={mean_val:6.3f}, std={std_val:5.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")
        
        # Convert to torch tensor
        self.actions = torch.from_numpy(self.actions).float().to(self.device)
        
        # Step counter for indexing into action array
        self.step_idx = 0
        self.max_steps = len(self.actions)
        
        print(f"\n{'='*80}")
        print("✓ Data Replay Policy initialized successfully")
        print(f"{'='*80}\n")
    
    def _get_joint_name(self, idx: int) -> str:
        """Get human-readable joint name for index."""
        names = [
            "left_arm_joint_1",
            "left_arm_joint_2", 
            "left_arm_joint_3",
            "left_arm_joint_4",
            "left_arm_joint_5",
            "left_arm_joint_6",
            "right_arm_joint_1",
            "right_arm_joint_2",
            "right_arm_joint_3", 
            "right_arm_joint_4",
            "right_arm_joint_5",
            "right_arm_joint_6",
            "left_gripper",
            "right_gripper",
        ]
        return names[idx]
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Return the next action from recorded data.
        
        Args:
            qpos: Current joint positions (not used in replay, but kept for interface compatibility)
            images: Current camera images (not used in replay)
        
        Returns:
            action: Next action from recorded trajectory, shape (1, 14)
        """
        # Get current action from pre-loaded data
        if self.step_idx >= self.max_steps:
            print(f"[WARNING] Replay finished! Reached end of data at step {self.step_idx}/{self.max_steps}")
            print(f"          Looping back to beginning...")
            self.step_idx = 0
        
        action = self.actions[self.step_idx].unsqueeze(0)  # (1, 14)
        
        # Debug output every 50 steps
        if self.step_idx % 50 == 0:
            print(f"\n[Replay Step {self.step_idx}/{self.max_steps}]")
            print(f"  Left arm action:  {action[0, :6].cpu().numpy()}")
            print(f"  Right arm action: {action[0, 6:12].cpu().numpy()}")
            print(f"  Grippers: L={action[0, 12].item():.3f}, R={action[0, 13].item():.3f}")
        
        self.step_idx += 1
        return action
    
    @property
    def camera_names(self) -> list:
        """Return camera names."""
        return self._camera_names
    
    def reset(self):
        """Reset replay to beginning of trajectory."""
        print(f"\n[Data Replay] Resetting to step 0")
        self.step_idx = 0


class LeRobotPolicyWrapper(PolicyWrapper):
    """
    Wrapper for LeRobot policies (ACT, Diffusion, etc.) trained via LeRobot.
    """
    
    def __init__(self, checkpoint_path: str, policy_type: str = None):
        """
        Initialize LeRobot policy wrapper.
        
        Args:
            checkpoint_path: Path to local checkpoint folder or HF Hub ID
            policy_type: Optional policy type override (if not provided, read from config)
        """
        self.checkpoint_path = checkpoint_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add LeRobot to path if needed (assuming standard location)
        # Using absolute path based on workspace structure
        # /home/hls/codes/gearboxAssembly/lerobot/src
        lerobot_path = "/home/hls/codes/gearboxAssembly/lerobot/src"
        if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
            sys.path.append(lerobot_path)
            
        try:
            from lerobot.policies.factory import make_policy, make_pre_post_processors, get_policy_class
            from lerobot.policies.pretrained import PreTrainedPolicy
            from lerobot.processor import PolicyAction
        except ImportError:
            print(f"⚠ Could not import LeRobot from {lerobot_path}. Trying relative path...")
            # Try relative path backup
            lerobot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../lerobot/src'))
            if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
                sys.path.append(lerobot_path)
                from lerobot.policies.factory import make_policy, make_pre_post_processors, get_policy_class
                from lerobot.policies.pretrained import PreTrainedPolicy
                from lerobot.processor import PolicyAction
            else:
                raise ImportError("Could not find or import LeRobot library.")

        print(f"\nLoading LeRobot policy from: {checkpoint_path}")
        
        # Load policy
        # 1. Infer type from config.json if not provided
        if not policy_type:
            config_path = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                policy_type = config_dict.get('type')
        
        if not policy_type:
            raise ValueError("Could not infer policy type from config.json and none provided.")

        # 2. Get policy class and load
        try:
            policy_cls = get_policy_class(policy_type)
        except ValueError:
             raise ValueError(f"Unsupported LeRobot policy type: {policy_type}")

        self.policy = policy_cls.from_pretrained(checkpoint_path)
        self.policy.to(self._device)
        self.policy.eval()
        
        # 3. Create processors (normalization, etc.)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint_path
        )
        
        # Move processors to device if possible
        if hasattr(self.preprocessor, "to"):
            self.preprocessor.to(self._device)
        
        # 4. Determine camera names/mapping
        self._camera_names = []
        self._camera_key_mapping = {}  # Map env_name -> policy_key
        
        # Mappings from LeRobot keys to standard Environment keys
        # Env keys: 'head_rgb', 'left_hand_rgb', 'right_hand_rgb'
        key_mapping_heuristics = {
            "head": "head_rgb",
            "top": "head_rgb",
            "phone": "head_rgb",
            "left_hand": "left_hand_rgb",
            "left_wrist": "left_hand_rgb",
            "right_hand": "right_hand_rgb",
            "right_wrist": "right_hand_rgb"
        }
        
        if self.policy.config.image_features:
            for key in self.policy.config.image_features:
                # key e.g. "observation.images.head_camera"
                parts = key.split('.')
                cam_name = parts[-1]  # "head_camera"
                
                # Find matching standard name
                found_match = False
                for h_key, standard_name in key_mapping_heuristics.items():
                    if h_key in cam_name:
                        self._camera_names.append(standard_name)
                        self._camera_key_mapping[standard_name] = key
                        found_match = True
                        break
                
                if not found_match:
                    print(f"⚠ Warning: Could not automatically map policy camera feature '{key}' to standard env cameras.")
                    print(f"  Available standard env cameras: {list(set(key_mapping_heuristics.values()))}")
        
        # Sort for consistency
        self._camera_names.sort()
        
        print("\n✓ LeRobot policy loaded successfully")
        print(f"  - Type: {policy_type}")
        print(f"  - Input Keys: {list(self.policy.config.input_features.keys())}")
        print(f"  - Mapped Cameras: {self._camera_names}")
        print(f"  - FPS: {self.required_control_frequency}")

    @property
    def required_control_frequency(self) -> Optional[float]:
        """Get required control frequency from config if available."""
        if hasattr(self.policy.config, "fps") and self.policy.config.fps:
            return float(self.policy.config.fps)
        return 20.0  # Default to 20Hz (match training data) if unknown

    @property
    def camera_names(self) -> list:
        return self._camera_names

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Predict action using LeRobot policy.
        
        Args:
            qpos: (B, 14)
            images: (B, N, 3, H, W) in order of self.camera_names
        """
        # 1. Construct input batch
        batch = {}
        
        # State
        if "observation.state" in self.policy.config.input_features:
            batch["observation.state"] = qpos
            
        # Images
        for i, env_cam_name in enumerate(self.camera_names):
            policy_key = self._camera_key_mapping[env_cam_name]
            batch[policy_key] = images[:, i] # (B, 3, H, W)
            
        # 2. Preprocess (Normalize)
        batch = self.preprocessor(batch)
        
        # 3. Inference
        with torch.no_grad():
            action = self.policy.select_action(batch)
            
        # 4. Postprocess (Unnormalize)
        action = self.postprocessor(action)
        
        return action

    def reset(self):
        self.policy.reset()


class RemotePolicyWrapper(PolicyWrapper):
    """
    Client wrapper that talks to a remote LeRobot Policy Server.
    Decouples the simulator (Python 3.11) from the policy (Python 3.10).
    """
    
    def __init__(
        self, 
        host: str = "127.0.0.1", 
        port: int = 8080, 
        policy_type: str = "lerobot",
        checkpoint: str = None,
        lerobot_policy_type: str = None
    ):
        self.server_address = f"{host}:{port}"
        self._device = torch.device("cpu") # Communication handles device properties
        
        # Connect to server
        print(f"\nconnecting to Policy Server at {self.server_address}...")
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        try:
            # Handshake
            self.stub.Ready(services_pb2.Empty())
            print("✓ Connected to server")
        except grpc.RpcError as e:
            print(f"❌ Failed to connect to server: {e}")
            raise ConnectionError(f"Could not connect to Policy Server at {self.server_address}")

        # Send policy config
        # We need to map our environment features to LeRobot features
        # Note: In a full implementation, we'd inspect the robot/env config.
        # Here we hardcode common defaults or use the provided types.
        
        # Heuristic: Map standard camera names to assumed LeRobot features
        # This mirrors 'map_robot_keys_to_lerobot_features' logic:
        # We need to define "observation.state" with a list of "names" corresponding to the keys in raw_obs
        
        # 1. Define the joint names we will send in raw_obs
        # We must simply list 14 scalar names so build_dataset_frame stacks them into a (14,) vector.
        state_names = [f"joint_{i}" for i in range(14)]
        
        self.lerobot_features = {
            "observation.images.head": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
            "observation.images.left_wrist": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
            "observation.images.right_wrist": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
            "observation.state": {
                "dtype": "float32", 
                "shape": (14,), 
                "names": state_names
            }
        }
        
        # If specific policy type passed, use it, else default to 'act' or similar logic
        # For simplicity, we assume 'lerobot' generally or pass the subtype
        actual_type = lerobot_policy_type if lerobot_policy_type else (policy_type if policy_type != 'lerobot' else 'act')
        
        policy_config = RemotePolicyConfig(
            policy_type=actual_type,
            pretrained_name_or_path=checkpoint,
            lerobot_features=self.lerobot_features, # Full implementation might need precise mapping
            actions_per_chunk=200, # Default
            device="cuda", # Server should use cuda
            rename_map={}
        )
        
        print(f"Sending policy config: {actual_type} from {checkpoint}")
        config_bytes = pickle.dumps(policy_config)
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=config_bytes))
        
        self.timestep = 0
        
        # Action queue for async/temporal behavior
        self.action_queue = Queue()
        self.latest_action_step = -1
        
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Send observation to server, get action back.
        This blocks for simplicity in this wrapper, but server is async capable.
        """
        # 1. Prepare observation dict for transport
        # The keys here must match the "names" list in self.lerobot_features["observation.state"]
        
        raw_obs = {}
        
        # Qpos: Decompose into 14 scalars
        # Assumes qpos is (B, 14), we take first batch item B=0
        qpos_cpu = qpos[0].cpu() # (14,)
        for i in range(14):
            raw_obs[f"joint_{i}"] = qpos_cpu[i].item() # Send as python float
            
        # Images: [head, left_hand, right_hand]
        # We needs to send (H, W, C) for server helper compatibility (it does permute(2,0,1))
        # Input images is (B, N, C, H, W). We take B=0.
        
        mapping = {
            'head_rgb': 'head',
            'left_hand_rgb': 'left_wrist',
            'right_hand_rgb': 'right_wrist'
        }
        
        for i, env_name in enumerate(self.camera_names):
             # Extract (C, H, W) -> Permute to (H, W, C) -> CPU
             img_tensor = images[0, i].permute(1, 2, 0).cpu() # (H, W, C)
             
             if env_name in mapping:
                 raw_obs[mapping[env_name]] = img_tensor
             else:
                 # Fallback
                 raw_obs[env_name] = img_tensor
             
        # Add task name if needed (dummy here)
        raw_obs["task"] = "remote_deployment"
        
        # Create TimedObservation
        obs_obj = TimedObservation(
            timestamp=time.time(),
            observation=raw_obs,
            timestep=self.timestep,
            must_go=True # Force inference for this step
        )
        
        # 2. Send Observation
        obs_bytes = pickle.dumps(obs_obj)
        obs_iter = send_bytes_in_chunks(
            obs_bytes, 
            services_pb2.Observation
        )
        self.stub.SendObservations(obs_iter)
        
        # 3. Request Actions
        # In a true async loop, we'd poll. Here we block until we get the action for THIS timestep.
        # But the server sends chunks. We might get a chunk covering [t, t+50].
        
        # Simple blocking retry loop
        action_tensor = None
        max_retries = 100
        
        # Check local queue first
        action_tensor = self._get_from_queue(self.timestep)
        
        if action_tensor is None:
            # Poll server
            for _ in range(max_retries):
                response = self.stub.GetActions(services_pb2.Empty())
                if len(response.data) > 0:
                    # Parse actions
                    # returns list[TimedAction]
                                # GetActions returns a single Actions message with the complete pickled data
                    actions = pickle.loads(response.data)
                    
                    # Add to queue
                    for ta in actions:
                         # Merge/Update logic: simply overwrite or add
                         self.action_queue.put(ta)
                    
                    # Try getting again
                    action_tensor = self._get_from_queue(self.timestep)
                    if action_tensor is not None:
                        break
                
                time.sleep(0.01)
                
        if action_tensor is None:
            print(f"Warning: Timed out waiting for action at step {self.timestep}")
            # Fail-safe: reuse last action or zero? 
            # For now return zeros to avoid crash, but robot will stop
            action_tensor = torch.zeros(1, 14, device=qpos.device)
            
        self.timestep += 1
        return action_tensor.to(qpos.device)

    def _get_from_queue(self, timestep):
        # Scan queue (inefficient but safe for simple wrapper)
        # We need to look for exact timestep match
        
        # NOTE: The queue might contain old actions. We should clean up?
        # For this prototype, we just search.
        
        # Snapshot current queue
        temp_list = []
        found_action = None
        
        while not self.action_queue.empty():
            item = self.action_queue.get()
            if item.timestep == timestep:
                found_action = item.action
                # We found it! We can discard older actions? 
                # Ideally yes, but let's keep future ones.
            
            if item.timestep >= timestep:
                temp_list.append(item)
                
        # Put back future items
        for item in temp_list:
            self.action_queue.put(item)
            
        return found_action

    @property
    def required_control_frequency(self) -> Optional[float]:
        return 20.0 # Assumed default for now

