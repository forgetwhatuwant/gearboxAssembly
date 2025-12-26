"""
Professional-grade data recording utilities for Isaac Lab environments.
Provides robust validation, NaN filtering, episode tracking, and HDF5 management.
"""

import numpy as np
import torch
import h5py
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RecordingConfig:
    """Configuration for data recording."""
    output_dir: str = "./data"
    prefix: str = "episode"
    record_freq: int = 1  # Record every N env steps
    save_only_successful: bool = True
    success_score_threshold: int = 3
    skip_nan_frames: bool = True
    image_height: int = 480
    image_width: int = 640
    num_arm_joints: int = 6
    enable_depth: bool = False  # Set False for VLA training
    enable_velocity: bool = False  # Set False for VLA training


@dataclass 
class RecordingStats:
    """Statistics for the recording session."""
    total_frames_recorded: int = 0
    frames_skipped_nan: int = 0
    episodes_saved: int = 0
    episodes_discarded: int = 0
    current_episode_frames: int = 0
    
    def reset_episode(self):
        self.current_episode_frames = 0
    
    def log_summary(self):
        """Log recording statistics."""
        print(f"\n{'='*60}")
        print(f"RECORDING STATISTICS")
        print(f"{'='*60}")
        print(f"  Episodes saved:     {self.episodes_saved}")
        print(f"  Episodes discarded: {self.episodes_discarded}")
        print(f"  Total frames:       {self.total_frames_recorded}")
        print(f"  Frames skipped:     {self.frames_skipped_nan} (NaN)")
        if self.total_frames_recorded > 0:
            skip_rate = self.frames_skipped_nan / (self.total_frames_recorded + self.frames_skipped_nan) * 100
            print(f"  Skip rate:          {skip_rate:.2f}%")
        print(f"{'='*60}\n")


class DataRecorder:
    """
    Professional data recorder for Isaac Lab environments.
    
    Features:
    - NaN validation and filtering
    - Inf value detection
    - Joint limit validation
    - Success-only episode saving
    - Memory-efficient buffering
    - Comprehensive statistics
    """
    
    # Expected data shapes for validation
    DATA_SCHEMAS = {
        'left_arm_joint_pos': {'shape': (6,), 'dtype': 'float32', 'limits': (-3.14, 3.14)},
        'right_arm_joint_pos': {'shape': (6,), 'dtype': 'float32', 'limits': (-3.14, 3.14)},
        'left_gripper_joint_pos': {'shape': (), 'dtype': 'float32', 'limits': (-0.1, 0.1)},
        'right_gripper_joint_pos': {'shape': (), 'dtype': 'float32', 'limits': (-0.1, 0.1)},
        'left_arm_action': {'shape': (6,), 'dtype': 'float32', 'limits': (-3.14, 3.14)},
        'right_arm_action': {'shape': (6,), 'dtype': 'float32', 'limits': (-3.14, 3.14)},
        'left_gripper_action': {'shape': (), 'dtype': 'float32', 'limits': (-0.1, 0.1)},
        'right_gripper_action': {'shape': (), 'dtype': 'float32', 'limits': (-0.1, 0.1)},
    }
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        self.stats = RecordingStats()
        self._buffer: Dict[str, List] = {}
        self._reset_buffer()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def _reset_buffer(self):
        """Initialize or reset the data buffer."""
        self._buffer = {
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
        
        # Optional depth data
        if self.config.enable_depth:
            self._buffer['/observations/head_depth'] = []
            self._buffer['/observations/left_hand_depth'] = []
            self._buffer['/observations/right_hand_depth'] = []
        
        # Optional velocity data
        if self.config.enable_velocity:
            self._buffer['/observations/left_arm_joint_vel'] = []
            self._buffer['/observations/right_arm_joint_vel'] = []
            self._buffer['/observations/left_gripper_joint_vel'] = []
            self._buffer['/observations/right_gripper_joint_vel'] = []
        
        self.stats.reset_episode()
    
    @property
    def buffer(self) -> Dict[str, List]:
        return self._buffer
    
    @property
    def frame_count(self) -> int:
        """Current number of frames in buffer."""
        return len(self._buffer.get('/observations/head_rgb', []))
    
    def validate_array(self, name: str, data: np.ndarray, current_time: float) -> tuple[bool, str]:
        """
        Validate a numpy array for common issues.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for NaN
        if np.isnan(data).any():
            return False, f"NaN detected in {name}"
        
        # Check for Inf
        if np.isinf(data).any():
            return False, f"Inf detected in {name}"
        
        # Check value limits if defined
        if name in self.DATA_SCHEMAS:
            schema = self.DATA_SCHEMAS[name]
            min_val, max_val = schema['limits']
            
            if np.any(data < min_val - 0.5) or np.any(data > max_val + 0.5):
                # Warning level - log but don't skip
                print(f"[WARN] {name} values out of expected range [{min_val}, {max_val}]: "
                      f"min={data.min():.3f}, max={data.max():.3f} at t={current_time:.3f}s")
        
        return True, ""
    
    def record_frame(self, obs: Dict, act: Dict, score: int, current_time: float) -> bool:
        """
        Record a single frame of data with validation.
        
        Args:
            obs: Observation dictionary from environment
            act: Action dictionary from environment  
            score: Current task score
            current_time: Current simulation time
            
        Returns:
            True if frame was recorded, False if skipped
        """
        # Extract numerical data
        try:
            data_to_validate = {
                'left_arm_joint_pos': obs['left_arm_joint_pos'].cpu().numpy().squeeze(0),
                'right_arm_joint_pos': obs['right_arm_joint_pos'].cpu().numpy().squeeze(0),
                'left_gripper_joint_pos': obs['left_gripper_joint_pos'].cpu().numpy()[0].squeeze(0),
                'right_gripper_joint_pos': obs['right_gripper_joint_pos'].cpu().numpy()[0].squeeze(0),
                'left_arm_action': act['left_arm_action'].cpu().numpy().squeeze(0),
                'right_arm_action': act['right_arm_action'].cpu().numpy().squeeze(0),
                'left_gripper_action': act['left_gripper_action'].cpu().numpy()[0].squeeze(0),
                'right_gripper_action': act['right_gripper_action'].cpu().numpy()[0].squeeze(0),
            }
        except (KeyError, AttributeError, IndexError) as e:
            print(f"[ERROR] Failed to extract data at t={current_time:.3f}s: {e}")
            self.stats.frames_skipped_nan += 1
            return False
        
        # Validate all numerical data
        for name, data in data_to_validate.items():
            is_valid, error_msg = self.validate_array(name, data, current_time)
            if not is_valid:
                if self.config.skip_nan_frames:
                    print(f"[WARN] Skipping frame at t={current_time:.3f}s: {error_msg}")
                    self.stats.frames_skipped_nan += 1
                    return False
                else:
                    # Replace with zeros (not recommended, but fallback)
                    data_to_validate[name] = np.zeros_like(data)
                    print(f"[WARN] Replacing invalid {name} with zeros at t={current_time:.3f}s")
        
        # All validation passed - record the frame
        # Images (convert RGBA to RGB if needed)
        head_rgb = obs['head_rgb'].cpu().numpy().squeeze(0)
        left_rgb = obs['left_hand_rgb'].cpu().numpy().squeeze(0)
        right_rgb = obs['right_hand_rgb'].cpu().numpy().squeeze(0)
        
        # Handle RGBA -> RGB conversion
        if head_rgb.shape[-1] == 4:
            head_rgb = head_rgb[..., :3]
        if left_rgb.shape[-1] == 4:
            left_rgb = left_rgb[..., :3]
        if right_rgb.shape[-1] == 4:
            right_rgb = right_rgb[..., :3]
        
        self._buffer['/observations/head_rgb'].append(head_rgb)
        self._buffer['/observations/left_hand_rgb'].append(left_rgb)
        self._buffer['/observations/right_hand_rgb'].append(right_rgb)
        
        # Joint positions
        self._buffer['/observations/left_arm_joint_pos'].append(data_to_validate['left_arm_joint_pos'])
        self._buffer['/observations/right_arm_joint_pos'].append(data_to_validate['right_arm_joint_pos'])
        self._buffer['/observations/left_gripper_joint_pos'].append(data_to_validate['left_gripper_joint_pos'])
        self._buffer['/observations/right_gripper_joint_pos'].append(data_to_validate['right_gripper_joint_pos'])
        
        # Actions
        self._buffer['/actions/left_arm_action'].append(data_to_validate['left_arm_action'])
        self._buffer['/actions/right_arm_action'].append(data_to_validate['right_arm_action'])
        self._buffer['/actions/left_gripper_action'].append(data_to_validate['left_gripper_action'])
        self._buffer['/actions/right_gripper_action'].append(data_to_validate['right_gripper_action'])
        
        # Optional depth
        if self.config.enable_depth:
            self._buffer['/observations/head_depth'].append(
                obs['head_depth'].cpu().numpy().squeeze(0).squeeze(-1))
            self._buffer['/observations/left_hand_depth'].append(
                obs['left_hand_depth'].cpu().numpy().squeeze(0).squeeze(-1))
            self._buffer['/observations/right_hand_depth'].append(
                obs['right_hand_depth'].cpu().numpy().squeeze(0).squeeze(-1))
        
        # Optional velocity
        if self.config.enable_velocity:
            self._buffer['/observations/left_arm_joint_vel'].append(
                obs['left_arm_joint_vel'].cpu().numpy().squeeze(0))
            self._buffer['/observations/right_arm_joint_vel'].append(
                obs['right_arm_joint_vel'].cpu().numpy().squeeze(0))
            self._buffer['/observations/left_gripper_joint_vel'].append(
                obs['left_gripper_joint_vel'].cpu().numpy()[0].squeeze(0))
            self._buffer['/observations/right_gripper_joint_vel'].append(
                obs['right_gripper_joint_vel'].cpu().numpy()[0].squeeze(0))
        
        # Metadata
        self._buffer['/score'].append(score)
        self._buffer['/current_time'].append(current_time)
        
        self.stats.total_frames_recorded += 1
        self.stats.current_episode_frames += 1
        
        return True
    
    def save_episode(self, score: int, force_save: bool = False) -> Optional[str]:
        """
        Save the current episode buffer to HDF5 file.
        
        Args:
            score: Final episode score
            force_save: If True, save regardless of success threshold
            
        Returns:
            Path to saved file, or None if discarded
        """
        num_frames = self.frame_count
        
        # Check if episode should be saved
        if num_frames == 0:
            print(f"[WARN] No frames to save, discarding episode")
            self.stats.episodes_discarded += 1
            self._reset_buffer()
            return None
        
        if self.config.save_only_successful and not force_save:
            if score < self.config.success_score_threshold:
                print(f"[INFO] Episode discarded: score {score} < threshold {self.config.success_score_threshold}")
                self.stats.episodes_discarded += 1
                self._reset_buffer()
                return None
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.config.prefix}_score{score}_{timestamp}.h5"
        filepath = os.path.join(self.config.output_dir, filename)
        
        print(f"[INFO] Saving episode: {num_frames} frames, score={score}")
        
        try:
            with h5py.File(filepath, 'w') as f:
                # Metadata
                f.attrs['sim'] = True
                f.attrs['num_frames'] = num_frames
                f.attrs['final_score'] = score
                f.attrs['timestamp'] = timestamp
                f.attrs['frames_skipped'] = self.stats.frames_skipped_nan
                
                # Create groups
                obs = f.create_group('observations')
                act = f.create_group('actions')
                
                # Images
                h, w = self.config.image_height, self.config.image_width
                obs.create_dataset('head_rgb', data=np.array(self._buffer['/observations/head_rgb']), 
                                 dtype='uint8', compression='gzip', compression_opts=4)
                obs.create_dataset('left_hand_rgb', data=np.array(self._buffer['/observations/left_hand_rgb']),
                                 dtype='uint8', compression='gzip', compression_opts=4)
                obs.create_dataset('right_hand_rgb', data=np.array(self._buffer['/observations/right_hand_rgb']),
                                 dtype='uint8', compression='gzip', compression_opts=4)
                
                # Joint positions
                obs.create_dataset('left_arm_joint_pos', 
                                 data=np.array(self._buffer['/observations/left_arm_joint_pos']), dtype='float32')
                obs.create_dataset('right_arm_joint_pos',
                                 data=np.array(self._buffer['/observations/right_arm_joint_pos']), dtype='float32')
                obs.create_dataset('left_gripper_joint_pos',
                                 data=np.array(self._buffer['/observations/left_gripper_joint_pos']), dtype='float32')
                obs.create_dataset('right_gripper_joint_pos',
                                 data=np.array(self._buffer['/observations/right_gripper_joint_pos']), dtype='float32')
                
                # Optional depth
                if self.config.enable_depth:
                    obs.create_dataset('head_depth',
                                     data=np.array(self._buffer['/observations/head_depth']), dtype='float32')
                    obs.create_dataset('left_hand_depth',
                                     data=np.array(self._buffer['/observations/left_hand_depth']), dtype='float32')
                    obs.create_dataset('right_hand_depth',
                                     data=np.array(self._buffer['/observations/right_hand_depth']), dtype='float32')
                
                # Optional velocity
                if self.config.enable_velocity:
                    obs.create_dataset('left_arm_joint_vel',
                                     data=np.array(self._buffer['/observations/left_arm_joint_vel']), dtype='float32')
                    obs.create_dataset('right_arm_joint_vel',
                                     data=np.array(self._buffer['/observations/right_arm_joint_vel']), dtype='float32')
                    obs.create_dataset('left_gripper_joint_vel',
                                     data=np.array(self._buffer['/observations/left_gripper_joint_vel']), dtype='float32')
                    obs.create_dataset('right_gripper_joint_vel',
                                     data=np.array(self._buffer['/observations/right_gripper_joint_vel']), dtype='float32')
                
                # Actions
                act.create_dataset('left_arm_action',
                                 data=np.array(self._buffer['/actions/left_arm_action']), dtype='float32')
                act.create_dataset('right_arm_action',
                                 data=np.array(self._buffer['/actions/right_arm_action']), dtype='float32')
                act.create_dataset('left_gripper_action',
                                 data=np.array(self._buffer['/actions/left_gripper_action']), dtype='float32')
                act.create_dataset('right_gripper_action',
                                 data=np.array(self._buffer['/actions/right_gripper_action']), dtype='float32')
                
                # Metadata arrays
                f.create_dataset('score', data=np.array(self._buffer['/score']), dtype='int32')
                f.create_dataset('current_time', data=np.array(self._buffer['/current_time']), dtype='float32')
            
            print(f"[SUCCESS] Saved: {filepath}")
            self.stats.episodes_saved += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to save episode: {e}")
            self.stats.episodes_discarded += 1
            filepath = None
        
        # Reset buffer for next episode
        self._reset_buffer()
        
        return filepath
    
    def discard_episode(self):
        """Explicitly discard current episode buffer."""
        print(f"[INFO] Discarding episode with {self.frame_count} frames")
        self.stats.episodes_discarded += 1
        self._reset_buffer()
