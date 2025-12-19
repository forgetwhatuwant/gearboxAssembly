
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors, get_policy_class
from lerobot.processor import PolicyAction
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os

# Ensure policy wrapper logic is available or replicated
# For direct comparison, we'll interface with LeRobot Policy directly

def compare_policy_vs_gt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="/media/hls/HIKSEMI/isaac_sim_data/lerobot_datasets/roco_updated_debug")
    parser.add_argument("--episode_idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Policy
    print(f"Loading policy from: {args.checkpoint}")
    from lerobot.policies.pretrained import PreTrainedPolicy
    # Assume ACT for now or let generic loading handle it if possible. 
    # Since we can't easily rely on wrapped class here without copying, let's look at config.
    import json
    with open(os.path.join(args.checkpoint, "config.json"), 'r') as f:
        config = json.load(f)
    
    policy_cls = get_policy_class(config['type'])
    policy = policy_cls.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # Create Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint
    )
    if hasattr(preprocessor, "to"): preprocessor.to(device)

    # 2. Load Dataset
    print(f"Loading dataset from: {args.dataset_dir}")
    dataset = LeRobotDataset(root=args.dataset_dir, repo_id="galaxea/roco_updated_debug")
    
    # Get frame range for episode
    meta = dataset.meta.episodes[args.episode_idx]
    
    # Handle int/tensor discrepancy in metadata safely
    def get_item(x): return x.item() if hasattr(x, 'item') else x
    
    from_idx = get_item(meta["dataset_from_index"])
    to_idx = get_item(meta["dataset_to_index"])
    length = to_idx - from_idx
    print(f"Episode {args.episode_idx}: Frames {from_idx} -> {to_idx} (Len: {length})")

    # Arrays to store results
    gt_l_grip = []
    pred_l_grip = []
    
    gt_l_arm_0 = []
    pred_l_arm_0 = []

    print("Running inference on dataset frames...")
    
    with torch.no_grad():
        for i in range(length):
            idx = from_idx + i
            frame = dataset[idx]

            # Prepare Input Batch
            batch = {}
            
            # --- State ---
            # Dataset: "observation.state" -> (14,)
            obs_state = frame["observation.state"]
            batch["observation.state"] = obs_state.unsqueeze(0).to(device) # (1, 14)

            # --- Images ---
            # Dataset: "observation.images.head" -> (C, H, W) float32 normalized?
            # LeRobotDataset usually returns (C,H,W) float [0,1] if generic. verify.
            # roco_updated_debug might be returning video frames.
            
            # We need to match keys required by policy.
            # Policy input_features keys: e.g. "observation.images.head_camera"
            # Dataset keys: "observation.images.head"
            
            # Simple Mapping based on typical conversion:
            # Policy "head_camera" <-> Dataset "head"
            
            # Populate batch based on Policy Requirements
            # Policy expects specific keys defined in config.input_features or image_features
            
            # 1. Map typical dataset keys to potential policy keys
            # Dataset keys: "observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"
            dataset_key_map = {
                "head": "observation.images.head",
                "head_camera": "observation.images.head",
                "left_wrist": "observation.images.left_wrist",
                "left_hand_camera": "observation.images.left_wrist",
                "right_wrist": "observation.images.right_wrist",
                "right_hand_camera": "observation.images.right_wrist",
                # Fallback for exact matches
                "observation.images.head": "observation.images.head",
                "observation.images.left_wrist": "observation.images.left_wrist",
                "observation.images.right_wrist": "observation.images.right_wrist"
            }

            for policy_key in policy.config.image_features:
                # policy_key example: "observation.images.head" or "observation.images.head_camera"
                found = False
                for token, ds_key in dataset_key_map.items():
                    if token in policy_key:
                        if ds_key in frame:
                            batch[policy_key] = frame[ds_key].unsqueeze(0).to(device)
                            found = True
                            break
                
                if not found:
                     # Fallback: check if exact key exists in frame (unlikely for nested names but possible)
                     if policy_key in frame:
                         batch[policy_key] = frame[policy_key].unsqueeze(0).to(device)
                     else:
                         print(f"Warning: Could not find dataset match for policy key: {policy_key}")

            # Preprocess
            batch = preprocessor(batch)

            # Predict
            output_raw = policy.select_action(batch)
            
            # Postprocess
            # output_dict = PolicyAction(action=output_raw) <-- No, just pass tensor based on previous fix
            output_dict = postprocessor(output_raw)
            if isinstance(output_dict, dict):
                 pred_action = output_dict["action"].cpu().numpy().squeeze()
            else:
                 # It returns tensor directly
                 pred_action = output_dict.cpu().numpy().squeeze()
            
            # Ground Truth Action
            gt_action = frame["action"].numpy().squeeze()

            # Store Data
            # Indices: [0-5 Left Arm] [6-11 Right Arm] [12 Left Grip] [13 Right Grip]
            gt_l_grip.append(gt_action[12])
            pred_l_grip.append(pred_action[12])
            
            gt_l_arm_0.append(gt_action[0])
            pred_l_arm_0.append(pred_action[0])
            
            if i % 20 == 0:
                print(f"Frame {i}: GT_Grip={gt_action[12]:.4f} Pred_Grip={pred_action[12]:.4f}")

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Gripper
    ax[0].plot(gt_l_grip, label='GT Left Gripper', color='black', linewidth=2)
    ax[0].plot(pred_l_grip, label='Pred Left Gripper', color='orange', linestyle='--')
    ax[0].set_title(f'Left Gripper Action: GT vs Pred (Ep {args.episode_idx})')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot 2: Arm Joint 0
    ax[1].plot(gt_l_arm_0, label='GT Left Arm Joint 0', color='blue', linewidth=2)
    ax[1].plot(pred_l_arm_0, label='Pred Left Arm Joint 0', color='red', linestyle='--')
    ax[1].set_title('Left Arm Joint 0 Action: GT vs Pred')
    ax[1].legend()
    ax[1].grid(True)

    out_path = f"eval_policy_vs_gt_ep{args.episode_idx}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved comparison plot to: {out_path}")

if __name__ == "__main__":
    compare_policy_vs_gt()
