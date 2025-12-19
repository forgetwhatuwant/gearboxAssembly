
import h5py
import argparse
import numpy as np

def compare_actions_obs():
    parser = argparse.ArgumentParser(description="Compare Actions and Observations in ROCO HDF5.")
    parser.add_argument("file_path", type=str, help="Path to HDF5 file")
    args = parser.parse_args()

    print(f"Comparing: {args.file_path}")
    
    with h5py.File(args.file_path, 'r') as f:
        # Check if actions group exists
        if 'actions' not in f:
            print("No 'actions' group found!")
            return

        # Observations
        obs_l_grip = f['observations/left_gripper_joint_pos'][:]
        obs_r_grip = f['observations/right_gripper_joint_pos'][:]

        # Actions
        # Need to check exact key names in 'actions' group
        print("Keys in 'actions':", list(f['actions'].keys()))
        
        # Assuming typical naming, but let's be safe and try to match
        act_l_grip = f['actions/left_gripper_action'][:] if 'left_gripper_action' in f['actions'] else None
        act_r_grip = f['actions/right_gripper_action'][:] if 'right_gripper_action' in f['actions'] else None

        if act_l_grip is None:
            # Maybe it's named differently?
            # User snippet showed: actions/left_gripper (maybe without _action?)
            # User snippet: "Dataset: actions/left_gripper" 
            act_l_grip = f['actions/left_gripper'][:] if 'left_gripper' in f['actions'] else None
            act_r_grip = f['actions/right_gripper'][:] if 'right_gripper' in f['actions'] else None

        if act_l_grip is None:
             print("Could not find gripper actions!")
             return

        print("\n--- Statistics ---")
        print(f"Obs Left Grip: Min={obs_l_grip.min():.4f}, Mean={obs_l_grip.mean():.4f}, Max={obs_l_grip.max():.4f}")
        print(f"Act Left Grip: Min={act_l_grip.min():.4f}, Mean={act_l_grip.mean():.4f}, Max={act_l_grip.max():.4f}")
        
        print(f"Obs Right Grip: Min={obs_r_grip.min():.4f}, Mean={obs_r_grip.mean():.4f}, Max={obs_r_grip.max():.4f}")
        print(f"Act Right Grip: Min={act_r_grip.min():.4f}, Mean={act_r_grip.mean():.4f}, Max={act_r_grip.max():.4f}")

        # Check Arm Actions too
        act_l_arm = f['actions/left_arm_action'][:] if 'left_arm_action' in f['actions'] else None
        if act_l_arm is not None:
             print(f"Act Left Arm: Min={act_l_arm.min():.4f}, Max={act_l_arm.max():.4f}")
             print(f"Are arm actions all zero? {np.all(act_l_arm == 0)}")

        # Check for non-zero actions if they were thought to be empty
        print(f"\nAre gripper actions all zero? {np.all(act_l_grip == 0)}")
        
        # Check difference
        # Align lengths if needed (usually actions and obs are same length in HDF5)
        diff = np.mean(np.abs(obs_l_grip - act_l_grip))
        print(f"Mean Abs Diff (Obs - Act) Left: {diff:.6f}")

if __name__ == "__main__":
    compare_actions_obs()
