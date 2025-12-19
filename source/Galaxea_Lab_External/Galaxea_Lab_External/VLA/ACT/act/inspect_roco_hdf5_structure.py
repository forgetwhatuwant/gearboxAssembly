
import h5py
import argparse
import sys
import os

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        # Print first few elements if it's small or 1D
        if obj.size < 10:
             print(f"  Values: {obj[:]}")

def inspect_file():
    parser = argparse.ArgumentParser(description="Inspect HDF5 structure.")
    parser.add_argument("file_path", type=str, help="Path to HDF5 file")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        sys.exit(1)

    print(f"Inspecting: {args.file_path}")
    print("-" * 50)
    
    try:
        with h5py.File(args.file_path, 'r') as f:
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_file()
