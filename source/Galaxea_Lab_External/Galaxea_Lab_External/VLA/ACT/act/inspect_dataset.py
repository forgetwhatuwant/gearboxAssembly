
import os
import h5py
import argparse
import glob

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"{name}/ (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")
        # Print attributes if any
        if len(obj.attrs) > 0:
            print(f"  Attributes: {dict(obj.attrs)}")
        if obj.ndim > 0 and obj.dtype.kind in 'fi': # float or int
             print(f"  First 5 rows: {obj[:5]}")

def inspect_file(file_path):
    print(f"\ninspecting: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            f.visititems(print_structure)
            # Check for root attributes
            if len(f.attrs) > 0:
                print(f"Root Attributes: {dict(f.attrs)}")
        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset structure for ACT compatibility.")
    parser.add_argument("path", nargs='?', default="/mnt/nas/isaac_sim_data/datasets--rocochallenge2025--rocochallenge2025", 
                        help="Path to HDF5 file or directory containing HDF5 files")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}")
        return

    if os.path.isfile(path):
        inspect_file(path)
    elif os.path.isdir(path):
        print(f"Scanning directory: {path}")
        # Look for typical patterns
        patterns = ["*.hdf5", "*.h5"]
        found_files = []
        for p in patterns:
            found_files.extend(glob.glob(os.path.join(path, "**", p), recursive=True))
        
        if not found_files:
            print("No HDF5 files found in directory (recursive search for *.hdf5, *.h5).")
            # List first few files to give a hint
            print("Files present in directory:")
            for item in sorted(os.listdir(path))[:10]:
                print(f"  {item}")
        else:
            print(f"Found {len(found_files)} HDF5 files. Checking for valid files...")
            
            inspected_count = 0
            for file_path in found_files:
                if inspect_file(file_path):
                    inspected_count += 1
                    break # Stop after one successful inspection
            
            if inspected_count == 0:
                 print("\nCould not successfully read any of the found HDF5 files.")
            elif len(found_files) > 1:
                print("\n(Run with a specific file path to inspect others)")

if __name__ == "__main__":
    main()
