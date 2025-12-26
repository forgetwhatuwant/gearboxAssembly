#!/usr/bin/env python3
"""
Quick utility to inspect HDF5 episode files.
Usage: python scripts/check_hdf5.py <file.h5>
"""

import h5py
import os
import sys
import numpy as np
import argparse


def check_hdf5(filepath: str):
    """Analyze an HDF5 episode file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    with h5py.File(filepath, 'r') as f:
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        print("=" * 60)
        print("HDF5 FILE ANALYSIS")
        print("=" * 60)
        print(f"File: {os.path.basename(filepath)}")
        print(f"Size: {size_mb:.2f} MB")
        
        # Metadata
        print("\n📋 METADATA:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        # Structure
        print("\n📁 STRUCTURE:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
        
        # Resolution check
        if 'observations/head_rgb' in f:
            head_rgb = f['observations/head_rgb']
            height, width = head_rgb.shape[1], head_rgb.shape[2]
            print(f"\n📷 IMAGE RESOLUTION: {height}x{width}")
            if height >= 480:
                print("  ✅ High resolution (480p+)")
            else:
                print("  ⚠️ Standard resolution (240p)")
        
        # Data quality
        print("\n✅ DATA QUALITY:")
        for key in ['observations/left_arm_joint_pos', 'observations/right_arm_joint_pos',
                    'actions/left_arm_action', 'actions/right_arm_action']:
            if key in f:
                data = f[key][:]
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                status = "✅" if nan_count == 0 and inf_count == 0 else "❌"
                print(f"  {status} {key.split('/')[-1]}: NaN={nan_count}, Inf={inf_count}")
        
        # Score distribution
        if 'score' in f:
            score = f['score'][:]
            print("\n📊 SCORE DISTRIBUTION:")
            unique, counts = np.unique(score, return_counts=True)
            for s, c in zip(unique, counts):
                pct = c / len(score) * 100
                print(f"  Score {s}: {c} frames ({pct:.1f}%)")
        
        # Timing
        if 'current_time' in f:
            time_data = f['current_time'][:]
            num_frames = len(time_data)
            duration = time_data[-1]
            fps = num_frames / duration if duration > 0 else 0
            print(f"\n⏱️ TIMING:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Frames: {num_frames}")
            print(f"  Recording FPS: {fps:.1f}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 episode files")
    parser.add_argument("file", type=str, help="Path to HDF5 file")
    args = parser.parse_args()
    
    check_hdf5(args.file)


if __name__ == "__main__":
    main()
