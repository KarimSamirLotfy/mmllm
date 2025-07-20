#!/usr/bin/env python3
"""
Memory-safe benchmark runner to avoid double free errors.
Use this script when experiencing memory corruption issues on certain machines.
"""

import os
import sys
import subprocess
import argparse

def set_safe_environment():
    """Set environment variables to minimize memory corruption risks."""
    safe_env = {
        # TensorFlow settings
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'CUDA_VISIBLE_DEVICES': '',
        'GRPC_VERBOSITY': 'ERROR',
        'GLOG_minloglevel': '3',
        
        # Memory management
        'MALLOC_CHECK_': '0',  # Disable glibc malloc checking
        'MALLOC_PERTURB_': '0',  # Disable malloc perturbation
        
        # Python multiprocessing
        'PYTHONMULTIPROCESSING_START_METHOD': 'fork',
        
        # JAX settings
        'JAX_PLATFORMS': 'cpu',
        'JAX_ENABLE_X64': 'true',
    }
    
    for key, value in safe_env.items():
        os.environ[key] = value
        print(f"Set {key}={value}")

def run_safe_benchmark(args):
    """Run benchmark with conservative settings."""
    # Force safe settings
    cmd = [
        sys.executable, 'run_parallel_benchmark.py',
        '--datasets', args.dataset,
        '--workers', '1',  # Single worker to avoid multiprocessing issues
        '--batch-size', '1',  # Minimal batch size
        '--end-episode', str(args.episodes),
        '--verbose'
    ]
    
    if args.ocr:
        cmd.append('--ocr')
    
    if args.sequential:
        cmd.append('--sequential')
    
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    
    if args.run_name:
        cmd.extend(['--run-name', args.run_name])
    
    print(f"Running safe benchmark: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Memory-safe benchmark runner for systems with double free errors"
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
        default='general',
        help='Dataset to test (default: general)'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=5,
        help='Number of episodes to test (default: 5)'
    )
    
    parser.add_argument(
        '--ocr', 
        action='store_true',
        help='Enable OCR module'
    )
    
    parser.add_argument(
        '--sequential', 
        action='store_true',
        help='Use sequential processing (safest option)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--run-name',
        help='Name for this benchmark run'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MEMORY-SAFE BENCHMARK RUNNER")
    print("=" * 60)
    print("This script uses conservative settings to avoid memory corruption.")
    print("Use this when experiencing 'double free' or 'tcache' errors.")
    print()
    
    # Set safe environment
    print("Setting safe environment variables...")
    set_safe_environment()
    print()
    
    # Run benchmark
    print("Starting safe benchmark...")
    result = run_safe_benchmark(args)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("BENCHMARK FAILED")
        print("=" * 60)
        print("Try using --sequential flag for maximum safety")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
