#!/usr/bin/env python3
"""
Performance comparison script for parallel vs sequential benchmarking.
"""

import time
import logging
import os
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

def run_performance_comparison():
    """Compare parallel vs sequential performance."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration for comparison
    config = BenchmarkConfig(
        dataset_names=['general'],
        ocr_module=True,
        start_episode=0,
        end_episode=5,  # Small test set for comparison
        output_dir='./performance_comparison',
        run_name='performance_test',
        max_steps_per_episode=3,
        max_workers=4,
        batch_size=2
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    pipeline = BenchmarkingPipeline(config)
    
    results = {}
    
    # Run sequential benchmark
    print("Running sequential benchmark...")
    start_time = time.time()
    try:
        sequential_results = pipeline.run_benchmark_sequential()
        sequential_time = time.time() - start_time
        results['sequential'] = {
            'time': sequential_time,
            'episodes': len(sequential_results),
            'success': True
        }
        print(f"Sequential completed in {sequential_time:.2f} seconds")
    except Exception as e:
        results['sequential'] = {'time': None, 'episodes': 0, 'success': False, 'error': str(e)}
        print(f"Sequential failed: {e}")
    
    # Run parallel benchmark
    print("\nRunning parallel benchmark...")
    start_time = time.time()
    try:
        parallel_results = pipeline.run_benchmark()
        parallel_time = time.time() - start_time
        results['parallel'] = {
            'time': parallel_time,
            'episodes': len(parallel_results),
            'success': True
        }
        print(f"Parallel completed in {parallel_time:.2f} seconds")
    except Exception as e:
        results['parallel'] = {'time': None, 'episodes': 0, 'success': False, 'error': str(e)}
        print(f"Parallel failed: {e}")
    
    # Print comparison results
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if results['sequential']['success'] and results['parallel']['success']:
        speedup = results['sequential']['time'] / results['parallel']['time']
        print(f"Sequential time: {results['sequential']['time']:.2f} seconds")
        print(f"Parallel time:   {results['parallel']['time']:.2f} seconds")
        print(f"Speedup:         {speedup:.2f}x")
        print(f"Workers used:    {config.max_workers}")
        print(f"Episodes:        {results['parallel']['episodes']}")
        
        if speedup > 1:
            print(f"✅ Parallel processing is {speedup:.2f}x faster!")
        else:
            print(f"⚠️  Parallel processing is slower (overhead may exceed benefits for small datasets)")
    else:
        print("⚠️  Comparison incomplete due to errors:")
        if not results['sequential']['success']:
            print(f"  Sequential error: {results['sequential'].get('error', 'Unknown')}")
        if not results['parallel']['success']:
            print(f"  Parallel error: {results['parallel'].get('error', 'Unknown')}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    run_performance_comparison()
