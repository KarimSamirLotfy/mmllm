#!/usr/bin/env python3
"""
Command-line interface for parallel benchmarking pipeline.
"""

import argparse
import logging
import os
import sys
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from mmllm.evaluation.types import BenchmarkConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel benchmarking pipeline for AiTW dataset evaluation"
    )
    
    # Dataset configuration
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
        default=['general'],
        help='Datasets to process (default: general)'
    )
    
    # Episode range
    parser.add_argument(
        '--start-episode', 
        type=int, 
        default=0,
        help='Starting episode index (default: 0)'
    )
    parser.add_argument(
        '--end-episode', 
        type=int, 
        default=None,
        help='Ending episode index (default: None for all episodes)'
    )
    parser.add_argument(
        '--max-steps', 
        type=int, 
        default=10,
        help='Maximum steps per episode (default: 10)'
    )
    
    # Parallel processing configuration
    parser.add_argument(
        '--workers', 
        type=int, 
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=5,
        help='Episodes per batch (default: 5)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./benchmark_results',
        help='Output directory for results (default: ./benchmark_results)'
    )
    parser.add_argument(
        '--run-name', 
        type=str, 
        default='parallel_benchmark',
        help='Name for this benchmark run (default: parallel_benchmark)'
    )
    
    # Processing options
    parser.add_argument(
        '--ocr', 
        action='store_true',
        help='Enable OCR module (default: False)'
    )
    parser.add_argument(
        '--sequential', 
        action='store_true',
        help='Use sequential processing instead of parallel (default: False)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging (default: False)'
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_config(args) -> BenchmarkConfig:
    """Validate arguments and create configuration."""
    
    # Validate episode range
    if args.end_episode is not None and args.end_episode <= args.start_episode:
        raise ValueError(f"end_episode ({args.end_episode}) must be greater than start_episode ({args.start_episode})")
    
    # Validate parallel processing parameters
    if args.workers < 1:
        raise ValueError(f"workers ({args.workers}) must be at least 1")
    
    if args.batch_size < 1:
        raise ValueError(f"batch_size ({args.batch_size}) must be at least 1")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = BenchmarkConfig(
        dataset_names=args.datasets,
        ocr_module=args.ocr,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps_per_episode=args.max_steps,
        max_workers=args.workers,
        batch_size=args.batch_size
    )
    
    return config

def print_config_summary(config: BenchmarkConfig, sequential: bool):
    """Print configuration summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*60}")
    print(f"Datasets:        {', '.join(config.dataset_names)}")
    print(f"Episodes:        {config.start_episode} to {config.end_episode or 'end'}")
    print(f"Max steps:       {config.max_steps_per_episode}")
    print(f"OCR enabled:     {config.ocr_module}")
    print(f"Processing:      {'Sequential' if sequential else 'Parallel'}")
    if not sequential:
        print(f"Workers:         {config.max_workers}")
        print(f"Batch size:      {config.batch_size}")
    print(f"Output dir:      {config.output_dir}")
    print(f"Run name:        {config.run_name}")
    print(f"{'='*60}\n")

def main():
    """Main entry point."""
    try:
        args = parse_args()
        setup_logging(args.verbose)
        
        config = validate_config(args)
        print_config_summary(config, args.sequential)
        
        # Initialize pipeline
        pipeline = BenchmarkingPipeline(config)
        
        # Run benchmark
        if args.sequential:
            print("Starting sequential benchmark...")
            episode_results = pipeline.run_benchmark_sequential()
        else:
            print("Starting parallel benchmark...")
            try:
                episode_results = pipeline.run_benchmark()
            except Exception as e:
                print(f"Parallel processing failed: {e}")
                print("Falling back to sequential processing...")
                episode_results = pipeline.run_benchmark_sequential()
        
        # Generate report
        print("Generating comprehensive report...")
        report = pipeline.generate_comprehensive_report(episode_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETED")
        print(f"{'='*60}")
        print(f"Total episodes:  {len(episode_results)}")
        print(f"Overall accuracy: {report['overall_metrics']['accuracy']:.2f}%")
        print(f"Success rate:    {report['overall_metrics']['success_rate']:.2f}%")
        print(f"Output files:    {config.output_dir}")
        
        # Dataset breakdown
        if 'dataset_metrics' in report:
            print(f"\nDataset breakdown:")
            for dataset, metrics in report['dataset_metrics'].items():
                print(f"  {dataset:12} - Accuracy: {metrics['accuracy']:.1f}%, Episodes: {metrics['total_episodes']}")
        
        print(f"{'='*60}")
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
