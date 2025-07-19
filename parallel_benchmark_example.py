#!/usr/bin/env python3
"""
Example script demonstrating how to use the parallel benchmarking pipeline.
"""

import logging
import os
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

def main():
    """Run a parallel benchmark example."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration for parallel processing
    config = BenchmarkConfig(
        dataset_names=['general'],  # Start with just one dataset
        ocr_module=True,
        start_episode=0,
        end_episode=10,  # Process first 10 episodes
        output_dir='./parallel_benchmark_results',
        run_name='parallel_example_run',
        max_steps_per_episode=5,
        max_workers=4,  # Use 4 parallel workers
        batch_size=3    # Process 3 episodes per batch
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize and run parallel pipeline
    print("Starting parallel benchmarking pipeline...")
    pipeline = BenchmarkingPipeline(config)
    
    try:
        # Run the parallel benchmark
        episode_results = pipeline.run_benchmark()
        
        # Generate comprehensive report
        report = pipeline.generate_comprehensive_report(episode_results)
        
        print(f"\n{'='*60}")
        print("PARALLEL BENCHMARK COMPLETED")
        print(f"{'='*60}")
        print(f"Total episodes processed: {len(episode_results)}")
        print(f"Overall accuracy: {report['overall_metrics']['accuracy']:.2f}%")
        print(f"Overall success rate: {report['overall_metrics']['success_rate']:.2f}%")
        print(f"Workers used: {config.max_workers}")
        print(f"Batch size: {config.batch_size}")
        print(f"Output directory: {config.output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error running parallel benchmark: {e}")
        print("Trying sequential fallback...")
        
        # Fallback to sequential processing
        try:
            episode_results = pipeline.run_benchmark_sequential()
            report = pipeline.generate_comprehensive_report(episode_results)
            
            print(f"\n{'='*60}")
            print("SEQUENTIAL BENCHMARK COMPLETED (FALLBACK)")
            print(f"{'='*60}")
            print(f"Total episodes processed: {len(episode_results)}")
            print(f"Overall accuracy: {report['overall_metrics']['accuracy']:.2f}%")
            print(f"Overall success rate: {report['overall_metrics']['success_rate']:.2f}%")
            print(f"{'='*60}")
            
        except Exception as fallback_error:
            print(f"Sequential fallback also failed: {fallback_error}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
