#!/usr/bin/env python3
"""
Test script to verify the new configuration options work correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mmllm.evaluation.types import BenchmarkConfig
from mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent

def test_config_creation():
    """Test that BenchmarkConfig can be created with new parameters."""
    print("Testing BenchmarkConfig creation with new parameters...")
    
    config = BenchmarkConfig(
        dataset_names=['general'],
        ocr_module=True,
        prompt_with_android_tree=True,
        add_image_history=True,
        start_episode=0,
        end_episode=2,
        output_dir='./test_output',
        run_name='test_config',
        max_steps_per_episode=3,
        max_workers=2,
        batch_size=1
    )
    
    print(f"✓ BenchmarkConfig created successfully")
    print(f"  - OCR module: {config.ocr_module}")
    print(f"  - Android tree prompt: {config.prompt_with_android_tree}")
    print(f"  - Image history: {config.add_image_history}")
    return config

def test_agent_creation(config):
    """Test that SimpleOCRAgent can be created with new parameters."""
    print("\nTesting SimpleOCRAgent creation with new parameters...")
    
    agent = SimpleOCRAgent(
        ocr_module=config.ocr_module,
        prompt_with_android_tree=config.prompt_with_android_tree,
        add_image_history=config.add_image_history
    )
    
    print(f"✓ SimpleOCRAgent created successfully")
    print(f"  - OCR module: {agent.ocr_module}")
    print(f"  - Android tree prompt: {agent.prompt_with_android_tree}")
    print(f"  - Image history: {agent.add_image_history}")
    return agent

def test_pipeline_creation(config):
    """Test that BenchmarkingPipeline can be created with new configuration."""
    print("\nTesting BenchmarkingPipeline creation with new configuration...")
    
    pipeline = BenchmarkingPipeline(config)
    
    print(f"✓ BenchmarkingPipeline created successfully")
    print(f"  - Agent OCR module: {pipeline.agent.ocr_module}")
    print(f"  - Agent Android tree prompt: {pipeline.agent.prompt_with_android_tree}")
    print(f"  - Agent Image history: {pipeline.agent.add_image_history}")
    return pipeline

def main():
    """Run all tests."""
    print("Testing new configuration options for SimpleOCRAgent")
    print("=" * 60)
    
    try:
        # Test 1: Config creation
        config = test_config_creation()
        
        # Test 2: Agent creation
        agent = test_agent_creation(config)
        
        # Test 3: Pipeline creation
        pipeline = test_pipeline_creation(config)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! New configuration options are working correctly.")
        print("\nConfiguration summary:")
        print(f"  - Datasets: {', '.join(config.dataset_names)}")
        print(f"  - OCR module: {config.ocr_module}")
        print(f"  - Android tree prompt: {config.prompt_with_android_tree}")
        print(f"  - Image history: {config.add_image_history}")
        print(f"  - Workers: {config.max_workers}")
        print(f"  - Batch size: {config.batch_size}")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
