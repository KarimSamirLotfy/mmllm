# Quick Start Guide - Parallel Benchmarking

## Command Line Usage

### Basic parallel benchmark
```bash
python run_parallel_benchmark.py --datasets general --workers 4 --batch-size 5 --end-episode 20
```

### High-performance parallel benchmark with enhanced agent
```bash
python run_parallel_benchmark.py \
  --datasets general google_apps \
  --workers 8 \
  --batch-size 10 \
  --end-episode 100 \
  --max-steps 15 \
  --ocr \
  --prompt-with-android-tree \
  --add-image-history \
  --output-dir ./high_perf_results \
  --run-name enhanced_benchmark
```

### Development/testing (small, fast)
```bash
python run_parallel_benchmark.py \
  --datasets general \
  --workers 2 \
  --batch-size 2 \
  --end-episode 5 \
  --max-steps 3 \
  --verbose
```

### Sequential fallback (for debugging)
```bash
python run_parallel_benchmark.py --sequential --datasets general --end-episode 10
```

### Memory-safe mode (for unstable systems)
```bash
# If you get "double free" or "tcache" errors, use this
python run_parallel_benchmark.py \
  --datasets general \
  --workers 1 \
  --batch-size 1 \
  --end-episode 10 \
  --verbose
```

## Python API Usage

### Simple parallel benchmark
```python
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

config = BenchmarkConfig(
    dataset_names=['general'],
    ocr_module=True,
    prompt_with_android_tree=False,  # Use default prompt
    add_image_history=True,  # Enable image history for better context
    start_episode=0,
    end_episode=50,
    output_dir='./results',
    run_name='my_benchmark',
    max_workers=4,
    batch_size=5
)

pipeline = BenchmarkingPipeline(config)
results = pipeline.run_benchmark()
report = pipeline.generate_comprehensive_report(results)
```

### Error handling with fallback
```python
pipeline = BenchmarkingPipeline(config)

try:
    # Try parallel first
    results = pipeline.run_benchmark()
    print("Parallel processing completed successfully")
except Exception as e:
    print(f"Parallel failed: {e}, using sequential fallback")
    results = pipeline.run_benchmark_sequential()

report = pipeline.generate_comprehensive_report(results)
```

## Performance Testing

### Compare parallel vs sequential
```bash
python performance_comparison.py
```

### Safe benchmark runner (for problematic machines)
```bash
# Use the safe benchmark runner for memory corruption issues
python run_safe_benchmark.py --dataset general --episodes 5 --ocr

# Or with sequential processing (safest)
python run_safe_benchmark.py --dataset general --episodes 5 --sequential
```

### Example with different worker counts
```python
import time
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

base_config = {
    'dataset_names': ['general'],
    'ocr_module': True,
    'prompt_with_android_tree': False,
    'add_image_history': True,
    'start_episode': 0,
    'end_episode': 10,
    'output_dir': './perf_test',
    'run_name': 'perf_test',
    'batch_size': 3
}

for workers in [1, 2, 4, 8]:
    config = BenchmarkConfig(**base_config, max_workers=workers)
    pipeline = BenchmarkingPipeline(config)
    
    start_time = time.time()
    results = pipeline.run_benchmark()
    elapsed = time.time() - start_time
    
    print(f"Workers: {workers}, Time: {elapsed:.2f}s, Episodes: {len(results)}")
```

## Recommended Configurations

### Development
- Workers: 2-4
- Batch size: 2-3
- Episodes: 5-20
- Good for testing and debugging

### Production Evaluation
- Workers: 8-16
- Batch size: 10-15
- Episodes: 100+
- Good for comprehensive benchmarks

### CI/CD Pipeline
- Workers: 4-8
- Batch size: 5-8
- Episodes: 50-100
- Good balance of speed and resource usage

## Monitoring Tips

1. **Check CPU usage**: `htop` or `top` while running
2. **Memory usage**: Monitor with `free -h`
3. **Log output**: Use `--verbose` for detailed logging
4. **Progress tracking**: Workers log completion of each batch

## Troubleshooting

### Performance not improving?
- Check if you're I/O bound (dataset loading)
- Reduce batch size if memory constrained
- Don't exceed 2x CPU core count for workers

### Workers crashing?
- Use sequential mode for debugging: `--sequential`
- Check memory usage and reduce batch size
- Verify all dependencies are installed

### Memory corruption errors (double free, tcache)?
- **Immediate fix**: Use sequential mode: `--sequential`
- **Root cause**: Usually from TensorFlow/JAX in multiprocessing
- **Prevention strategies**:
  - Reduce worker count: `--workers 1` or `--workers 2`
  - Use smaller batch sizes: `--batch-size 1`
  - Set environment variables:
    ```bash
    export TF_CPP_MIN_LOG_LEVEL=3
    export CUDA_VISIBLE_DEVICES=""
    export TF_ENABLE_ONEDNN_OPTS=0
    ```
  - Force fork start method (Linux only):
    ```bash
    export PYTHONMULTIPROCESSING_START_METHOD=fork
    ```

### Import errors in workers?
- Ensure PYTHONPATH includes src directory
- Check that all modules can be imported
- Try sequential mode to isolate the issue

### Machine-specific crashes?
Different machines may have varying tolerance for parallel processing due to:
- Different TensorFlow/CUDA installations
- Memory management differences
- Kernel version variations
- Hardware-specific optimizations

**Quick fixes for unstable machines**:
```bash
# Conservative parallel settings
python run_parallel_benchmark.py --datasets general --workers 1 --batch-size 1 --end-episode 10

# Or use sequential processing
python run_parallel_benchmark.py --sequential --datasets general --end-episode 10
```
