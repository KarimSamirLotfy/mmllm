# Parallel Benchmarking Pipeline

This enhanced benchmarking pipeline supports parallel processing to significantly speed up evaluation of large datasets.

## Key Features

### Parallel Processing
- **Multi-worker processing**: Configure the number of parallel workers
- **Batch processing**: Episodes are processed in configurable batches
- **Result aggregation**: Results from all workers are automatically collected and aggregated
- **Error handling**: Failed batches don't stop the entire benchmark
- **Fallback support**: Sequential processing available as fallback

### Configuration Parameters

The `BenchmarkConfig` class now includes parallel processing parameters:

```python
@dataclass
class BenchmarkConfig:
    # ... existing parameters ...
    max_workers: int = 4      # Number of parallel workers
    batch_size: int = 5       # Episodes per batch for worker processing
```

## Usage Examples

### Basic Parallel Usage

```python
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

# Configure parallel processing
config = BenchmarkConfig(
    dataset_names=['general', 'google_apps'],
    ocr_module=True,
    start_episode=0,
    end_episode=50,
    output_dir='./results',
    run_name='parallel_benchmark',
    max_workers=8,      # Use 8 parallel workers
    batch_size=5        # Process 5 episodes per batch
)

# Run parallel benchmark
pipeline = BenchmarkingPipeline(config)
results = pipeline.run_benchmark()  # Uses parallel processing by default

# Generate report with parallel processing stats
report = pipeline.generate_comprehensive_report(results)
```

### Sequential Fallback

```python
# If parallel processing fails, use sequential fallback
try:
    results = pipeline.run_benchmark()  # Parallel
except Exception as e:
    print(f"Parallel processing failed: {e}")
    results = pipeline.run_benchmark_sequential()  # Sequential fallback
```

## Performance Optimization

### Worker Configuration
- **CPU-bound tasks**: Set `max_workers` to number of CPU cores (typically 4-16)
- **Memory constraints**: Reduce `batch_size` if running out of memory
- **I/O-bound tasks**: Can use more workers than CPU cores

### Batch Size Tuning
- **Small batches (1-3)**: Better load balancing, more overhead
- **Medium batches (5-10)**: Good balance for most cases
- **Large batches (15+)**: Less overhead, potential load imbalance

### Recommended Settings

| Dataset Size | max_workers | batch_size | Use Case |
|-------------|-------------|------------|----------|
| Small (1-20 episodes) | 2-4 | 2-3 | Development/testing |
| Medium (20-100 episodes) | 4-8 | 5-8 | Regular evaluation |
| Large (100+ episodes) | 8-16 | 10-15 | Production benchmarks |

## Architecture Overview

```
BenchmarkingPipeline
├── Dataset Loading (Sequential)
├── Episode Batching (Sequential)
├── Parallel Processing
│   ├── Worker 1: process_episode_batch()
│   ├── Worker 2: process_episode_batch()
│   ├── Worker N: process_episode_batch()
│   └── Result Collection
└── Report Generation (Sequential)
```

### Worker Process Flow

Each worker process:
1. Initializes its own agent and episode loader
2. Processes a batch of episodes sequentially within the worker
3. Returns results to the main process
4. Main process aggregates all worker results

## Error Handling

The parallel pipeline includes robust error handling:

- **Worker failures**: Individual worker failures don't stop other workers
- **Episode failures**: Failed episodes are logged but don't stop the batch
- **Batch failures**: Failed batches are logged and empty results are used
- **Complete failure**: Automatic fallback to sequential processing

## Monitoring and Logging

Enhanced logging provides visibility into parallel execution:

```
INFO - Starting parallel benchmark execution with 8 workers...
INFO - Created 10 batches for general (50 total episodes)
INFO - Worker 12345 processing episode 5
INFO - Completed batch 1/10: 5 episodes
INFO - Parallel benchmark completed: 45 total episodes processed
```

## Performance Comparison

Use the included performance comparison script:

```bash
python performance_comparison.py
```

Expected output:
```
PERFORMANCE COMPARISON RESULTS
========================================
Sequential time: 120.45 seconds
Parallel time:   35.23 seconds
Speedup:         3.42x
Workers used:    4
Episodes:        20
✅ Parallel processing is 3.42x faster!
```

## Troubleshooting

### Common Issues

1. **Memory Usage**: Reduce `batch_size` if workers run out of memory
2. **Too Many Workers**: Don't exceed 2x your CPU core count
3. **Pickle Errors**: Ensure all objects passed to workers are serializable
4. **Import Errors**: Make sure all modules are properly importable in worker processes

### Performance Not Improving

- Check CPU utilization during parallel runs
- Verify I/O isn't the bottleneck (dataset loading)
- Consider if overhead exceeds parallelization benefits for small datasets
- Try different batch_size values

### Debugging

For debugging, use sequential processing:
```python
results = pipeline.run_benchmark_sequential()
```

This provides clearer error messages and easier debugging than parallel execution.
