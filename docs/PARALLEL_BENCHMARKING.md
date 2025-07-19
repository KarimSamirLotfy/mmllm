# Parallel Benchmarking Pipeline

This enhanced benchmarking pipeline supports parallel processing to significantly speed up evaluation of large datasets.

## Key Features

### Parallel Processing
- **Multi-worker processing**: Configure the number of parallel workers
- **Batch processing**: Episodes are processed in configurable batches
- **Result aggregation**: Results from all workers are automatically collected and aggregated
- **Error handling**: Failed batches don't stop the entire benchmark
- **Fallback support**: Sequential processing available as fallback

## CLI Usage

The parallel benchmarking pipeline can be accessed through two different command-line interfaces:

### 1. Integrated CLI (Recommended)

Use the main CLI through the `mmllm` module:

```bash
# Basic benchmark
python -m mmllm benchmark --datasets general --end-episode 20

# Parallel benchmark with custom workers
python -m mmllm benchmark \
  --datasets general google_apps \
  --episodes 0:50 \
  --max-workers 8 \
  --batch-size 10 \
  --ocr-module \
  --output-dir ./benchmark_results \
  --run-name my_benchmark

# Quick development test
python -m mmllm benchmark \
  --datasets general \
  --episodes 0:5 \
  --max-workers 2 \
  --batch-size 2

# Show time estimates without running
python -m mmllm benchmark \
  --datasets general google_apps \
  --episodes 0:100 \
  --estimate-time \
  --dry-run

# Load configuration from file
python -m mmllm benchmark --config ./my_config.json
```

### 2. Standalone Script

Use the dedicated script for more direct control:

```bash
# Basic parallel benchmark
python run_parallel_benchmark.py \
  --datasets general \
  --workers 4 \
  --batch-size 5 \
  --end-episode 20

# High-performance benchmark
python run_parallel_benchmark.py \
  --datasets general google_apps \
  --workers 8 \
  --batch-size 10 \
  --end-episode 100 \
  --max-steps 15 \
  --ocr \
  --output-dir ./results \
  --run-name high_performance_benchmark

# Sequential fallback for debugging
python run_parallel_benchmark.py \
  --sequential \
  --datasets general \
  --end-episode 10 \
  --verbose
```

### CLI Options Reference

#### Common Options (Both CLIs)
- `--datasets`: Choose datasets (`general`, `google_apps`, `install`, `single`, `web_shopping`)
- `--start-episode`, `--end-episode`: Episode range
- `--episodes`: Episode range in `start:end` format (integrated CLI only)
- `--max-steps`: Maximum steps per episode
- `--output-dir`: Output directory for results
- `--run-name`: Custom name for the benchmark run
- `--verbose`: Enable detailed logging

#### Parallel Processing Options
- `--max-workers` / `--workers`: Number of parallel workers (default: 4)
- `--batch-size`: Episodes per batch (default: 5)

#### Agent Configuration
- `--ocr-module` / `--ocr`: Enable OCR module
- `--no-ocr`: Disable OCR module (integrated CLI only)

#### Utility Options (Integrated CLI)
- `--config`: Load configuration from JSON file
- `--save-config`: Save configuration to JSON file
- `--dry-run`: Show configuration without running
- `--estimate-time`: Show time estimates
- `--recommended`: Use recommended dataset configurationshe entire benchmark
- **Fallback support**: Sequential processing available as fallback

## Results and Output

### Generated Files

The benchmarking pipeline generates several output files:

```
./benchmark_results/
├── benchmark_results_20250719_143531.csv      # Detailed episode results
├── benchmark_report_20250719_143531.json      # Comprehensive report
├── performance_charts_20250719_143531.png     # Visualizations
└── config_20250719_143531.json               # Configuration used
```

### Understanding the Output

#### CSV Results File
Contains detailed results for each episode:
- `episode_id`: Unique identifier for the episode
- `dataset_name`: Source dataset
- `success_rate`: Overall success rate for the episode
- `step_accuracy`: Accuracy of individual steps
- `total_steps`: Total number of steps in the episode
- `steps_completed`: Number of steps successfully completed
- `goal`: Episode objective

#### JSON Report
Comprehensive report including:
- **Overall metrics**: Accuracy, success rate, precision, recall, F1 score
- **Dataset breakdown**: Per-dataset performance metrics
- **Configuration**: All parameters used for the benchmark
- **Episode results**: Detailed results for each episode
- **Parallel processing stats**: Worker count, batch size, processing time

#### Example Summary Output
```
=== Benchmark Results Summary ===
Total episodes processed: 45
Total steps: 380
Overall accuracy: 67.4%
Overall success rate: 62.2%
Precision: 0.674
Recall: 0.622
F1 Score: 0.647

Dataset breakdown:
  general: Accuracy=65.2%, F1=0.634, Episodes=25
  google_apps: Accuracy=70.5%, F1=0.663, Episodes=20

Generated files:
  csv_export: ./benchmark_results/benchmark_results_20250719_143531.csv
  visualization: ./benchmark_results/performance_charts_20250719_143531.png
  json_report: ./benchmark_results/benchmark_report_20250719_143531.json
```

### Viewing Results

```bash
# View CSV results with headers
head -5 ./benchmark_results/benchmark_results_*.csv

# View JSON report summary
jq '.overall_metrics' ./benchmark_results/benchmark_report_*.json

# Check parallel processing stats
jq '.report_metadata.parallel_processing' ./benchmark_results/benchmark_report_*.json

# List all generated files
ls -la ./benchmark_results/
```

## Configuration Parameters

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
