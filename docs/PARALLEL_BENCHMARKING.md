# Parallel Benchmarking Pipeline

This enhanced benchmarking pipeline supports parallel processing to significantly speed up evaluation of large datasets. This is the primary benchmarking system used for generating reports and comprehensive evaluations.

## Overview

The parallel benchmarking system provides:
- **High-performance evaluation**: Process multiple episodes simultaneously
- **Scalable architecture**: Configure workers and batch sizes based on system resources
- **Comprehensive testing**: Support for various agent configurations and datasets
- **Research-ready output**: Generate detailed reports suitable for analysis

## Key Features

### Parallel Processing
- **Multi-worker processing**: Configure the number of parallel workers
- **Batch processing**: Episodes are processed in configurable batches
- **Result aggregation**: Results from all workers are automatically collected and aggregated
- **Error handling**: Failed batches don't stop the entire benchmark
- **Fallback support**: Sequential processing available as fallback

## Quick Start

### Basic Usage

```bash
# Run parallel benchmark with default settings
uv run run_parallel_benchmark.py --datasets general --end-episode 10
```

### Command Line Interface

```bash
usage: run_parallel_benchmark.py [-h]
                                 [--datasets {general,google_apps,install,single,web_shopping} [{general,google_apps,install,single,web_shopping} ...]]
                                 [--start-episode START_EPISODE]
                                 [--end-episode END_EPISODE]
                                 [--max-steps MAX_STEPS]
                                 [--workers WORKERS]
                                 [--batch-size BATCH_SIZE]
                                 [--output-dir OUTPUT_DIR]
                                 [--run-name RUN_NAME] [--ocr]
                                 [--prompt-with-android-tree]
                                 [--add-image-history]
                                 [--sequential] [--verbose]
```

### Command Line Options

- `--datasets`: Datasets to process (choices: general, google_apps, install, single, web_shopping; default: general)
- `--start-episode`: Starting episode index (default: 0)
- `--end-episode`: Ending episode index (default: None for all episodes)
- `--max-steps`: Maximum steps per episode (default: 10)
- `--workers`: Number of parallel workers (default: 4)
- `--batch-size`: Episodes per batch (default: 5)
- `--output-dir`: Output directory for results (default: ./benchmark_results)
- `--run-name`: Name for this benchmark run (default: parallel_benchmark)
- `--ocr`: Enable OCR module (default: False)
- `--prompt-with-android-tree`: Use Android tree prompt instead of default prompt (default: False)
- `--add-image-history`: Include image history in agent context for multi-step reasoning (default: False)
- `--sequential`: Use sequential processing instead of parallel (default: False)
- `--verbose`: Enable verbose logging (default: False)

## Example Commands

### Basic Benchmarks

**Standard benchmark with OCR:**
```bash
uv run run_parallel_benchmark.py --run-name "standard-ocr" --ocr --end-episode 10 --workers 4
```

**Multi-dataset benchmark:**
```bash
uv run run_parallel_benchmark.py \
  --datasets general google_apps install single web_shopping \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1
```

**High-performance benchmark:**
```bash
uv run run_parallel_benchmark.py \
  --datasets general \
  --end-episode 50 \
  --workers 8 \
  --batch-size 5 \
  --ocr \
  --max-steps 15
```

## Research Benchmarks

These are the benchmark configurations used for generating the research report:

### 1. OCR vs NO OCR Comparison

Test the impact of OCR module across all datasets with 10 episodes each:

```bash
# Without OCR
uv run run_parallel_benchmark.py \
  --run-name "NO-OCR" \
  --end-episode 5 \
  --workers 5 \
  --batch-size 1 \
  --datasets general google_apps install single web_shopping

# With OCR
uv run run_parallel_benchmark.py \
  --run-name "OCR" \
  --ocr \
  --end-episode 5 \
  --workers 5 \
  --batch-size 1 \
  --datasets general google_apps install single web_shopping
```

### 2. Model Comparison (GPT-4 vs GPT-4 Omni Mini)

Change model deployment in `.env` file between runs:

```bash
# GPT-4 (set AZURE_DEPLOYMENT="gpt-4" in .env)
uv run run_parallel_benchmark.py \
  --run-name "GPT4" \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single web_shopping

# GPT-4 Omni Mini (set AZURE_DEPLOYMENT="o4-mini" in .env)
uv run run_parallel_benchmark.py \
  --run-name "O-Mini" \
  --ocr \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single 
```

### 3. Stateful vs Stateless Agent Comparison

Test the impact of image history on agent performance:

```bash
# Stateless (no image history)
uv run run_parallel_benchmark.py \
  --run-name "stateless" \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single 

# Stateful (with image history)
uv run run_parallel_benchmark.py \
  --add-image-history \
  --run-name "stateful" \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single 
```

### 4. Android Tree Prompt vs Standard Prompt

Test different prompting strategies:

```bash
# Standard prompt
uv run run_parallel_benchmark.py \
  --run-name "standard-prompt" \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single 

# Android tree prompt
uv run run_parallel_benchmark.py \
  --prompt-with-android-tree \
  --run-name "android-tree-prompt" \
  --end-episode 10 \
  --workers 10 \
  --batch-size 1 \
  --datasets general google_apps install single 
```

## Performance Optimization

### Time Estimates

- **Steps per Worker**: Approximately 3 minutes per step
- **Example**: 50 steps on 10 workers = ~15 minutes total
- **Scaling**: Linear scaling with number of workers (up to system limits)

### Worker Configuration

#### Recommended Worker Settings

- **Development/Testing**: 2-4 workers
- **Standard Benchmarks**: 4-8 workers
- **High-Performance**: 8-16 workers (depends on system resources)

#### Batch Size Guidelines

- **Small batches (1-2)**: Better error isolation, more overhead
- **Medium batches (5-10)**: Good balance of performance and reliability
- **Large batches (10+)**: Maximum throughput, potential memory issues

### System Requirements

- **CPU**: Multi-core processor (8+ cores recommended for high-performance)
- **Memory**: 16GB+ RAM for large-scale benchmarks
- **Network**: Stable internet for API calls
- **Storage**: SSD recommended for dataset access

## Output and Results

### Generated Files

- **CSV Report**: Detailed episode-by-episode results (`benchmark_results_TIMESTAMP.csv`)
- **JSON Report**: Structured benchmark data (`benchmark_report_TIMESTAMP.json`)
- **Configuration**: The exact configuration used (`config_TIMESTAMP.json`)
- **Logs**: Execution logs (`TIMESTAMP.log`)

### Result Analysis

Results include:
- **Success rates** per dataset and episode
- **Error classifications** and common failure modes
- **Performance metrics** (steps taken, accuracy scores)
- **Timing information** for performance analysis

## Troubleshooting

### Common Issues

1. **Worker crashes**: Reduce batch size or number of workers
2. **Memory errors**: Lower worker count or episode range
3. **API rate limits**: Reduce workers or add delays
4. **Network timeouts**: Check internet connection and API endpoints

### Debugging Options

```bash
# Sequential processing for debugging
uv run run_parallel_benchmark.py --sequential --verbose --end-episode 1

# Single worker with verbose output
uv run run_parallel_benchmark.py --workers 1 --verbose --end-episode 5
```

### Performance Monitoring

Monitor system resources during execution:
- **CPU usage**: Should be near maximum with proper parallelization
- **Memory usage**: Watch for memory leaks in long runs
- **Network I/O**: Monitor API call patterns
- **Disk I/O**: Dataset loading can be I/O intensive

## Next Steps

- **Result Analysis**: Use visualization tools to analyze benchmark results
- **Configuration Tuning**: Adjust settings based on initial benchmark results
- **Custom Experiments**: Modify agent configurations for specific research questions
- **Integration**: Incorporate results into larger evaluation pipelines

For visualization and analysis tools, see [Utilities Guide](UTILITIES.md).



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
- `--prompt-with-android-tree`: Use Android tree prompt instead of default prompt
- `--add-image-history`: Include image history in agent context for multi-step reasoning

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

The `BenchmarkConfig` class includes parallel processing and agent configuration parameters:

```python
@dataclass
class BenchmarkConfig:
    # Core configuration
    dataset_names: List[str]
    ocr_module: bool
    start_episode: int
    end_episode: Optional[int]
    output_dir: str
    run_name: str
    max_steps_per_episode: int = 10
    
    # Parallel processing parameters
    max_workers: int = 4      # Number of parallel workers
    batch_size: int = 5       # Episodes per batch for worker processing
    
    # Agent configuration parameters
    prompt_with_android_tree: bool = False  # Use Android tree prompt
    add_image_history: bool = False         # Include image history context
```

### Agent Configuration Options

#### OCR Module (`ocr_module`)
- **Default**: `False`
- **Description**: Enables OCR text extraction and UI element processing
- **Use Case**: Better understanding of UI text and structure

#### Android Tree Prompt (`prompt_with_android_tree`)
- **Default**: `False`
- **Description**: Uses Android accessibility tree prompt instead of default prompt
- **Use Case**: Leverages Android's accessibility information for more precise UI navigation

#### Image History (`add_image_history`)
- **Default**: `False`
- **Description**: Includes previous images in agent context for multi-step reasoning
- **Use Case**: Helps agent understand UI transitions and maintain context across steps

## Usage Examples

### Basic Parallel Usage

```python
from src.mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from src.mmllm.evaluation.types import BenchmarkConfig

# Configure parallel processing with enhanced agent
config = BenchmarkConfig(
    dataset_names=['general', 'google_apps'],
    ocr_module=True,
    prompt_with_android_tree=False,  # Use default prompt
    add_image_history=True,          # Enable image history for better context
    start_episode=0,
    end_episode=50,
    output_dir='./results',
    run_name='enhanced_parallel_benchmark',
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
