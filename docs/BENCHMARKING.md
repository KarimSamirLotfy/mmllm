# Benchmarking Guide

This guide covers the standard benchmarking system for evaluating models on the Android in the Wild (AiTW) dataset.

## Overview

The benchmarking system provides comprehensive evaluation across multiple Android in the Wild datasets with support for various configuration options and output formats.

## Quick Start

Run the benchmarking pipeline with default settings:

```bash
python -m mmllm.main benchmark
```

## Basic Commands

### Dataset-Specific Benchmarks

**Run benchmark on specific dataset with episode range:**
```bash
python -m mmllm.main benchmark --datasets general --episodes 0:1
```

**Run benchmark on multiple datasets:**
```bash
python -m mmllm.main benchmark --datasets general google_apps install
```

**Quick benchmark with limited episodes:**
```bash
python -m mmllm.main benchmark --datasets google_apps --episodes 0:3 --max-steps 5
```

## Available Datasets

- `general` - General Android interactions
- `google_apps` - Google applications specific tasks
- `install` - App installation scenarios  
- `single` - Single-step tasks
- `web_shopping` - Web shopping workflows

## Command Line Options

### Dataset Selection
- `--datasets`: Choose one or more datasets (default: `google_apps`)
- `--episodes`: Episode range in format "start:end" (e.g., "0:5")
- `--start-episode`: Starting episode index (default: 0)
- `--end-episode`: Ending episode index (exclusive)

### Agent Configuration
- `--ocr-module`: Enable OCR module (default: True)
- `--no-ocr`: Disable OCR module
- `--max-steps`: Maximum steps per episode (default: 10)

### Output Options
- `--output-dir`: Output directory for results (default: `./benchmark_results`)
- `--run-name`: Custom name for this benchmark run
- `--csv-only`: Only generate CSV output, skip visualizations
- `--no-visualizations`: Skip visualization generation

### Configuration Management
- `--config`: Load configuration from JSON file
- `--save-config`: Save configuration to JSON file before running

### Utility Options
- `--dry-run`: Show configuration and estimates without running
- `--estimate-time`: Show time estimates for the benchmark
- `--recommended`: Use predefined configurations (`general`, `quick`, `comprehensive`, `ui_focused`)

## Example Commands

### Basic Examples

**Basic evaluation on Google Apps dataset:**
```bash
python -m mmllm.main benchmark --datasets google_apps --episodes 0:5
```

**Comprehensive benchmark across all datasets:**
```bash
python -m mmllm.main benchmark --datasets general google_apps install single web_shopping --episodes 0:10
```

**Quick test run:**
```bash
python -m mmllm.main benchmark --datasets google_apps --episodes 0:1 --max-steps 3 --dry-run
```

### Configuration Examples

**Benchmark without OCR module:**
```bash
python -m mmllm.main benchmark --datasets general --episodes 0:2 --no-ocr
```

**Save configuration for later use:**
```bash
python -m mmllm.main benchmark --datasets google_apps --episodes 0:5 --save-config my_benchmark_config.json
```

**Load and run from saved configuration:**
```bash
python -m mmllm.main benchmark --config my_benchmark_config.json
```

**Use recommended settings for quick testing:**
```bash
python -m mmllm.main benchmark --recommended quick
```

## Output and Results

Benchmark results are saved to the specified output directory and include:

- **CSV Report**: Detailed episode-by-episode results
- **JSON Report**: Structured benchmark data
- **Visualizations**: Performance charts and graphs (unless disabled)
- **Configuration**: The exact configuration used for the benchmark

Results are timestamped and organized by run for easy comparison.

## Configuration Files

You can create reusable benchmark configurations in JSON format:

```json
{
  "dataset_names": ["google_apps", "general"],
  "episode_range": {"start": 0, "end": 5},
  "max_steps_per_episode": 10,
  "enable_ocr": true,
  "output_dir": "./my_results"
}
```

### Configuration Parameters

- `dataset_names`: List of datasets to evaluate
- `episode_range`: Start and end episode indices
- `max_steps_per_episode`: Maximum steps per episode
- `enable_ocr`: Whether to use OCR module
- `output_dir`: Directory for saving results

## Legacy Commands

The system also supports legacy demo and evaluation commands:

**Run demo mode:**
```bash
python -m mmllm.main demo
```

**Run evaluation mode:**
```bash
python -m mmllm.main evaluate --dataset google_apps --episodes 3 --max-steps 10
```

## Performance Considerations

- **Memory Usage**: Larger datasets and longer episodes require more memory
- **API Costs**: Consider rate limits and costs when running extensive benchmarks
- **Time Estimates**: Use `--estimate-time` to preview benchmark duration
- **Parallel Processing**: For faster execution, see [Parallel Benchmarking Guide](PARALLEL_BENCHMARKING.md)

## Troubleshooting

- Ensure all dependencies are installed correctly
- Check that TensorFlow datasets are accessible
- Verify sufficient disk space for results output
- Use `--dry-run` to validate configuration before running

## Next Steps

- For faster execution: [Parallel Benchmarking Guide](PARALLEL_BENCHMARKING.md)
- For episode visualization: [Utilities Guide](UTILITIES.md)
- For system details: [System Architecture](SYSTEM_ARCHITECTURE.md)
