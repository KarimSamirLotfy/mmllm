# Utilities Guide

This guide covers the various utility tools and scripts available in the MMLLM project for data visualization, analysis, and debugging.

## Overview

The MMLLM project includes several utility tools that support development, debugging, and analysis workflows:

- **Episode Plotting**: Visualize individual episodes with ground truth actions
- **Performance Analysis**: Analyze benchmark results and performance metrics
- **Data Exploration**: Explore datasets and understand episode structures
- **Debugging Tools**: Tools for troubleshooting and development

## Episode Plotting

The `plot_episode.py` script provides visualization capabilities for individual episodes from the Android in the Wild dataset.

### Basic Usage

**Plot a specific episode by index:**
```bash
python plot_episode.py 5
```

**Plot from a different dataset:**
```bash
python plot_episode.py 10 --dataset general
```

**Save plots to a specific directory:**
```bash
python plot_episode.py 5 --output_dir ./my_plots
```

### Command Line Options

- `episode_index` (required): The episode index to plot (0-based)
- `--dataset`: Dataset to use (choices: `general`, `google_apps`, `install`, `single`, `web_shopping`, default: `google_apps`)
- `--output_dir`: Output directory for plots (default: `./plots`)

### Features

- **Ground Truth Visualization**: Episodes are plotted with ground truth actions highlighted
- **Automatic Directory Creation**: Output directory is created if it doesn't exist
- **Multiple Dataset Support**: Works with all Android in the Wild datasets
- **Simple Interface**: Minimal command-line interface focused on episode visualization

### Examples

**Plot episode 0 from Google Apps dataset:**
```bash
python plot_episode.py 0
```

**Plot episode 15 from Install dataset to custom directory:**
```bash
python plot_episode.py 15 --dataset install --output_dir ./episode_visualizations
```

**Plot multiple episodes (run separately):**
```bash
python plot_episode.py 0 --output_dir ./plots
python plot_episode.py 1 --output_dir ./plots
python plot_episode.py 2 --output_dir ./plots
```

### Output

The generated plots show:
- **Episode screens**: Screenshots from each step
- **Ground truth actions**: Expected user interactions overlaid
- **Action sequences**: Step-by-step progression through the episode
- **Metadata**: Episode information and statistics

## Performance Analysis

### Benchmark Result Analysis

The `performance_comparison.py` script provides tools for analyzing benchmark results.

#### Basic Usage

```bash
# Analyze single benchmark result
python performance_comparison.py --input benchmark_results_20250719_171704.csv

# Compare multiple benchmark runs
python performance_comparison.py \
  --input benchmark_results_run1.csv benchmark_results_run2.csv \
  --labels "OCR Enabled" "OCR Disabled"
```

#### Features

- **Success Rate Analysis**: Calculate and compare success rates across datasets
- **Error Classification**: Categorize and analyze failure modes
- **Performance Metrics**: Timing analysis and resource usage
- **Comparative Analysis**: Side-by-side comparison of different configurations

### Visualization Tools

#### Performance Charts

Generate performance charts from benchmark results:

```bash
# Generate success rate charts
python plot_performance.py --type success_rates --input results.csv

# Generate timing analysis
python plot_performance.py --type timing --input results.csv --output ./charts
```

#### Error Analysis

Analyze error patterns and common failure modes:

```bash
# Generate error distribution charts
python analyze_errors.py --input results.csv --output ./error_analysis
```

## Data Exploration

### Dataset Explorer

Explore the structure and content of Android in the Wild datasets:

```bash
# List available datasets
python explore_datasets.py --list

# Show dataset statistics
python explore_datasets.py --dataset general --stats

# Export episode metadata
python explore_datasets.py --dataset google_apps --export metadata.json
```

### Episode Inspector

Detailed inspection of individual episodes:

```bash
# Inspect episode structure
python inspect_episode.py --dataset general --episode 5

# Export episode data
python inspect_episode.py --dataset general --episode 5 --export episode_5.json

# Compare episodes
python inspect_episode.py --compare --episodes 1,2,3 --dataset general
```

## Debugging Tools

### Configuration Validator

Validate benchmark configurations before running:

```bash
# Validate configuration file
python validate_config.py --config my_benchmark.json

# Check environment setup
python validate_config.py --check-env

# Test API connections
python validate_config.py --test-apis
```

### Agent Debugger

Debug agent behavior and decision-making:

```bash
# Debug single episode with verbose output
python debug_agent.py --dataset general --episode 5 --verbose

# Test OCR pipeline
python debug_agent.py --test-ocr --image screenshot.png

# Analyze agent decisions
python debug_agent.py --dataset general --episode 5 --analyze-decisions
```

### Log Analyzer

Analyze execution logs for debugging:

```bash
# Parse log file for errors
python analyze_logs.py --log logs/20250719_171705.log --errors

# Generate log summary
python analyze_logs.py --log logs/20250719_171705.log --summary

# Extract timing information
python analyze_logs.py --log logs/20250719_171705.log --timing
```

## Batch Processing Tools

### Batch Episode Plotter

Plot multiple episodes in batch:

```bash
# Plot episodes 0-9 from general dataset
python batch_plot.py --dataset general --episodes 0:10 --output ./batch_plots

# Plot all episodes from multiple datasets
python batch_plot.py \
  --datasets general google_apps \
  --episodes 0:5 \
  --output ./comparison_plots
```

### Batch Analysis

Analyze multiple benchmark results:

```bash
# Analyze all results in directory
python batch_analyze.py --input ./benchmark_results --output ./analysis

# Compare configurations
python batch_analyze.py \
  --input ./benchmark_results \
  --group-by run_name \
  --output ./comparison_analysis
```

## Data Conversion Tools

### Format Converters

Convert between different data formats:

```bash
# Convert CSV results to JSON
python convert_results.py --input results.csv --output results.json --format json

# Convert JSON to Excel
python convert_results.py --input results.json --output results.xlsx --format excel

# Merge multiple result files
python convert_results.py --merge results1.csv results2.csv --output merged.csv
```

### Export Tools

Export data for external analysis:

```bash
# Export for R analysis
python export_data.py --input results.csv --format r --output analysis.R

# Export for MATLAB
python export_data.py --input results.csv --format matlab --output analysis.m

# Export for Jupyter notebooks
python export_data.py --input results.csv --format notebook --output analysis.ipynb
```

## Development Utilities

### Test Data Generation

Generate test data for development:

```bash
# Generate mock episode data
python generate_test_data.py --episodes 10 --output ./test_data

# Create benchmark configuration templates
python generate_test_data.py --config-template --output ./configs
```

### Environment Setup

Utilities for environment management:

```bash
# Check system requirements
python check_requirements.py

# Setup development environment
python setup_dev_env.py

# Verify installation
python verify_install.py --full-check
```

## Custom Utilities

### Creating Custom Scripts

The project supports custom utility scripts. Example template:

```python
#!/usr/bin/env python3
"""
Custom utility script template
"""
import argparse
from mmllm.utils import load_dataset, save_results

def main():
    parser = argparse.ArgumentParser(description='Custom utility')
    parser.add_argument('--input', required=True, help='Input file')
    parser.add_argument('--output', required=True, help='Output file')
    args = parser.parse_args()
    
    # Custom processing logic here
    data = load_dataset(args.input)
    results = process_data(data)
    save_results(results, args.output)

if __name__ == '__main__':
    main()
```

### Integration with Main Pipeline

Custom utilities can be integrated with the main benchmarking pipeline:

```bash
# Run custom preprocessing
python custom_preprocess.py --input ./episodes --output ./processed

# Run benchmark with custom configuration
python -m mmllm.main benchmark --config ./custom_config.json

# Run custom post-processing
python custom_postprocess.py --input ./results --output ./final_analysis
```

## Common Use Cases

### Research Workflow

Typical research workflow using utilities:

```bash
# 1. Explore dataset
python explore_datasets.py --dataset general --stats

# 2. Plot sample episodes
python plot_episode.py 0 --dataset general
python plot_episode.py 1 --dataset general

# 3. Run benchmark
python -m mmllm.main benchmark --datasets general --episodes 0:10

# 4. Analyze results
python performance_comparison.py --input results.csv

# 5. Generate visualizations
python plot_performance.py --input results.csv --output ./charts
```

### Development Workflow

Typical development workflow:

```bash
# 1. Validate environment
python check_requirements.py

# 2. Test configuration
python validate_config.py --config test_config.json --test-apis

# 3. Debug single episode
python debug_agent.py --dataset general --episode 0 --verbose

# 4. Run small benchmark
python -m mmllm.main benchmark --datasets general --episodes 0:3 --dry-run

# 5. Analyze logs
python analyze_logs.py --log latest.log --errors
```

### Production Workflow

Production benchmarking workflow:

```bash
# 1. Validate configuration
python validate_config.py --config production_config.json

# 2. Run benchmark
uv run run_parallel_benchmark.py --config production_config.json

# 3. Generate reports
python generate_report.py --input results.csv --output report.html

# 4. Archive results
python archive_results.py --input ./benchmark_results --archive ./archive
```

## Configuration Files

### Utility Configuration

Many utilities support configuration files:

```json
{
  "plotting": {
    "default_output_dir": "./plots",
    "image_format": "png",
    "dpi": 300
  },
  "analysis": {
    "default_metrics": ["success_rate", "avg_steps", "error_rate"],
    "chart_style": "seaborn",
    "export_formats": ["csv", "json"]
  },
  "debugging": {
    "log_level": "DEBUG",
    "verbose_output": true,
    "save_intermediate": true
  }
}
```

### Environment Variables

Utilities respect environment variables:

```env
# Default output directory
MMLLM_OUTPUT_DIR=./outputs

# Default dataset
MMLLM_DEFAULT_DATASET=general

# Plotting settings
MMLLM_PLOT_FORMAT=png
MMLLM_PLOT_DPI=300

# Analysis settings
MMLLM_ANALYSIS_STYLE=seaborn
```

## Troubleshooting

### Common Issues

1. **Plot generation fails**: Check matplotlib backend and display settings
2. **Large dataset memory issues**: Use batch processing or limit episode range
3. **File permission errors**: Verify write permissions for output directories
4. **Missing dependencies**: Run `pip install -e .[dev]` for development dependencies

### Debug Mode

Most utilities support debug mode:

```bash
# Enable debug output
python plot_episode.py 5 --debug

# Verbose logging
python performance_comparison.py --input results.csv --verbose

# Save intermediate files
python analyze_results.py --input results.csv --save-intermediate
```

## Next Steps

- **Advanced Analysis**: Explore statistical analysis tools
- **Custom Visualizations**: Create domain-specific charts
- **Automation**: Set up automated reporting pipelines
- **Integration**: Connect with external analysis tools

For system architecture details, see [System Architecture](SYSTEM_ARCHITECTURE.md).
For benchmarking capabilities, see [Benchmarking Guide](BENCHMARKING.md).
