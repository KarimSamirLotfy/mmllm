# MMLLM - Multi-Modal Language Learning Model

A multi-agent system for Android in the Wild (AiTW) dataset evaluation and benchmarking.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mmllm

# Install dependencies using uv
uv sync

# Or install in development mode
uv pip install -e .
```

### Basic Usage

Run the benchmarking pipeline with default settings:

```bash
python -m mmllm.main benchmark
```

## Benchmarking

The benchmarking system provides comprehensive evaluation across multiple Android in the Wild datasets.

### Basic Benchmark Commands

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

### Available Datasets

- `general` - General Android interactions
- `google_apps` - Google applications specific tasks
- `install` - App installation scenarios  
- `single` - Single-step tasks
- `web_shopping` - Web shopping workflows

### Command Line Options

#### Dataset Selection
- `--datasets`: Choose one or more datasets (default: `google_apps`)
- `--episodes`: Episode range in format "start:end" (e.g., "0:5")
- `--start-episode`: Starting episode index (default: 0)
- `--end-episode`: Ending episode index (exclusive)

#### Agent Configuration
- `--ocr-module`: Enable OCR module (default: True)
- `--no-ocr`: Disable OCR module
- `--max-steps`: Maximum steps per episode (default: 10)

#### Output Options
- `--output-dir`: Output directory for results (default: `./benchmark_results`)
- `--run-name`: Custom name for this benchmark run
- `--csv-only`: Only generate CSV output, skip visualizations
- `--no-visualizations`: Skip visualization generation

#### Configuration Management
- `--config`: Load configuration from JSON file
- `--save-config`: Save configuration to JSON file before running

#### Utility Options
- `--dry-run`: Show configuration and estimates without running
- `--estimate-time`: Show time estimates for the benchmark
- `--recommended`: Use predefined configurations (`general`, `quick`, `comprehensive`, `ui_focused`)

### Example Commands

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

### Output

Benchmark results are saved to the specified output directory and include:

- **CSV Report**: Detailed episode-by-episode results
- **JSON Report**: Structured benchmark data
- **Visualizations**: Performance charts and graphs (unless disabled)
- **Configuration**: The exact configuration used for the benchmark

Results are timestamped and organized by run for easy comparison.

### Configuration Files

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


## Episode Plotting

For visualizing individual episodes from the Android in the Wild dataset, use the `plot_episode.py` script:

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

### Available Options

- `episode_index` (required): The episode index to plot (0-based)
- `--dataset`: Dataset to use (choices: `general`, `google_apps`, `install`, `single`, `web_shopping`, default: `google_apps`)
- `--output_dir`: Output directory for plots (default: `./plots`)

### Features

- **Ground Truth Visualization**: The script plots the episode with ground truth actions highlighted
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

The generated plots will show the episode screens with ground truth actions overlaid, making it easy to understand the expected user interactions for each step.

## Troubleshooting

- Ensure all dependencies are installed correctly
- Check that TensorFlow datasets are accessible
- Verify sufficient disk space for results output
- Use `--dry-run` to validate configuration before running


# Benchmarking results 
## What we want to test?
1. OCR vs NO OCR 
* all datasets, 10 episodes, max_steps=5

2. different models o-mini vs gpt4
* general, 10 spisodes, max_steps=5

3. Some visulasaiotns
* What is most comman error
* average devation from user input to model input
* 