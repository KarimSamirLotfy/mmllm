# MMLLM - Multi-Modal Language Learning Model

A multi-agent system for Android in the Wild (AiTW) dataset evaluation and benchmarking.

## Overview

MMLLM is a sophisticated evaluation framework designed to assess multi-modal AI agents on Android user interface interaction tasks. The system combines computer vision, natural language processing, and structured reasoning to interact with Android applications through the Android in the Wild dataset.

### Key Features

- **Multi-Agent Architecture**: Specialized agents for vision, action planning, and strategy
- **OCR Integration**: Advanced text recognition for enhanced UI understanding
- **Parallel Processing**: High-performance benchmarking with configurable parallelization
- **Comprehensive Evaluation**: Support for multiple datasets and evaluation metrics
- **Research-Ready**: Tools and utilities for analysis, visualization, and reporting

## Quick Start

### Basic Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd mmllm

# Install dependencies
uv sync && uv pip install -e .

# Set up environment variables (see Installation Guide)
cp .env.example .env
# Edit .env with your API keys
```

### Run Your First Benchmark

```bash
# Simple benchmark
python -m mmllm.main benchmark --datasets google_apps --episodes 0:3

# Parallel benchmark (recommended)
uv run run_parallel_benchmark.py --datasets general --end-episode 5 --workers 4
```

## Documentation

### 📚 Core Guides

| Guide | Description | Use Case |
|-------|-------------|----------|
| **[Installation Guide](docs/INSTALLATION.md)** | Complete setup instructions, dependencies, and environment configuration | Setting up MMLLM for the first time |
| **[Benchmarking Guide](docs/BENCHMARKING.md)** | Standard benchmarking system using the main CLI | Running basic evaluations and experiments |
| **[Parallel Benchmarking Guide](docs/PARALLEL_BENCHMARKING.md)** | High-performance parallel evaluation system | Large-scale benchmarks and research studies |
| **[System Architecture](docs/SYSTEM_ARCHITECTURE.md)** | Technical architecture, components, and data flow | Understanding system internals and customization |
| **[Utilities Guide](docs/UTILITIES.md)** | Tools for visualization, analysis, and debugging | Data exploration and result analysis |

### 📋 Quick Reference

| Task | Command | Documentation |
|------|---------|---------------|
| Install system | `uv sync && uv pip install -e .` | [Installation](docs/INSTALLATION.md#installation-steps) |
| Basic benchmark | `python -m mmllm.main benchmark --datasets general --episodes 0:5` | [Benchmarking](docs/BENCHMARKING.md#quick-start) |
| Parallel benchmark | `uv run run_parallel_benchmark.py --datasets general --end-episode 10` | [Parallel Benchmarking](docs/PARALLEL_BENCHMARKING.md#quick-start) |
| Plot episode | `python plot_episode.py 5 --dataset general` | [Utilities](docs/UTILITIES.md#episode-plotting) |
| Analyze results | `python performance_comparison.py --input results.csv` | [Utilities](docs/UTILITIES.md#performance-analysis) |

## Available Datasets

The system supports evaluation on multiple Android in the Wild dataset variants:

- **`general`** - General Android interactions and navigation
- **`google_apps`** - Google applications specific tasks
- **`install`** - App installation and setup scenarios
- **`single`** - Single-step interaction tasks
- **`web_shopping`** - Web shopping and e-commerce workflows

## Key Capabilities

### 🔬 Research Features

- **OCR vs No-OCR Comparison**: Evaluate impact of text recognition
- **Model Comparison**: Test different LLM backends (GPT-4, GPT-4 Omni Mini)
- **Prompt Strategy Testing**: Compare standard vs Android-specific prompts
- **Stateful vs Stateless**: Analyze impact of conversation history

### ⚡ Performance Features

- **Parallel Processing**: Multi-worker evaluation for faster results
- **Batch Processing**: Configurable batch sizes for optimization
- **Resource Management**: Dynamic worker allocation and error recovery
- **Time Estimation**: Preview benchmark duration before execution

### 📊 Analysis Tools

- **Episode Visualization**: Plot episodes with ground truth overlays
- **Performance Metrics**: Success rates, timing analysis, error classification
- **Comparative Analysis**: Side-by-side benchmark comparisons
- **Export Capabilities**: Multiple output formats for further analysis

## Getting Started by Use Case

### 🔬 Researchers

1. **Setup**: Follow [Installation Guide](docs/INSTALLATION.md)
2. **Explore Data**: Use [episode plotting](docs/UTILITIES.md#episode-plotting) to understand datasets
3. **Run Experiments**: Use [parallel benchmarking](docs/PARALLEL_BENCHMARKING.md) for comprehensive evaluation
4. **Analyze Results**: Use [analysis tools](docs/UTILITIES.md#performance-analysis) for insights

### 👩‍💻 Developers

1. **Setup**: Follow [Installation Guide](docs/INSTALLATION.md)
2. **Understand Architecture**: Read [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
3. **Start Simple**: Use [basic benchmarking](docs/BENCHMARKING.md) for development
4. **Debug**: Use [debugging tools](docs/UTILITIES.md#debugging-tools) for troubleshooting

### 📈 Analysts

1. **Setup**: Follow [Installation Guide](docs/INSTALLATION.md)
2. **Run Benchmarks**: Use [benchmarking guides](docs/BENCHMARKING.md) to generate data
3. **Visualize**: Use [plotting tools](docs/UTILITIES.md#episode-plotting) for visualization
4. **Export**: Use [conversion tools](docs/UTILITIES.md#data-conversion-tools) for external analysis

## Example Workflows

### Research Benchmark Workflow

```bash
# 1. OCR vs No-OCR comparison
uv run run_parallel_benchmark.py --run-name "NO-OCR" --datasets general --end-episode 10
uv run run_parallel_benchmark.py --run-name "OCR" --ocr --datasets general --end-episode 10

# 2. Analyze results
python performance_comparison.py --input benchmark_results/

# 3. Generate visualizations
python plot_performance.py --input results.csv --output ./charts
```

### Development Testing Workflow

```bash
# 1. Quick validation
python -m mmllm.main benchmark --datasets general --episodes 0:3 --dry-run

# 2. Debug single episode
python debug_agent.py --dataset general --episode 0 --verbose

# 3. Small-scale test
python -m mmllm.main benchmark --datasets general --episodes 0:5
```

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended for parallel processing
- **Storage**: 5GB free space for datasets and results
- **Network**: Stable internet connection for API calls and dataset downloads

## API Requirements

- **OpenAI API Key** or **Azure OpenAI** credentials for LLM inference
- **Tavily API Key** for web search functionality (optional)
- **LangSmith API Key** for tracing and monitoring (optional)

See [Installation Guide](docs/INSTALLATION.md#environment-configuration) for detailed setup instructions.

## Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## Support

### Documentation Issues

If you find issues with the documentation:
1. Check the relevant guide in the [docs/](docs/) directory
2. Search existing issues
3. Create a new issue with specific details

### Technical Issues

For technical problems:
1. Check [troubleshooting sections](docs/INSTALLATION.md#troubleshooting) in the guides
2. Use [debugging tools](docs/UTILITIES.md#debugging-tools)
3. Check system requirements and dependencies

### Feature Requests

For new features or enhancements:
1. Review [System Architecture](docs/SYSTEM_ARCHITECTURE.md) for extension points
2. Check existing feature requests
3. Create a detailed feature request issue

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Android in the Wild dataset from Google Research
- TensorFlow Datasets for data infrastructure
- OpenAI and Azure OpenAI for language model capabilities

---

**Next Steps**: Start with the [Installation Guide](docs/INSTALLATION.md) to set up your environment, then explore the specific guides based on your use case.