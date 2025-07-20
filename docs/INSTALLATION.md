# Installation Guide

This guide covers the installation and initial setup of the MMLLM project.

## Prerequisites

- Python 3.8 or higher
- Git
- UV package manager (recommended) or pip

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mmllm
```

### 2. Install Dependencies

#### Using UV (Recommended)

```bash
# Install dependencies using uv
uv sync

# Install the project in editable mode
uv pip install -e .
```

#### Using Pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY="sk-None-******"

# Tavily Search API
TAVILY_API_KEY="tvly-dev-*******"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY="*******"
AZURE_OPENAI_ENDPOINT="https://*******.openai.azure.com/"
AZURE_API_VERSION="2024-12-01-preview"
# AZURE_DEPLOYMENT="gpt-4.1"
AZURE_DEPLOYMENT="o4-mini"

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="lsv2_*******"
LANGSMITH_PROJECT="******"
```

### 4. Verify Installation

Test your installation by running:

```bash
python -m mmllm.main --help
```

## Configuration Notes

### API Keys

- **OpenAI API Key**: Required for GPT models
- **Azure OpenAI**: Alternative to OpenAI, configure endpoint and deployment
- **Tavily API**: Required for web search functionality
- **LangSmith**: Optional, for tracing and debugging

### Model Selection

You can switch between different models by changing the `AZURE_DEPLOYMENT` variable:
- `o4-mini` - GPT-4 Omni Mini (faster, cost-effective)
- `gpt-4` - GPT-4 (higher quality, more expensive)

## Troubleshooting

### Common Issues

1. **Dependencies not found**: Ensure you're using the correct Python environment
2. **API key errors**: Verify your `.env` file is properly configured
3. **Permission errors**: Make sure you have write access to the project directory

### System Requirements

- **Memory**: At least 8GB RAM recommended
- **Storage**: 5GB free space for datasets and results
- **Network**: Stable internet connection for API calls

### Dataset Access

The system automatically downloads Android in the Wild datasets via TensorFlow Datasets. Ensure:
- Sufficient disk space for dataset downloads
- Network access to download datasets
- TensorFlow and related dependencies are properly installed

## Next Steps

After installation, you can:
- Run basic benchmarks: See [Benchmarking Guide](BENCHMARKING.md)
- Use parallel processing: See [Parallel Benchmarking Guide](PARALLEL_BENCHMARKING.md)
- Explore utilities: See [Utilities Guide](UTILITIES.md)
