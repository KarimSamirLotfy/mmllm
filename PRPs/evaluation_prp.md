name: "Evaluation Pipeline PRP - Benchmarking System for AiTW Dataset"
description: |
  Complete implementation plan for a comprehensive benchmarking pipeline that evaluates model performance on Android-in-the-Wild datasets with configurable parameters, detailed metrics, CSV/JSON reporting, and visualization capabilities.

---

## Goal
Build a comprehensive benchmarking pipeline that evaluates model performance on Android-in-the-Wild (AiTW) datasets, providing detailed metrics (accuracy, success rate, F1, recall), configurable execution parameters (dataset selection, ocr_module toggle), and multiple output formats (CSV, JSON, visualizations) for reproducible research and performance analysis.

## Why
- **Research Value**: Enable systematic evaluation and comparison of different model configurations
- **Performance Tracking**: Monitor model improvements over time with consistent metrics
- **Configuration Testing**: Compare performance with/without OCR module and across different datasets
- **Reproducibility**: Ensure experiments can be replicated with saved configurations
- **Visualization**: Provide clear visual insights into model performance patterns across datasets

## What
A benchmarking system that:
1. Runs the model on configurable AiTW dataset subsets
2. Tracks step-by-step accuracy and overall success rates
3. Calculates comprehensive metrics (F1, recall, precision)
4. Exports results to CSV and JSON formats
5. Generates visualization charts comparing performance across datasets/configurations
6. Saves configuration files for reproducibility

### Success Criteria
- [ ] Successfully runs evaluation on all 5 AiTW dataset types (general, google_apps, install, single, web_shopping)
- [ ] Produces accurate step-by-step accuracy and episode success rate metrics
- [ ] Implements F1, recall, and precision calculations for action matching
- [ ] Exports detailed CSV with all fields needed for visualization
- [ ] Generates JSON reports with aggregated metrics and configuration
- [ ] Creates visualization charts showing performance across datasets and metrics
- [ ] Supports configurable ocr_module parameter for comparison studies
- [ ] Saves and loads configuration files for reproducible runs

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
  why: Plotting functions for visualization charts
  section: pyplot.subplots, pyplot.bar, pyplot.plot for multi-metric charts
  
- url: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
  why: CSV export functionality with proper formatting
  critical: Use index=False to avoid unnamed index column
  
- url: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
  why: DataFrame construction and manipulation for metrics data
  
- file: src/mmllm/agent_ocr/aitw.py
  why: Complete example of running agent on AiTW data with action matching
  critical: Shows dataset loading, episode iteration, action matching pattern
  
- file: src/mmllm/android_in_the_wild/action_matching.py
  why: Core action matching logic for evaluation metrics
  critical: check_actions_match function is the foundation for all metrics
  
- file: src/mmllm/utils/episode_loader.py
  why: Episode loading and state management patterns
  critical: load_episode_with_history method and episode structure
  
- file: src/mmllm/evaluation/episode_runner.py
  why: Existing evaluation framework patterns for batch processing
  critical: Shows error handling and progress tracking patterns
  
- file: src/mmllm/evaluation/action_comparator.py
  why: Existing action comparison logic to extend
  critical: Base patterns for metric calculations
  
- file: src/mmllm/evaluation/evaluation_reporter.py
  why: Existing reporting infrastructure to enhance
  critical: JSON report generation and aggregation patterns
  
- file: src/mmllm/utils/visualization.py
  why: Existing visualization utilities and patterns
  critical: plot_episode function and matplotlib configuration
  
- file: tests/test_episode_evaluation.py
  why: Test patterns for evaluation, error handling, metric validation
  critical: Shows proper test structure and edge case handling
  
- file: src/mmllm/main.py
  why: CLI integration patterns and entry points
  critical: Shows how to add new commands and configuration handling
```

### Current Codebase Tree
```bash
src/mmllm/
├── evaluation/
│   ├── episode_runner.py          # Batch evaluation runner
│   ├── action_comparator.py       # Action comparison logic
│   └── evaluation_reporter.py     # JSON/Markdown reporting
├── utils/
│   ├── episode_loader.py          # Episode loading utilities
│   └── visualization.py           # Plotting utilities
├── android_in_the_wild/
│   ├── action_matching.py         # Core action matching
│   └── visualization_utils.py     # AiTW-specific plotting
├── agent_ocr/
│   ├── aitw.py                   # Complete example implementation
│   └── simple_ocr_agent.py      # Agent with ocr_module parameter
└── main.py                       # CLI entry point
```

### Desired Codebase Tree with Files to be Added
```bash
src/mmllm/
├── evaluation/
│   ├── benchmarking_pipeline.py   # Main benchmarking orchestrator
│   ├── metrics_calculator.py      # F1, recall, precision calculations
│   ├── csv_exporter.py           # CSV export with pandas
│   ├── config_manager.py         # Configuration save/load
│   └── benchmark_visualizer.py   # Visualization charts
├── utils/
│   └── dataset_utils.py          # Dataset selection utilities
└── cli/
    └── benchmark_command.py      # CLI command for benchmarking
```

### Known Gotchas of our Codebase & Library Quirks
```python
# CRITICAL: All imports must be absolute (mmllm.module.file)
# Example: from mmllm.android_in_the_wild.action_matching import check_actions_match

# CRITICAL: Use logging instead of print statements
import logging
logger = logging.getLogger(__name__)

# CRITICAL: AiTW dataset requires specific TensorFlow handling
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

# CRITICAL: Episode iteration pattern from aitw.py
# Must handle episode boundaries correctly using episode_id
for episode_idx, episode_tf in enumerate(get_episodes(raw_dataset, start_episode=0, end_episode=3)):

# CRITICAL: Action matching requires JAX arrays for UI annotations
annotation_positions = jax.numpy.array([ui_element['position'] for ui_element in ui_annotations])

# CRITICAL: CSV export must use index=False to avoid unnamed columns
df.to_csv(filename, index=False)

# CRITICAL: Configuration management should use JSON for reproducibility
# Save configuration alongside results for full reproducibility

# CRITICAL: Error handling must be comprehensive for dataset loading
# AiTW datasets can have corrupted episodes or missing fields
```

## Implementation Blueprint

### Data Models and Structure

Create the core data models to ensure type safety and consistency.
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    dataset_names: List[str]  # ['general', 'google_apps', etc.]
    ocr_module: bool
    start_episode: int
    end_episode: Optional[int]
    output_dir: str
    run_name: str
    
@dataclass
class StepResult:
    """Result for individual step"""
    episode_id: str
    step_number: int
    action_match: bool
    model_action: Dict[str, Any]
    ground_truth_action: Dict[str, Any]
    task_completed: bool
    
@dataclass
class EpisodeResult:
    """Result for complete episode"""
    episode_id: str
    dataset_name: str
    success_rate: float  # Percentage of steps completed before first error
    step_accuracy: float  # Percentage of correct steps
    total_steps: int
    steps_completed: int
    step_results: List[StepResult]
    
@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for analysis"""
    accuracy: float
    success_rate: float
    precision: float
    recall: float
    f1_score: float
    total_episodes: int
    total_steps: int
```

### List of Tasks to be Completed in Order

```yaml
Task 1 - Create Core Benchmarking Pipeline:
MODIFY src/mmllm/evaluation/:
  - CREATE benchmarking_pipeline.py
  - MIRROR pattern from: src/mmllm/agent_ocr/aitw.py
  - ENHANCE with configuration management and batch processing
  - PRESERVE existing evaluation patterns from episode_runner.py

Task 2 - Implement Metrics Calculator:
CREATE src/mmllm/evaluation/metrics_calculator.py:
  - EXTEND patterns from: src/mmllm/evaluation/action_comparator.py
  - IMPLEMENT F1, recall, precision calculations
  - MIRROR error handling from: tests/test_episode_evaluation.py
  - PRESERVE action matching logic from action_matching.py

Task 3 - Build CSV Export Module:
CREATE src/mmllm/evaluation/csv_exporter.py:
  - USE pandas DataFrame construction and to_csv method
  - INCLUDE all fields needed for visualization
  - MIRROR data structure patterns from existing reporters
  - ENSURE index=False to avoid unnamed columns

Task 4 - Configuration Management:
CREATE src/mmllm/evaluation/config_manager.py:
  - IMPLEMENT JSON-based configuration save/load
  - MIRROR pattern from existing configuration handling
  - ENSURE reproducibility with complete parameter capture
  - VALIDATE configuration parameters

Task 5 - Visualization Module:
CREATE src/mmllm/evaluation/benchmark_visualizer.py:
  - EXTEND patterns from: src/mmllm/utils/visualization.py
  - IMPLEMENT multi-dataset, multi-metric charts
  - USE matplotlib subplots for comprehensive views
  - MIRROR plotting patterns from visualization_utils.py

Task 6 - Dataset Utilities:
CREATE src/mmllm/utils/dataset_utils.py:
  - EXTRACT dataset selection logic from aitw.py
  - IMPLEMENT configurable dataset filtering
  - PRESERVE existing dataset directory mappings
  - ADD validation for dataset availability

Task 7 - CLI Integration:
CREATE src/mmllm/cli/benchmark_command.py:
  - MIRROR CLI patterns from: src/mmllm/main.py
  - IMPLEMENT argument parsing for all configuration options
  - INTEGRATE with existing CLI structure
  - PRESERVE logging and error handling patterns

Task 8 - Integration and Testing:
MODIFY src/mmllm/main.py:
  - ADD benchmark command registration
  - PRESERVE existing CLI structure
  - INTEGRATE with logging setup

CREATE tests/test_benchmarking_pipeline.py:
  - MIRROR test patterns from: tests/test_episode_evaluation.py
  - IMPLEMENT comprehensive test coverage
  - VALIDATE all metrics calculations
  - TEST configuration save/load functionality
```

### Per Task Pseudocode

```python
# Task 1: BenchmarkingPipeline
class BenchmarkingPipeline:
    def __init__(self, config: BenchmarkConfig):
        # PATTERN: Initialize agent with ocr_module from config
        self.agent = SimpleOCRAgent(ocr_module=config.ocr_module)
        self.config = config
        # PATTERN: Use existing logger setup
        self.logger = logging.getLogger(__name__)
    
    async def run_benchmark(self) -> List[EpisodeResult]:
        # PATTERN: Follow aitw.py dataset loading
        episode_results = []
        for dataset_name in self.config.dataset_names:
            # CRITICAL: Use TensorFlow dataset loading from aitw.py
            filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
            raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
            
            # PATTERN: Use get_episodes iterator from aitw.py
            for episode_idx, episode_tf in enumerate(get_episodes(...)):
                episode_result = await self._process_episode(episode_tf, dataset_name)
                episode_results.append(episode_result)
        
        return episode_results

# Task 2: MetricsCalculator
class MetricsCalculator:
    def calculate_metrics(self, episode_results: List[EpisodeResult]) -> BenchmarkMetrics:
        # PATTERN: Aggregate results similar to evaluation_reporter.py
        all_matches = [step.action_match for episode in episode_results 
                      for step in episode.step_results]
        
        # CRITICAL: Handle edge cases (empty results, all false, etc.)
        if not all_matches:
            return BenchmarkMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate standard metrics
        accuracy = sum(all_matches) / len(all_matches)
        # IMPLEMENT: F1, recall, precision calculations
        
        return BenchmarkMetrics(...)

# Task 3: CSVExporter
class CSVExporter:
    def export_to_csv(self, episode_results: List[EpisodeResult], filename: str):
        # PATTERN: Create DataFrame with all visualization fields
        rows = []
        for episode in episode_results:
            for step in episode.step_results:
                rows.append({
                    'episode_id': step.episode_id,
                    'dataset_name': episode.dataset_name,
                    'step_number': step.step_number,
                    'action_match': step.action_match,
                    'task_completed': step.task_completed,
                    'ocr_module': self.config.ocr_module,
                    # ... all fields needed for visualization
                })
        
        df = pd.DataFrame(rows)
        # CRITICAL: Use index=False
        df.to_csv(filename, index=False)

# Task 5: BenchmarkVisualizer
class BenchmarkVisualizer:
    def create_performance_charts(self, episode_results: List[EpisodeResult]):
        # PATTERN: Use matplotlib subplots for multiple charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Chart 1: Accuracy by dataset
        self._plot_accuracy_by_dataset(axes[0, 0], episode_results)
        # Chart 2: Success rate comparison
        self._plot_success_rates(axes[0, 1], episode_results)
        # Chart 3: F1 scores
        self._plot_f1_scores(axes[1, 0], episode_results)
        # Chart 4: OCR module comparison (if applicable)
        self._plot_ocr_comparison(axes[1, 1], episode_results)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/performance_charts.png")
```

### Integration Points
```yaml
DATABASE:
  - none: All data stored in files (CSV, JSON)
  
CONFIG:
  - add to: src/mmllm/evaluation/config_manager.py
  - pattern: JSON-based configuration with validation
  - save location: {output_dir}/config.json
  
CLI:
  - add to: src/mmllm/main.py
  - pattern: "benchmark" subcommand with argparse
  - example: "python -m mmllm benchmark --datasets google_apps general --ocr-module --episodes 0:10"
  
LOGGING:
  - use existing: logging.getLogger(__name__)
  - pattern: Info for progress, debug for detailed steps
  - output: Both console and log files
  
EXPORTS:
  - CSV: {output_dir}/benchmark_results_{timestamp}.csv
  - JSON: {output_dir}/benchmark_report_{timestamp}.json
  - Charts: {output_dir}/performance_charts_{timestamp}.png
  - Config: {output_dir}/config_{timestamp}.json
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
uv run ruff check src/mmllm/evaluation/ --fix
uv run mypy src/mmllm/evaluation/
uv run ruff check src/mmllm/cli/ --fix
uv run mypy src/mmllm/cli/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE tests/test_benchmarking_pipeline.py with comprehensive test cases:
def test_metrics_calculation():
    """Test F1, recall, precision calculations"""
    # PATTERN: Use mock episode results with known outcomes
    episode_results = create_mock_episode_results()
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(episode_results)
    
    assert metrics.accuracy == 0.75  # Known expected value
    assert metrics.f1_score > 0  # Ensure calculation works
    assert 0 <= metrics.recall <= 1  # Valid range

def test_csv_export():
    """Test CSV export format and content"""
    episode_results = create_mock_episode_results()
    exporter = CSVExporter()
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        exporter.export_to_csv(episode_results, f.name)
        
        # Verify CSV format
        df = pd.read_csv(f.name)
        assert 'episode_id' in df.columns
        assert 'action_match' in df.columns
        assert len(df) > 0

def test_configuration_roundtrip():
    """Test configuration save and load"""
    config = BenchmarkConfig(
        dataset_names=['google_apps'],
        ocr_module=True,
        start_episode=0,
        end_episode=5,
        output_dir='/tmp/test',
        run_name='test_run'
    )
    
    manager = ConfigManager()
    manager.save_config(config, '/tmp/test_config.json')
    loaded_config = manager.load_config('/tmp/test_config.json')
    
    assert config == loaded_config

def test_error_handling():
    """Test graceful error handling for corrupted data"""
    with pytest.raises(ValidationError):
        # Test invalid configuration
        BenchmarkConfig(dataset_names=[], ocr_module=True, ...)
    
    # Test handling of corrupted episodes
    pipeline = BenchmarkingPipeline(valid_config)
    result = pipeline._process_corrupted_episode(corrupted_data)
    assert result is not None  # Should handle gracefully
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_benchmarking_pipeline.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test the complete pipeline with sample data
uv run python -m mmllm benchmark \
  --datasets google_apps \
  --ocr-module \
  --episodes 0:2 \
  --output-dir /tmp/benchmark_test

# Expected outputs:
# - /tmp/benchmark_test/benchmark_results_*.csv
# - /tmp/benchmark_test/benchmark_report_*.json  
# - /tmp/benchmark_test/performance_charts_*.png
# - /tmp/benchmark_test/config_*.json

# Verify CSV format
head -5 /tmp/benchmark_test/benchmark_results_*.csv

# Verify JSON structure
jq '.metrics | keys' /tmp/benchmark_test/benchmark_report_*.json

# If error: Check logs for detailed error information
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] CLI command works: `uv run python -m mmllm benchmark --help`
- [ ] CSV export contains all required fields for visualization
- [ ] JSON report includes comprehensive metrics (accuracy, F1, recall, precision)
- [ ] Configuration save/load preserves all parameters
- [ ] Visualization charts display correctly with multiple datasets
- [ ] Error handling gracefully manages corrupted episodes
- [ ] OCR module comparison works correctly
- [ ] Reproducibility: Same config produces same results

---

## Anti-Patterns to Avoid
- ❌ Don't use print statements - use configured logger
- ❌ Don't use relative imports - use absolute mmllm.module.file format
- ❌ Don't ignore TensorFlow configuration for AiTW datasets
- ❌ Don't save CSV with default index - use index=False
- ❌ Don't hardcode dataset paths - use existing dataset_directories mapping
- ❌ Don't skip error handling for corrupted episodes
- ❌ Don't create new visualization patterns - extend existing ones
- ❌ Don't ignore JAX array requirements for action matching
- ❌ Don't skip configuration validation - validate all parameters
- ❌ Don't create inconsistent metric calculations - follow mathematical definitions

## Implementation Confidence Score: 9/10

**Justification**: 
- All necessary codebase patterns identified and documented
- Existing evaluation framework provides solid foundation
- Clear implementation path with specific file references
- Comprehensive test strategy with executable validation gates
- External documentation (pandas, matplotlib) readily available
- Known gotchas and anti-patterns clearly documented
- Modular design allows for incremental implementation and testing

**Risk Mitigation**: The main risk is handling edge cases in the AiTW dataset (corrupted episodes, missing fields), but this is mitigated by comprehensive error handling patterns from existing evaluation code and thorough test coverage.
