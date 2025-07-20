name: "Analysis Visualization Notebooks for Benchmark Results"
description: |
  Create comprehensive multiple Jupyter notebooks in vis/ folder that analyze and visualize benchmark results from multiple runs, providing data-driven insights through matplotlib and pandas visualization stories.

## Goal
Create visualization notebooks in `vis/` folder that take the `results/` data from multiple runs and create compelling data stories using matplotlib and pandas. The notebooks should:
- Analyze performance across different experimental conditions (stateless vs stateful, tree vs no-tree)
- Show individual and dataset-segregated performance comparisons
- Identify common failure patterns and dataset-specific challenges
- Generate publication-ready visualizations for research insights

## Why
- **Research Insights**: Enable data-driven understanding of agent performance across different configurations
- **Publication Support**: Generate professional visualizations for research papers and presentations
- **Decision Making**: Provide clear visual evidence of the preformance.

## What
Interactive Jupyter notebooks that, each notebook should test 1 aspect:
1. **Compare Experimental Runs**: stateless-vs-stateful, tree-vs-no-tree performance analysis
2. **Dataset Breakdown**: Performance segregated by dataset (general, google_apps, install, single)
3. **Cross-Run Analysis**: Common failure types, mismatch patterns across all runs
4. **Report Integration**: Utilize JSON reports for context and additional insights

### Success Criteria
- [ ] `vis/` folder created with organized notebook structure
- [ ] Individual run comparison notebooks with performance breakdowns
- [ ] Cross-run analysis notebook showing aggregate patterns
- [ ] All visualizations use consistent styling and are publication-ready
- [ ] Notebooks can be run end-to-end without errors
- [ ] Clear narrative flow with markdown explanations between code cells

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://matplotlib.org/stable/gallery/index.html
  why: Best practices for publication-ready plots and styling
  
- url: https://pandas.pydata.org/docs/user_guide/visualization.html
  why: Pandas plotting integration patterns
  
- file: src/mmllm/evaluation/csv_exporter.py
  why: Understanding CSV structure and column meanings for proper analysis
  
- file: src/mmllm/utils/visualization.py  
  why: to show how to plot episodes. 
  
- doc: https://seaborn.pydata.org/tutorial/introduction.html
  section: Statistical visualization patterns
  critical: Better statistical plots than pure matplotlib for analysis

- file: src/mmllm/android_in_the_wild/action_matching.py
  why: Has the used action_matching code. that should be alos used on the csv, to show any insights. use this as the ground truth metric.
```

### Current Codebase Structure (relevant files)
```
mmllm/
├── results/
│   ├── stateless-vs-statefull/
│   │   ├── statefull/
│   │   │   ├── benchmark_report_20250720_125011.json
│   │   │   └── benchmark_results_20250720_125011.csv
│   │   └── stateless/
│   │       ├── benchmark_report_20250720_121356.json
│   │       └── benchmark_results_20250720_121356.csv
│   └── tree-vs-no-tree/
│       ├── no-tree/
│       │   ├── benchmark_report_20250720_130908.json
│       │   └── benchmark_results_20250720_130908.csv
│       └── tree/
│           └── ...
├── src/mmllm/
│   ├── evaluation/
│   │   ├── csv_exporter.py          # CSV structure definition
│   │   └── evaluation_reporter.py   # JSON report structure
│   └── utils/
│       └── visualization.py         # Existing plot functions
└── pyproject.toml                   # Dependencies (matplotlib, pandas already included)
```

### Desired Codebase Structure
```
mmllm/
├── vis/                             # NEW: Visualization notebooks
│   ├── README.md                    # Guide to notebook usage
│   ├── tree_vs_no_tree_comparison.ipynb      # Compare specific experimental runs
│   ├── statefull_vs_stateless.ipynb    # Dataset-specific performance 
.....
│   ├── 03_failure_analysis.ipynb    # Cross-run failure pattern analysis
```

### Known CSV Structure (from csv_exporter.py analysis)
```python
# CRITICAL: CSV columns for analysis
CSV_COLUMNS = [
    # Metadata
    'run_name', 'timestamp', 'dataset_name', 'ocr_module', 'episode_id', 'goal',
    
    # Episode-level metrics
    'episode_success_rate', 'episode_step_accuracy', 'episode_total_steps', 
    'episode_steps_completed', 'episode_overall_success_rate',
    
    # Step-level details
    'step_number', 'step_action_match', 'step_action_type_match', 
    'step_evaluation_score', 'step_task_completed', 'step_coordinate_distance',
    
    # Model vs Ground Truth comparison
    'model_action_type', 'model_coordinates_y', 'model_coordinates_x',
    'gt_action_type', 'gt_coordinates_y', 'gt_coordinates_x', 
    
    # Computed analysis columns
    'coordinate_error', 'action_type_category', 'performance_category'
]

# CRITICAL: Action type categories
ACTION_CATEGORIES = ['dual_point', 'type', 'navigation', 'status', 'other']

# CRITICAL: Performance categories  
PERFORMANCE_CATEGORIES = ['excellent', 'good', 'fair', 'poor', 'unknown']
```

### JSON Report Structure (from evaluation_reporter.py)
```python
# CRITICAL: JSON report structure for additional context
REPORT_STRUCTURE = {
    "report_metadata": {
        "timestamp", "run_name", "total_episodes", "datasets", "ocr_module"
    },
    "overall_metrics": {
        "accuracy", "success_rate", "precision", "recall", "f1_score"
    },
    "dataset_metrics": {
        # Per dataset breakdown
        "general": {"accuracy", "success_rate", "total_episodes", "total_steps"},
        "google_apps": {...},
        "install": {...},
        "single": {...}
    }
}
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Matplotlib figure management for notebooks
# Always use plt.figure() or fig, ax = plt.subplots() to avoid figure overlap
# Use plt.close('all') between major plotting sections

# CRITICAL: Pandas CSV reading
# CSV files use index=False, so no unnamed index column to worry about
# Float precision is 6 decimal places (%.6f format)
# Missing coordinates are stored as None/NaN

# CRITICAL: Data aggregation patterns
# Episode-level data needs groupby('episode_id').first() for unique episodes
# Step-level data can be analyzed directly for step-specific insights
# Always check for NaN values before statistical calculations

# CRITICAL: Color consistency with existing codebase
# Ground truth actions: red/blue (from visualization_utils.py)
# Model predictions: green (from utils/visualization.py)
# Use existing _ACTION_COLOR = 'blue' pattern for consistency

# CRITICAL: Notebook best practices
# Use %%matplotlib inline for proper plot display
# Clear outputs between runs with plt.close('all')
# Save figures to avoid re-computation: plt.savefig('output.png', dpi=150, bbox_inches='tight')
```

## Implementation Blueprint

### Data Infrastructure
Create the foundational data loading and analysis utilities that all notebooks will use.

```python
# vis/utils/data_loader.py - PATTERN: Mirror src/utils patterns
class BenchmarkDataLoader:
    """Load and preprocess benchmark CSV and JSON data for analysis."""
    
    def load_run_data(self, run_path: str) -> Dict[str, Any]:
        """Load both CSV and JSON data for a benchmark run."""
        # PATTERN: Use pathlib for robust path handling
        # CRITICAL: Handle missing files gracefully
        # RETURN: {'csv': DataFrame, 'json': dict, 'metadata': dict}
    
    def load_all_runs(self, results_dir: str = 'results') -> Dict[str, Dict]:
        """Load data from all available runs."""
        # PATTERN: Recursive directory scanning
        # CRITICAL: Auto-detect run names from folder structure
    
    def preprocess_episode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare episode-level data for analysis."""
        # PATTERN: Standard preprocessing pipeline
        # CRITICAL: Handle NaN values, normalize data types
```

### Task List (in order of completion)

```yaml
Task 1: Create Data Infrastructure
CREATE vis/utils/data_loader.py:
  - IMPLEMENT BenchmarkDataLoader class following src/evaluation patterns
  - ADD robust CSV/JSON loading with error handling
  - INCLUDE data validation and preprocessing methods
  
CREATE vis/utils/plot_helpers.py:
  - MIRROR color schemes from src/mmllm/utils/visualization.py
  - IMPLEMENT consistent plot styling functions
  - ADD publication-ready matplotlib rcParams setup

CREATE vis/utils/analysis_utils.py:
  - ADD statistical analysis helper functions
  - IMPLEMENT performance comparison utilities
  - INCLUDE failure pattern detection methods

Task 2: Run Comparison Notebook
CREATE vis/01_run_comparison.ipynb:
  - LOAD stateless vs stateful data using data_loader
  - IMPLEMENT side-by-side performance comparisons
  - GENERATE dataset-segregated performance plots
  - ADD statistical significance testing

Task 3: Dataset Analysis Notebook  
CREATE vis/02_dataset_analysis.ipynb:
  - ANALYZE performance patterns per dataset (general, google_apps, install, single)
  - VISUALIZE action type distributions per dataset
  - IDENTIFY dataset-specific challenges and success patterns
  - COMPARE coordinate accuracy across datasets

Task 4: Failure Analysis Notebook
CREATE vis/03_failure_analysis.ipynb:
  - AGGREGATE failure patterns across all runs
  - VISUALIZE most common action type mismatches
  - ANALYZE coordinate error distributions
  - GENERATE heatmaps of failure patterns by dataset

Task 5: Documentation and Polish
CREATE vis/README.md:
  - DOCUMENT notebook usage and dependencies
  - EXPLAIN analysis methodology and assumptions
  - PROVIDE clear usage instructions
  
POLISH notebooks:
  - ADD narrative markdown explanations
  - ENSURE consistent styling across all visualizations
  - VALIDATE all notebooks run end-to-end without errors
```

### Per Task Pseudocode

```python
# Task 1: Data Infrastructure
class BenchmarkDataLoader:
    def load_run_data(self, run_path: str) -> Dict[str, Any]:
        # PATTERN: Use pathlib for cross-platform compatibility
        run_path = Path(run_path)
        
        # CRITICAL: Find CSV and JSON files in run directory
        csv_files = list(run_path.glob("*.csv"))
        json_files = list(run_path.glob("*.json"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {run_path}")
            
        # PATTERN: Load CSV with proper dtypes
        df = pd.read_csv(csv_files[0], 
                        dtype={'episode_id': str, 'run_name': str})
        
        # PATTERN: Load JSON for metadata
        json_data = {}
        if json_files:
            with open(json_files[0]) as f:
                json_data = json.load(f)
        
        return {
            'csv': df,
            'json': json_data, 
            'metadata': {'path': str(run_path), 'name': run_path.name}
        }

# Task 2: Run Comparison Notebook Structure
"""
# Cell 1: Setup and Data Loading
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from vis.utils.data_loader import BenchmarkDataLoader
from vis.utils.plot_helpers import setup_publication_style

# Cell 2: Load Comparison Data  
loader = BenchmarkDataLoader()
stateful_data = loader.load_run_data('results/stateless-vs-statefull/statefull')
stateless_data = loader.load_run_data('results/stateless-vs-statefull/stateless')

# Cell 3: Episode-Level Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# PATTERN: Compare success rates, step accuracy, steps completed
# CRITICAL: Use box plots for distribution comparison

# Cell 4: Dataset-Segregated Analysis
datasets = ['general', 'google_apps', 'install', 'single']
for dataset in datasets:
    # PATTERN: Filter by dataset, compare metrics
    # VISUALIZATION: Side-by-side bar charts
"""

# Task 3: Dataset Analysis Patterns
"""
# Statistical analysis per dataset
for dataset in datasets:
    dataset_data = combined_df[combined_df['dataset_name'] == dataset]
    
    # CRITICAL: Action type distribution analysis
    action_dist = dataset_data['action_type_category'].value_counts()
    
    # CRITICAL: Performance vs action type correlation
    perf_by_action = dataset_data.groupby('action_type_category')['step_evaluation_score'].mean()
    
    # VISUALIZATION: Stacked bar charts, correlation heatmaps
"""
```

### Integration Points
```yaml
ENVIRONMENT:
  - notebooks: "Use existing uv environment with matplotlib, pandas installed"
  - dependencies: "All required packages already in pyproject.toml"
  
DATA_SOURCES:
  - csv_files: "results/*/benchmark_results_*.csv"
  - json_files: "results/*/benchmark_report_*.json"
  - pattern: "Auto-detect available runs from results/ folder structure"
  
OUTPUT:
  - figures: "Save publication-ready figures in vis/outputs/"
  - format: "PNG for quick viewing, PDF for publication quality"
```


## Final Validation Checklist
- [ ] All notebooks execute without errors: `uv run jupyter nbconvert --execute vis/*.ipynb`
- [ ] Data loading handles missing files gracefully
- [ ] Visualizations use consistent styling and colors
- [ ] Statistical analyses are mathematically sound
- [ ] Notebooks have clear narrative flow with markdown explanations
- [ ] Publication-ready figures saved to vis/outputs/
- [ ] README provides clear usage instructions
- [ ] Cross-run analysis provides actionable insights

## Anti-Patterns to Avoid
- ❌ Don't hardcode file paths - use relative paths and auto-detection
- ❌ Don't ignore NaN/missing data - handle explicitly with clear documentation  
- ❌ Don't create inconsistent plot styles - use shared helper functions
- ❌ Don't skip statistical validation - verify assumptions before analysis
- ❌ Don't create notebooks without narrative - explain what you're showing and why
- ❌ Don't duplicate code across notebooks - extract to shared utilities
- ❌ Don't generate plots without saving them - enable reproducible analysis

## Quality Score: 9/10

High confidence for one-pass implementation because:
✅ **Complete Context**: Full understanding of CSV structure, existing patterns, and data sources
✅ **Clear Requirements**: Specific analysis goals with concrete deliverables  
✅ **Existing Patterns**: Rich codebase with visualization utilities to build upon
✅ **Validation Strategy**: Progressive testing from data loading to full analysis
✅ **Tool Familiarity**: Standard data science stack (pandas, matplotlib, jupyter)

Minor risk factor: Ensuring statistical analyses are appropriately chosen for the data characteristics, but this is mitigated by starting with descriptive statistics and standard comparison methods.
