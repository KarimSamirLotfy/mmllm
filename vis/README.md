# Visualization Notebooks

This folder contains visualization notebooks for analyzing MMLLM benchmark results.

## Notebooks

### 1. `comprehensive_analysis.ipynb`
**Main analysis notebook** that provides a complete overview of all experimental results.

**Features:**
- Loads and analyzes all CSV and JSON data from the `results/` folder
- Compares stateless vs stateful configurations
- Analyzes performance across different datasets
- Identifies common failure patterns and mismatch types
- Creates comprehensive dashboard visualizations
- Integrates insights from JSON reports

**Sections:**
1. Setup Environment and Import Libraries
2. Load and Explore Data Structure  
3. Individual Run Analysis - Stateless vs Stateful
4. Dataset-Specific Performance Comparison
5. Cross-Run Analysis - Common Failure Types
6. Cross-Run Analysis - Dataset Mismatch Patterns
7. Report Integration and Insights
8. Summary Visualizations - Complete Story Dashboard

### 2. `tree_analysis.ipynb`
**Focused analysis** comparing tree vs no-tree configurations.

**Features:**
- Direct comparison between tree and no-tree approaches
- Efficiency analysis (steps vs completion rates)
- Dataset-specific effects of tree structures
- Performance trade-off analysis

## Usage

1. **Prerequisites**: Make sure you have the required packages installed:
   ```bash
   # The project uses uv environment
   uv pip install matplotlib pandas seaborn numpy
   ```

2. **Data Requirements**: 
   - Results should be in `../results/` folder relative to the notebooks
   - Each experiment should have both CSV and JSON files
   - Current structure supports:
     - `results/stateless-vs-statefull/stateless/`
     - `results/stateless-vs-statefull/statefull/`
     - `results/tree-vs-no-tree/tree/`
     - `results/tree-vs-no-tree/no-tree/`

3. **Running the Analysis**:
   - Open the notebooks in Jupyter or VS Code
   - Run all cells sequentially
   - The notebooks will automatically discover and load data
   - Visualizations will be generated inline

## Output

The notebooks generate:
- **Performance comparison charts** (bar charts, heatmaps)
- **Failure pattern analysis** (pie charts, scatter plots)
- **Dataset-specific breakdowns** (grouped bar charts)
- **Comprehensive dashboards** (multi-panel figures)
- **Statistical summaries** (printed reports)

## Data Format

The notebooks expect CSV files with columns including:
- `episode_success_rate`, `episode_step_accuracy`
- `dataset_name`, `model_action_type`
- `step_action_match`, `performance_category`
- `coordinate_error`, `run_name`

JSON reports should contain:
- `overall_metrics` (accuracy, success_rate, f1_score)
- `dataset_metrics` (per-dataset performance)
- `report_metadata` (run information)

## Customization

To adapt for new experimental configurations:
1. Update the path discovery logic in section 2
2. Modify the run type detection based on folder structure
3. Add new visualization types as needed
4. Extend the dashboard with additional metrics

## Tips

- Run the comprehensive analysis first for a complete overview
- Use the tree analysis for specific tree vs no-tree insights  
- The notebooks handle missing data gracefully
- All visualizations are publication-ready with proper labels and formatting
