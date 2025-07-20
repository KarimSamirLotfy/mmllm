"""Visualization module for benchmarking results with comprehensive charts."""

import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from mmllm.evaluation.types import EpisodeResult, BenchmarkConfig, BenchmarkMetrics

logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Create comprehensive visualization charts for benchmarking results."""
    
    def __init__(self):
        """Initialize benchmark visualizer."""
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style for better-looking plots
        plt.style.use('default')
        
        # Define color scheme for consistency
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463', 
            'accent': '#F39C12',
            'warning': '#E74C3C',
            'neutral': '#85929E',
            'datasets': ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'],
            'metrics': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']
        }
    
    def create_performance_charts(self, episode_results: List[EpisodeResult], 
                                dataset_metrics: Dict[str, BenchmarkMetrics], 
                                filename: str, config: BenchmarkConfig) -> None:
        """
        Create comprehensive performance charts showing multiple metrics and comparisons.
        
        Args:
            episode_results: List of episode results
            dataset_metrics: Metrics by dataset
            filename: Output filename for the chart
            config: Benchmark configuration
        """
        if not episode_results:
            self.logger.warning("No episode results for visualization")
            return
        
        try:
            self.logger.info(f"Creating performance charts: {filename}")
            
            # Create figure with subplots using matplotlib pattern
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Benchmark Performance Report: {config.run_name}', 
                        fontsize=16, fontweight='bold')
            
            # Chart 1: Overall metrics comparison by dataset
            self._plot_metrics_by_dataset(axes[0, 0], dataset_metrics, config)
            
            # Chart 2: Success rate distribution
            self._plot_success_rate_distribution(axes[0, 1], episode_results)
            
            # Chart 3: Action type performance
            self._plot_action_type_performance(axes[1, 0], episode_results)
            
            # Chart 4: Step-level accuracy trends
            self._plot_step_accuracy_trends(axes[1, 1], episode_results)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI for quality
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Performance charts saved successfully: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error creating performance charts: {e}")
            raise
    
    def _plot_metrics_by_dataset(self, ax, dataset_metrics: Dict[str, BenchmarkMetrics], config: BenchmarkConfig) -> None:
        """Plot overall metrics comparison by dataset."""
        if not dataset_metrics:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics by Dataset')
            return
        
        datasets = list(dataset_metrics.keys())
        metrics_names = ['Accuracy', 'Success Rate', 'Precision', 'Recall', 'F1 Score']
        
        # Extract metric values
        accuracy_vals = [dataset_metrics[d].accuracy for d in datasets]
        success_vals = [dataset_metrics[d].success_rate for d in datasets]
        precision_vals = [dataset_metrics[d].precision for d in datasets]
        recall_vals = [dataset_metrics[d].recall for d in datasets]
        f1_vals = [dataset_metrics[d].f1_score for d in datasets]
        
        # Create grouped bar chart
        x = np.arange(len(datasets))
        width = 0.15
        
        ax.bar(x - 2*width, accuracy_vals, width, label='Accuracy', color=self.colors['metrics'][0], alpha=0.8)
        ax.bar(x - width, success_vals, width, label='Success Rate', color=self.colors['metrics'][1], alpha=0.8)
        ax.bar(x, precision_vals, width, label='Precision', color=self.colors['metrics'][2], alpha=0.8)
        ax.bar(x + width, recall_vals, width, label='Recall', color=self.colors['metrics'][3], alpha=0.8)
        ax.bar(x + 2*width, f1_vals, width, label='F1 Score', color=self.colors['accent'], alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title(f'Performance Metrics by Dataset (OCR: {config.ocr_module})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, dataset in enumerate(datasets):
            metrics_vals = [accuracy_vals[i], success_vals[i], precision_vals[i], recall_vals[i], f1_vals[i]]
            positions = [i - 2*width, i - width, i, i + width, i + 2*width]
            for pos, val in zip(positions, metrics_vals):
                ax.text(pos, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_success_rate_distribution(self, ax, episode_results: List[EpisodeResult]) -> None:
        """Plot distribution of episode success rates."""
        success_rates = [ep.success_rate for ep in episode_results]
        
        if not success_rates:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Rate Distribution')
            return
        
        # Create histogram
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        n, bins, patches = ax.hist(success_rates, bins=bins, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        
        # Color code the bars
        for i, p in enumerate(patches):
            if bins[i] >= 0.8:
                p.set_facecolor(self.colors['secondary'])  # Green for high success
            elif bins[i] >= 0.5:
                p.set_facecolor(self.colors['accent'])     # Orange for medium success
            else:
                p.set_facecolor(self.colors['warning'])    # Red for low success
        
        ax.set_xlabel('Success Rate')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Episode Success Rate Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_success = np.mean(success_rates)
        median_success = np.median(success_rates)
        ax.axvline(mean_success, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_success:.2f}')
        ax.axvline(median_success, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_success:.2f}')
        ax.legend()
    
    def _plot_action_type_performance(self, ax, episode_results: List[EpisodeResult]) -> None:
        """Plot performance by action type."""
        # Categorize actions and calculate performance
        action_performance = {}
        
        for episode in episode_results:
            for step in episode.step_results:
                action_type = self._categorize_action_type(step.model_action.get('action_type', 'unknown'))
                
                if action_type not in action_performance:
                    action_performance[action_type] = {'scores': [], 'matches': 0, 'total': 0}
                
                action_performance[action_type]['scores'].append(step.evaluation_score)
                action_performance[action_type]['total'] += 1
                if step.action_match:
                    action_performance[action_type]['matches'] += 1
        
        if not action_performance:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Action Type Performance')
            return
        
        # Calculate metrics per action type
        action_types = list(action_performance.keys())
        avg_scores = [np.mean(action_performance[at]['scores']) for at in action_types]
        match_rates = [action_performance[at]['matches'] / action_performance[at]['total'] for at in action_types]
        counts = [action_performance[at]['total'] for at in action_types]
        
        # Create dual-axis plot
        x = np.arange(len(action_types))
        
        # Bar chart for average scores
        bars1 = ax.bar(x - 0.2, avg_scores, 0.4, label='Avg Score', color=self.colors['primary'], alpha=0.7)
        
        # Bar chart for match rates
        bars2 = ax.bar(x + 0.2, match_rates, 0.4, label='Match Rate', color=self.colors['secondary'], alpha=0.7)
        
        ax.set_xlabel('Action Type')
        ax.set_ylabel('Score/Rate')
        ax.set_title('Performance by Action Type')
        ax.set_xticks(x)
        ax.set_xticklabels(action_types, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, counts)):
            ax.text(i, max(bar1.get_height(), bar2.get_height()) + 0.05, f'n={count}', 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_step_accuracy_trends(self, ax, episode_results: List[EpisodeResult]) -> None:
        """Plot accuracy trends by step number."""
        # Aggregate accuracy by step number
        step_accuracy = {}
        
        for episode in episode_results:
            for step in episode.step_results:
                step_num = step.step_number
                
                if step_num not in step_accuracy:
                    step_accuracy[step_num] = {'correct': 0, 'total': 0}
                
                step_accuracy[step_num]['total'] += 1
                if step.action_match:
                    step_accuracy[step_num]['correct'] += 1
        
        if not step_accuracy:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Step Accuracy Trends')
            return
        
        # Calculate accuracy by step
        steps = sorted(step_accuracy.keys())
        accuracies = [step_accuracy[step]['correct'] / step_accuracy[step]['total'] for step in steps]
        counts = [step_accuracy[step]['total'] for step in steps]
        
        # Plot line chart with confidence intervals
        ax.plot(steps, accuracies, 'o-', color=self.colors['primary'], linewidth=2, markersize=6)
        
        # Add trend line
        if len(steps) > 1:
            z = np.polyfit(steps, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), '--', color=self.colors['warning'], alpha=0.8, 
                   label=f'Trend (slope: {z[0]:.3f})')
            ax.legend()
        
        ax.set_xlabel('Step Number')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Trend by Step Number')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add data point count labels
        for step, acc, count in zip(steps, accuracies, counts):
            ax.annotate(f'n={count}', (step, acc), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
    
    def _categorize_action_type(self, action_type) -> str:
        """Categorize action type for visualization."""
        if not action_type:
            return 'unknown'
        
        action_str = str(action_type).upper()
        
        if 'DUAL_POINT' in action_str or action_str == '1':
            return 'dual_point'
        elif 'TYPE' in action_str or action_str == '5':
            return 'type'
        elif any(nav in action_str for nav in ['BACK', 'HOME', 'ENTER']) or action_str in ['6', '7', '8']:
            return 'navigation'
        elif any(stat in action_str for stat in ['COMPLETE', 'IMPOSSIBLE']) or action_str in ['9', '10']:
            return 'status'
        else:
            return 'other'
    
    def create_comparison_chart(self, results_a: List[EpisodeResult], results_b: List[EpisodeResult],
                              label_a: str, label_b: str, filename: str) -> None:
        """
        Create comparison chart between two different configurations.
        
        Args:
            results_a: Results from first configuration
            results_b: Results from second configuration  
            label_a: Label for first configuration
            label_b: Label for second configuration
            filename: Output filename
        """
        try:
            self.logger.info(f"Creating comparison chart: {filename}")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Configuration Comparison: {label_a} vs {label_b}', fontsize=14, fontweight='bold')
            
            # Chart 1: Overall metrics comparison
            self._plot_configuration_comparison(axes[0], results_a, results_b, label_a, label_b)
            
            # Chart 2: Success rate distributions
            self._plot_success_rate_comparison(axes[1], results_a, results_b, label_a, label_b)
            
            plt.tight_layout()
            
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Comparison chart saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {e}")
            raise
    
    def _plot_configuration_comparison(self, ax, results_a: List[EpisodeResult], results_b: List[EpisodeResult],
                                     label_a: str, label_b: str) -> None:
        """Plot overall metrics comparison between configurations."""
        # Calculate metrics for both configurations
        def calc_metrics(results):
            if not results:
                return [0, 0, 0, 0]
            
            total_steps = sum(len(ep.step_results) for ep in results)
            correct_actions = sum(sum(1 for step in ep.step_results if step.action_match) for ep in results)
            avg_success_rate = np.mean([ep.success_rate for ep in results])
            avg_step_accuracy = correct_actions / total_steps if total_steps > 0 else 0
            avg_evaluation_score = np.mean([step.evaluation_score for ep in results for step in ep.step_results])
            
            return [avg_success_rate, avg_step_accuracy, avg_evaluation_score, len(results)]
        
        metrics_a = calc_metrics(results_a)
        metrics_b = calc_metrics(results_b)
        
        metric_names = ['Success Rate', 'Step Accuracy', 'Avg Score', 'Episodes']
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars_a = ax.bar(x - width/2, metrics_a, width, label=label_a, color=self.colors['primary'], alpha=0.8)
        bars_b = ax.bar(x + width/2, metrics_b, width, label=label_b, color=self.colors['secondary'], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars, metrics in [(bars_a, metrics_a), (bars_b, metrics_b)]:
            for bar, metric in zip(bars, metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{metric:.2f}' if metric < 10 else f'{int(metric)}',
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_success_rate_comparison(self, ax, results_a: List[EpisodeResult], results_b: List[EpisodeResult],
                                    label_a: str, label_b: str) -> None:
        """Plot success rate distribution comparison."""
        success_rates_a = [ep.success_rate for ep in results_a] if results_a else []
        success_rates_b = [ep.success_rate for ep in results_b] if results_b else []
        
        bins = np.linspace(0, 1, 21)
        
        ax.hist(success_rates_a, bins=bins, alpha=0.6, label=label_a, 
               color=self.colors['primary'], edgecolor='black')
        ax.hist(success_rates_b, bins=bins, alpha=0.6, label=label_b, 
               color=self.colors['secondary'], edgecolor='black')
        
        ax.set_xlabel('Success Rate')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Success Rate Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        if success_rates_a:
            mean_a = np.mean(success_rates_a)
            ax.axvline(mean_a, color=self.colors['primary'], linestyle='--', alpha=0.8, 
                      label=f'{label_a} mean: {mean_a:.2f}')
        
        if success_rates_b:
            mean_b = np.mean(success_rates_b)
            ax.axvline(mean_b, color=self.colors['secondary'], linestyle='--', alpha=0.8,
                      label=f'{label_b} mean: {mean_b:.2f}')
