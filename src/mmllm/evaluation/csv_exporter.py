"""CSV export module for benchmarking results with pandas."""

import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from mmllm.evaluation.types import EpisodeResult, BenchmarkConfig

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export benchmarking results to CSV format with comprehensive data."""
    
    def __init__(self):
        """Initialize CSV exporter."""
        self.logger = logging.getLogger(__name__)
    
    def export_to_csv(self, episode_results: List[EpisodeResult], filename: str, config: BenchmarkConfig) -> None:
        """
        Export episode results to CSV format with all fields needed for visualization.
        
        Args:
            episode_results: List of episode results to export
            filename: Output CSV filename
            config: Benchmark configuration for additional metadata
        """
        if not episode_results:
            self.logger.warning("No episode results to export")
            return
        
        try:
            self.logger.info(f"Exporting {len(episode_results)} episodes to CSV: {filename}")
            
            # Prepare data rows
            rows = []
            
            for episode in episode_results:
                # Episode-level data
                episode_base = {
                    'run_name': config.run_name,
                    'timestamp': datetime.now().isoformat(),
                    'episode_id': episode.episode_id,
                    'dataset_name': episode.dataset_name,
                    'goal': episode.goal,
                    'episode_success_rate': episode.success_rate,
                    'episode_step_accuracy': episode.step_accuracy,
                    'episode_total_steps': episode.total_steps,
                    'episode_steps_completed': episode.steps_completed,
                    'episode_overall_success_rate': episode.overall_success_rate,
                    'ocr_module': config.ocr_module,
                    'max_steps_per_episode': config.max_steps_per_episode,
                    'start_episode': config.start_episode,
                    'end_episode': config.end_episode
                }
                
                # Step-level data
                for step in episode.step_results:
                    row = episode_base.copy()
                    row.update({
                        'step_number': step.step_number,
                        'step_action_match': step.action_match,
                        'step_action_type_match': step.action_type_match,
                        'step_evaluation_score': step.evaluation_score,
                        'step_task_completed': step.task_completed,
                        'step_coordinate_distance': step.coordinate_distance,
                        'step_text_match': step.text_match,
                        
                        # Model action details
                        'model_action_type': step.model_action.get('action_type', 'unknown'),
                        'model_coordinates_y': step.model_action.get('coordinates', [None, None])[0] if step.model_action.get('coordinates') else None,
                        'model_coordinates_x': step.model_action.get('coordinates', [None, None])[1] if step.model_action.get('coordinates') else None,
                        'model_lift_coordinates_y': step.model_action.get('lift_coordinates', [None, None])[0] if step.model_action.get('lift_coordinates') else None,
                        'model_lift_coordinates_x': step.model_action.get('lift_coordinates', [None, None])[1] if step.model_action.get('lift_coordinates') else None,
                        'model_text': step.model_action.get('text', None),
                        
                        # Ground truth action details
                        'gt_action_type': step.ground_truth_action.get('action_type', 'unknown'),
                        'gt_coordinates_y': step.ground_truth_action.get('coordinates', [None, None])[0] if step.ground_truth_action.get('coordinates') else None,
                        'gt_coordinates_x': step.ground_truth_action.get('coordinates', [None, None])[1] if step.ground_truth_action.get('coordinates') else None,
                        'gt_lift_coordinates_y': step.ground_truth_action.get('lift_coordinates', [None, None])[0] if step.ground_truth_action.get('lift_coordinates') else None,
                        'gt_lift_coordinates_x': step.ground_truth_action.get('lift_coordinates', [None, None])[1] if step.ground_truth_action.get('lift_coordinates') else None,
                        'gt_text': step.ground_truth_action.get('text', None)
                    })
                    
                    rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Add computed columns for analysis
            df['coordinate_error'] = df.apply(self._calculate_coordinate_error, axis=1)
            df['action_type_category'] = df['model_action_type'].apply(self._categorize_action_type)
            df['performance_category'] = df['step_evaluation_score'].apply(self._categorize_performance)
            df['has_coordinate_data'] = df['step_coordinate_distance'].notna()
            df['has_text_data'] = df['step_text_match'].notna()
            
            # Ensure proper column order for readability
            column_order = [
                # Metadata
                'run_name', 'timestamp', 'dataset_name', 'ocr_module',
                
                # Episode info
                'episode_id', 'goal', 'episode_success_rate', 'episode_step_accuracy', 
                'episode_total_steps', 'episode_steps_completed', 'episode_overall_success_rate',
                
                # Step info
                'step_number', 'step_action_match', 'step_action_type_match', 
                'step_evaluation_score', 'step_task_completed',
                'step_coordinate_distance', 'step_text_match',
                
                # Model action
                'model_action_type', 'model_coordinates_y', 'model_coordinates_x',
                'model_lift_coordinates_y', 'model_lift_coordinates_x', 'model_text',
                
                # Ground truth action  
                'gt_action_type', 'gt_coordinates_y', 'gt_coordinates_x',
                'gt_lift_coordinates_y', 'gt_lift_coordinates_x', 'gt_text',
                
                # Computed columns
                'coordinate_error', 'action_type_category', 'performance_category',
                'has_coordinate_data', 'has_text_data',
                
                # Configuration
                'max_steps_per_episode', 'start_episode', 'end_episode'
            ]
            
            # Reorder columns, keeping any extra columns at the end
            available_columns = [col for col in column_order if col in df.columns]
            extra_columns = [col for col in df.columns if col not in column_order]
            final_columns = available_columns + extra_columns
            
            df = df[final_columns]
            
            # Export to CSV with proper formatting
            # CRITICAL: Use index=False to avoid unnamed index column
            df.to_csv(filename, index=False, float_format='%.6f')
            
            self.logger.info(f"Successfully exported {len(rows)} rows to {filename}")
            self.logger.info(f"CSV contains {len(df.columns)} columns with step-level detail")
            
            # Log summary statistics
            self._log_export_summary(df)
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def _calculate_coordinate_error(self, row) -> float:
        """Calculate coordinate error between model and ground truth."""
        try:
            if pd.isna(row['model_coordinates_y']) or pd.isna(row['gt_coordinates_y']):
                return None
                
            model_y = float(row['model_coordinates_y'])
            model_x = float(row['model_coordinates_x'])
            gt_y = float(row['gt_coordinates_y'])
            gt_x = float(row['gt_coordinates_x'])
            
            # Calculate Euclidean distance
            error = ((model_y - gt_y)**2 + (model_x - gt_x)**2)**0.5
            return error
            
        except (ValueError, TypeError):
            return None
    
    def _categorize_action_type(self, action_type) -> str:
        """Categorize action type for analysis."""
        if pd.isna(action_type):
            return 'unknown'
        
        action_str = str(action_type).upper()
        
        if 'DUAL_POINT' in action_str or action_str == '1':
            return 'dual_point'
        elif 'TYPE' in action_str or action_str == '5':
            return 'type'
        elif 'BACK' in action_str or action_str == '6':
            return 'navigation'
        elif 'HOME' in action_str or action_str == '7':
            return 'navigation'
        elif 'ENTER' in action_str or action_str == '8':
            return 'navigation'
        elif 'COMPLETE' in action_str or action_str == '9':
            return 'status'
        elif 'IMPOSSIBLE' in action_str or action_str == '10':
            return 'status'
        else:
            return 'other'
    
    def _categorize_performance(self, score) -> str:
        """Categorize performance score for analysis."""
        if pd.isna(score):
            return 'unknown'
        
        score = float(score)
        
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _log_export_summary(self, df: pd.DataFrame) -> None:
        """Log summary statistics of the exported data."""
        try:
            total_rows = len(df)
            total_episodes = df['episode_id'].nunique()
            total_datasets = df['dataset_name'].nunique()
            
            avg_success_rate = df.groupby('episode_id')['episode_success_rate'].first().mean()
            avg_step_score = df['step_evaluation_score'].mean()
            
            action_match_rate = df['step_action_match'].mean()
            action_type_match_rate = df['step_action_type_match'].mean()
            
            self.logger.info("Export Summary:")
            self.logger.info(f"  Total rows (steps): {total_rows}")
            self.logger.info(f"  Total episodes: {total_episodes}")
            self.logger.info(f"  Total datasets: {total_datasets}")
            self.logger.info(f"  Average episode success rate: {avg_success_rate:.3f}")
            self.logger.info(f"  Average step evaluation score: {avg_step_score:.3f}")
            self.logger.info(f"  Action match rate: {action_match_rate:.3f}")
            self.logger.info(f"  Action type match rate: {action_type_match_rate:.3f}")
            
            # Performance distribution
            perf_dist = df['performance_category'].value_counts()
            self.logger.info(f"  Performance distribution: {dict(perf_dist)}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate export summary: {e}")
    
    def export_summary_csv(self, episode_results: List[EpisodeResult], filename: str, config: BenchmarkConfig) -> None:
        """
        Export episode-level summary to CSV for high-level analysis.
        
        Args:
            episode_results: List of episode results to export
            filename: Output CSV filename
            config: Benchmark configuration
        """
        if not episode_results:
            self.logger.warning("No episode results to export")
            return
        
        try:
            self.logger.info(f"Exporting episode summary to CSV: {filename}")
            
            rows = []
            for episode in episode_results:
                # Calculate additional episode-level metrics
                total_steps = len(episode.step_results)
                action_matches = sum(1 for step in episode.step_results if step.action_match)
                action_type_matches = sum(1 for step in episode.step_results if step.action_type_match)
                avg_score = sum(step.evaluation_score for step in episode.step_results) / total_steps if total_steps > 0 else 0
                
                # Count different action types
                action_types = {}
                for step in episode.step_results:
                    action_type = self._categorize_action_type(step.model_action.get('action_type', 'unknown'))
                    action_types[action_type] = action_types.get(action_type, 0) + 1
                
                row = {
                    'run_name': config.run_name,
                    'timestamp': datetime.now().isoformat(),
                    'episode_id': episode.episode_id,
                    'dataset_name': episode.dataset_name,
                    'goal': episode.goal,
                    'success_rate': episode.success_rate,
                    'step_accuracy': episode.step_accuracy,
                    'total_steps': episode.total_steps,
                    'steps_completed': episode.steps_completed,
                    'overall_success_rate': episode.overall_success_rate,
                    'action_match_rate': action_matches / total_steps if total_steps > 0 else 0,
                    'action_type_match_rate': action_type_matches / total_steps if total_steps > 0 else 0,
                    'avg_evaluation_score': avg_score,
                    'ocr_module': config.ocr_module,
                    
                    # Action type counts
                    'dual_point_actions': action_types.get('dual_point', 0),
                    'type_actions': action_types.get('type', 0),
                    'navigation_actions': action_types.get('navigation', 0),
                    'status_actions': action_types.get('status', 0),
                    'other_actions': action_types.get('other', 0)
                }
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # CRITICAL: Use index=False to avoid unnamed index column
            df.to_csv(filename, index=False, float_format='%.6f')
            
            self.logger.info(f"Successfully exported episode summary: {len(rows)} episodes")
            
        except Exception as e:
            self.logger.error(f"Error exporting episode summary: {e}")
            raise
