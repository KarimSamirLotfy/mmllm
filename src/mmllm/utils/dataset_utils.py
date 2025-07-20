"""Dataset utilities for multi-dataset benchmarking support."""

import logging
from typing import List, Dict, Any, Optional, Iterator
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetUtils:
    """Utilities for dataset selection, validation, and management."""
    
    def __init__(self):
        """Initialize dataset utilities."""
        self.logger = logging.getLogger(__name__)
        
        # Dataset directories mapping (from aitw.py pattern)
        self.dataset_directories = {
            'general': 'gs://gresearch/android-in-the-wild/general/*',
            'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
            'install': 'gs://gresearch/android-in-the-wild/install/*',
            'single': 'gs://gresearch/android-in-the-wild/single/*',
            'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
        }
        
        # Dataset metadata for validation and information
        self.dataset_info = {
            'general': {
                'description': 'General Android interactions across various apps',
                'typical_episode_count': 1000,
                'avg_steps_per_episode': 8
            },
            'google_apps': {
                'description': 'Interactions with Google applications',
                'typical_episode_count': 500,
                'avg_steps_per_episode': 10
            },
            'install': {
                'description': 'App installation and setup tasks',
                'typical_episode_count': 300,
                'avg_steps_per_episode': 6
            },
            'single': {
                'description': 'Single-step interaction tasks',
                'typical_episode_count': 800,
                'avg_steps_per_episode': 3
            },
            'web_shopping': {
                'description': 'Web-based shopping interactions',
                'typical_episode_count': 400,
                'avg_steps_per_episode': 12
            }
        }
    
    def validate_dataset_names(self, dataset_names: List[str]) -> List[str]:
        """
        Validate and filter dataset names.
        
        Args:
            dataset_names: List of dataset names to validate
            
        Returns:
            List of valid dataset names
            
        Raises:
            ValueError: If no valid datasets are provided
        """
        if not dataset_names:
            raise ValueError("At least one dataset name must be provided")
        
        valid_datasets = set(self.dataset_directories.keys())
        invalid_datasets = []
        validated_datasets = []
        
        for dataset_name in dataset_names:
            if dataset_name in valid_datasets:
                validated_datasets.append(dataset_name)
            else:
                invalid_datasets.append(dataset_name)
        
        if invalid_datasets:
            self.logger.warning(f"Invalid dataset names: {invalid_datasets}")
            self.logger.info(f"Valid dataset options: {sorted(valid_datasets)}")
        
        if not validated_datasets:
            raise ValueError(f"No valid datasets found. Valid options: {sorted(valid_datasets)}")
        
        self.logger.info(f"Validated datasets: {validated_datasets}")
        return validated_datasets
    
    def check_dataset_availability(self, dataset_names: List[str]) -> Dict[str, bool]:
        """
        Check if datasets are available for access.
        
        Args:
            dataset_names: List of dataset names to check
            
        Returns:
            Dictionary mapping dataset names to availability status
        """
        availability = {}
        
        for dataset_name in dataset_names:
            try:
                if dataset_name not in self.dataset_directories:
                    availability[dataset_name] = False
                    continue
                
                # Try to access the dataset files
                filenames = tf.io.gfile.glob(self.dataset_directories[dataset_name])
                availability[dataset_name] = len(filenames) > 0
                
                if availability[dataset_name]:
                    self.logger.info(f"Dataset '{dataset_name}' available: {len(filenames)} files")
                else:
                    self.logger.warning(f"Dataset '{dataset_name}' not accessible or empty")
                    
            except Exception as e:
                self.logger.error(f"Error checking dataset '{dataset_name}': {e}")
                availability[dataset_name] = False
        
        return availability
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.dataset_info:
            return {
                'description': 'Unknown dataset',
                'typical_episode_count': 0,
                'avg_steps_per_episode': 0
            }
        
        info = self.dataset_info[dataset_name].copy()
        info['name'] = dataset_name
        info['directory_pattern'] = self.dataset_directories.get(dataset_name, 'Unknown')
        
        return info
    
    def estimate_processing_time(self, dataset_names: List[str], start_episode: int, 
                               end_episode: Optional[int], max_steps_per_episode: int) -> Dict[str, Any]:
        """
        Estimate processing time for the given configuration.
        
        Args:
            dataset_names: List of datasets to process
            start_episode: Starting episode index
            end_episode: Ending episode index (None for all)
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary with time estimates
        """
        # Time estimates (in seconds) - these are rough estimates
        TIME_PER_STEP = 2.0  # seconds per step processing
        TIME_PER_EPISODE_OVERHEAD = 5.0  # seconds overhead per episode
        TIME_PER_DATASET_OVERHEAD = 10.0  # seconds overhead per dataset
        
        total_episodes = 0
        total_steps = 0
        dataset_estimates = {}
        
        for dataset_name in dataset_names:
            info = self.get_dataset_info(dataset_name)
            
            # Calculate episodes for this dataset
            if end_episode is not None:
                episodes_for_dataset = min(end_episode - start_episode, info['typical_episode_count'])
            else:
                episodes_for_dataset = min(100, info['typical_episode_count'])  # Default limit
            
            # Calculate expected steps
            avg_steps = min(info['avg_steps_per_episode'], max_steps_per_episode)
            steps_for_dataset = episodes_for_dataset * avg_steps
            
            # Time estimate for this dataset
            dataset_time = (
                episodes_for_dataset * TIME_PER_EPISODE_OVERHEAD +
                steps_for_dataset * TIME_PER_STEP +
                TIME_PER_DATASET_OVERHEAD
            )
            
            dataset_estimates[dataset_name] = {
                'episodes': episodes_for_dataset,
                'expected_steps': steps_for_dataset,
                'estimated_time_seconds': dataset_time,
                'estimated_time_minutes': dataset_time / 60
            }
            
            total_episodes += episodes_for_dataset
            total_steps += steps_for_dataset
        
        total_time = sum(est['estimated_time_seconds'] for est in dataset_estimates.values())
        
        return {
            'total_episodes': total_episodes,
            'total_expected_steps': total_steps,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_time_hours': total_time / 3600,
            'dataset_breakdown': dataset_estimates,
            'recommendations': self._generate_time_recommendations(total_time, total_episodes)
        }
    
    def _generate_time_recommendations(self, total_time_seconds: float, total_episodes: int) -> List[str]:
        """Generate recommendations based on estimated processing time."""
        recommendations = []
        
        if total_time_seconds > 3600:  # More than 1 hour
            recommendations.append("Consider reducing the number of episodes or datasets for faster results")
        
        if total_episodes > 50:
            recommendations.append("Large number of episodes - consider running overnight or in batches")
        
        if total_time_seconds > 7200:  # More than 2 hours
            recommendations.append("Very long processing time - consider starting with a smaller subset")
        
        if total_time_seconds < 300:  # Less than 5 minutes
            recommendations.append("Quick run - good for testing and validation")
        
        return recommendations
    
    def filter_datasets_by_criteria(self, dataset_names: List[str], 
                                  min_episodes: Optional[int] = None,
                                  max_avg_steps: Optional[int] = None) -> List[str]:
        """
        Filter datasets based on criteria.
        
        Args:
            dataset_names: List of dataset names to filter
            min_episodes: Minimum number of episodes required
            max_avg_steps: Maximum average steps per episode
            
        Returns:
            Filtered list of dataset names
        """
        filtered = []
        
        for dataset_name in dataset_names:
            info = self.get_dataset_info(dataset_name)
            
            # Check minimum episodes
            if min_episodes is not None and info['typical_episode_count'] < min_episodes:
                self.logger.info(f"Filtered out '{dataset_name}': too few episodes ({info['typical_episode_count']} < {min_episodes})")
                continue
            
            # Check maximum average steps
            if max_avg_steps is not None and info['avg_steps_per_episode'] > max_avg_steps:
                self.logger.info(f"Filtered out '{dataset_name}': too many steps per episode ({info['avg_steps_per_episode']} > {max_avg_steps})")
                continue
            
            filtered.append(dataset_name)
        
        self.logger.info(f"Filtered datasets: {filtered}")
        return filtered
    
    def get_recommended_datasets(self, purpose: str = 'general') -> List[str]:
        """
        Get recommended datasets for different purposes.
        
        Args:
            purpose: Purpose of the benchmark ('general', 'quick', 'comprehensive', 'ui_focused')
            
        Returns:
            List of recommended dataset names
        """
        recommendations = {
            'general': ['google_apps', 'general'],
            'quick': ['single', 'google_apps'],
            'comprehensive': ['general', 'google_apps', 'install', 'single', 'web_shopping'],
            'ui_focused': ['general', 'google_apps'],
            'simple_tasks': ['single', 'install'],
            'complex_tasks': ['web_shopping', 'general']
        }
        
        recommended = recommendations.get(purpose, ['google_apps'])
        
        self.logger.info(f"Recommended datasets for '{purpose}': {recommended}")
        return recommended
    
    def create_dataset_summary(self, dataset_names: List[str]) -> str:
        """
        Create a human-readable summary of the selected datasets.
        
        Args:
            dataset_names: List of dataset names
            
        Returns:
            Formatted summary string
        """
        summary_lines = ["Dataset Summary:", "=" * 50]
        
        total_episodes = 0
        total_avg_steps = 0
        
        for dataset_name in dataset_names:
            info = self.get_dataset_info(dataset_name)
            total_episodes += info['typical_episode_count']
            total_avg_steps += info['avg_steps_per_episode']
            
            summary_lines.append(f"• {dataset_name.upper()}")
            summary_lines.append(f"  Description: {info['description']}")
            summary_lines.append(f"  Episodes: ~{info['typical_episode_count']}")
            summary_lines.append(f"  Avg steps: ~{info['avg_steps_per_episode']}")
            summary_lines.append("")
        
        summary_lines.extend([
            f"TOTAL ESTIMATED:",
            f"  Episodes: ~{total_episodes}",
            f"  Avg steps per episode: ~{total_avg_steps / len(dataset_names):.1f}",
            "=" * 50
        ])
        
        return "\n".join(summary_lines)
