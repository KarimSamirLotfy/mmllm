"""Metrics calculator for F1, recall, precision calculations in benchmarking."""

import logging
from typing import List, Dict, Any
import numpy as np

from mmllm.evaluation.types import EpisodeResult, BenchmarkMetrics

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive metrics for benchmarking results."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, episode_results: List[EpisodeResult], dataset_name: str, ocr_module: bool) -> BenchmarkMetrics:
        """
        Calculate comprehensive metrics for episode results.
        
        Args:
            episode_results: List of episode results
            dataset_name: Name of the dataset being evaluated
            ocr_module: Whether OCR module was used
            
        Returns:
            BenchmarkMetrics with calculated values
        """
        if not episode_results:
            self.logger.warning("No episode results provided for metrics calculation")
            return BenchmarkMetrics(
                accuracy=0.0,
                success_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_episodes=0,
                total_steps=0,
                dataset_name=dataset_name,
                ocr_module=ocr_module
            )
        
        try:
            # Aggregate all step results
            all_step_results = []
            for episode in episode_results:
                all_step_results.extend(episode.step_results)
            
            if not all_step_results:
                self.logger.warning("No step results found in episodes")
                return BenchmarkMetrics(
                    accuracy=0.0,
                    success_rate=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    total_episodes=len(episode_results),
                    total_steps=0,
                    dataset_name=dataset_name,
                    ocr_module=ocr_module
                )
            
            # Calculate basic metrics
            total_steps = len(all_step_results)
            total_episodes = len(episode_results)
            
            # Accuracy: percentage of correct action matches
            correct_actions = sum(1 for step in all_step_results if step.action_match)
            accuracy = correct_actions / total_steps if total_steps > 0 else 0.0
            
            # Success rate: average of episode success rates
            success_rates = [episode.success_rate for episode in episode_results]
            success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
            
            # Calculate precision, recall, and F1 score
            # For action matching, we treat:
            # - True Positive (TP): Agent predicted action and it matched ground truth
            # - False Positive (FP): Agent predicted action but it didn't match ground truth  
            # - False Negative (FN): Ground truth had action but agent failed to match it
            # - True Negative (TN): Not typically applicable in this context
            
            true_positives = sum(1 for step in all_step_results if step.action_match)
            false_positives = sum(1 for step in all_step_results if not step.action_match)
            
            # For recall calculation, we consider all ground truth actions as positives
            # since every step has a ground truth action that should be matched
            total_ground_truth_actions = total_steps
            false_negatives = false_positives  # Failed matches are false negatives
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics = BenchmarkMetrics(
                accuracy=accuracy,
                success_rate=success_rate,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_episodes=total_episodes,
                total_steps=total_steps,
                dataset_name=dataset_name,
                ocr_module=ocr_module
            )
            
            self.logger.info(f"Calculated metrics for {dataset_name}:")
            self.logger.info(f"  Accuracy: {accuracy:.3f}")
            self.logger.info(f"  Success Rate: {success_rate:.3f}")
            self.logger.info(f"  Precision: {precision:.3f}")
            self.logger.info(f"  Recall: {recall:.3f}")
            self.logger.info(f"  F1 Score: {f1_score:.3f}")
            self.logger.info(f"  Episodes: {total_episodes}, Steps: {total_steps}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return BenchmarkMetrics(
                accuracy=0.0,
                success_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_episodes=len(episode_results),
                total_steps=0,
                dataset_name=dataset_name,
                ocr_module=ocr_module
            )
    
    def calculate_detailed_metrics(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Calculate detailed metrics with breakdowns by action type, coordinate accuracy, etc.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            Dictionary with detailed metric breakdowns
        """
        if not episode_results:
            return {}
        
        try:
            all_step_results = []
            for episode in episode_results:
                all_step_results.extend(episode.step_results)
            
            # Action type accuracy
            action_type_matches = sum(1 for step in all_step_results if step.action_type_match)
            action_type_accuracy = action_type_matches / len(all_step_results) if all_step_results else 0.0
            
            # Coordinate accuracy (for DUAL_POINT actions)
            coordinate_steps = [step for step in all_step_results if step.coordinate_distance is not None]
            coordinate_matches = sum(1 for step in coordinate_steps if step.coordinate_distance <= 0.14)
            coordinate_accuracy = coordinate_matches / len(coordinate_steps) if coordinate_steps else 0.0
            
            # Text accuracy (for TYPE actions)
            text_steps = [step for step in all_step_results if step.text_match is not None]
            text_matches = sum(1 for step in text_steps if step.text_match)
            text_accuracy = text_matches / len(text_steps) if text_steps else 0.0
            
            # Average evaluation scores
            evaluation_scores = [step.evaluation_score for step in all_step_results]
            avg_evaluation_score = np.mean(evaluation_scores) if evaluation_scores else 0.0
            
            # Average coordinate distance
            coordinate_distances = [step.coordinate_distance for step in all_step_results if step.coordinate_distance is not None]
            avg_coordinate_distance = np.mean(coordinate_distances) if coordinate_distances else 0.0
            
            detailed_metrics = {
                "action_type_accuracy": action_type_accuracy,
                "coordinate_accuracy": coordinate_accuracy,
                "text_accuracy": text_accuracy,
                "avg_evaluation_score": avg_evaluation_score,
                "avg_coordinate_distance": avg_coordinate_distance,
                "total_coordinate_steps": len(coordinate_steps),
                "total_text_steps": len(text_steps),
                "score_distribution": {
                    "perfect_scores": sum(1 for score in evaluation_scores if score >= 0.95),
                    "good_scores": sum(1 for score in evaluation_scores if 0.8 <= score < 0.95),
                    "fair_scores": sum(1 for score in evaluation_scores if 0.5 <= score < 0.8),
                    "poor_scores": sum(1 for score in evaluation_scores if score < 0.5)
                }
            }
            
            return detailed_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating detailed metrics: {e}")
            return {}
    
    def compare_configurations(self, results_a: List[EpisodeResult], results_b: List[EpisodeResult], 
                             config_a_name: str, config_b_name: str) -> Dict[str, Any]:
        """
        Compare metrics between two different configurations (e.g., with/without OCR).
        
        Args:
            results_a: Episode results from first configuration
            results_b: Episode results from second configuration
            config_a_name: Name of first configuration
            config_b_name: Name of second configuration
            
        Returns:
            Comparison analysis
        """
        try:
            metrics_a = self.calculate_metrics(results_a, config_a_name, False)  # OCR module doesn't matter for comparison
            metrics_b = self.calculate_metrics(results_b, config_b_name, False)
            
            comparison = {
                "configurations": {
                    config_a_name: {
                        "accuracy": metrics_a.accuracy,
                        "success_rate": metrics_a.success_rate,
                        "precision": metrics_a.precision,
                        "recall": metrics_a.recall,
                        "f1_score": metrics_a.f1_score,
                        "total_episodes": metrics_a.total_episodes,
                        "total_steps": metrics_a.total_steps
                    },
                    config_b_name: {
                        "accuracy": metrics_b.accuracy,
                        "success_rate": metrics_b.success_rate,
                        "precision": metrics_b.precision,
                        "recall": metrics_b.recall,
                        "f1_score": metrics_b.f1_score,
                        "total_episodes": metrics_b.total_episodes,
                        "total_steps": metrics_b.total_steps
                    }
                },
                "improvements": {
                    "accuracy": metrics_b.accuracy - metrics_a.accuracy,
                    "success_rate": metrics_b.success_rate - metrics_a.success_rate,
                    "precision": metrics_b.precision - metrics_a.precision,
                    "recall": metrics_b.recall - metrics_a.recall,
                    "f1_score": metrics_b.f1_score - metrics_a.f1_score
                },
                "better_configuration": {
                    "accuracy": config_b_name if metrics_b.accuracy > metrics_a.accuracy else config_a_name,
                    "success_rate": config_b_name if metrics_b.success_rate > metrics_a.success_rate else config_a_name,
                    "precision": config_b_name if metrics_b.precision > metrics_a.precision else config_a_name,
                    "recall": config_b_name if metrics_b.recall > metrics_a.recall else config_a_name,
                    "f1_score": config_b_name if metrics_b.f1_score > metrics_a.f1_score else config_a_name
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing configurations: {e}")
            return {}
