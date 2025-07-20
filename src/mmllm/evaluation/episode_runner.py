"""Episode runner for backwards compatibility with main.py evaluation mode."""

import logging
from typing import List, Dict, Any
from mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from mmllm.evaluation.types import BenchmarkConfig

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Backwards compatibility wrapper for BenchmarkingPipeline."""
    
    def __init__(self):
        """Initialize episode runner."""
        self.logger = logging.getLogger(__name__)
    
    def run_batch_evaluation(self, dataset_name: str, num_episodes: int, max_steps_per_episode: int = 10) -> Dict[str, Any]:
        """
        Run batch evaluation using BenchmarkingPipeline.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            num_episodes: Number of episodes to process
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Evaluation results in the expected format
        """
        try:
            # Create temporary config for evaluation
            config = BenchmarkConfig(
                dataset_names=[dataset_name],
                ocr_module=True,  # Default to True
                start_episode=0,
                end_episode=num_episodes,
                output_dir="./evaluation_output",
                run_name=f"evaluation_{dataset_name}",
                max_steps_per_episode=max_steps_per_episode
            )
            
            # Run benchmarking pipeline
            pipeline = BenchmarkingPipeline(config)
            episode_results = pipeline.run_benchmark()
            
            # Convert to expected format
            converted_results = []
            for episode_result in episode_results:
                converted_result = {
                    "episode_id": episode_result.episode_id,
                    "overall_success_rate": episode_result.success_rate / 100.0,  # Convert back to decimal
                    "total_steps": episode_result.total_steps,
                    "successful_steps": episode_result.steps_completed,
                    "step_evaluations": [
                        {
                            "step_number": step.step_number,
                            "action_match": step.action_match,
                            "evaluation_score": step.evaluation_score
                        }
                        for step in episode_result.step_results
                    ]
                }
                converted_results.append(converted_result)
            
            # Create batch summary
            batch_summary = {
                "episodes_processed": len(converted_results),
                "total_steps": sum(r["total_steps"] for r in converted_results),
                "successful_steps": sum(r["successful_steps"] for r in converted_results),
                "average_success_rate": sum(r["overall_success_rate"] for r in converted_results) / len(converted_results) if converted_results else 0.0
            }
            
            return {
                "episode_results": converted_results,
                "batch_summary": batch_summary,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
