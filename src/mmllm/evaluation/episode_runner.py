"""Episode runner for step-by-step evaluation against AiTW ground truth."""

import logging
from typing import List, Dict, Any, Optional
import tensorflow as tf

from ..utils.episode_loader import EpisodeLoader
from ..multi_agent.coordinator import MultiAgentCoordinator
from ..multi_agent.state import StepEvaluationResult
from .action_comparator import ActionComparator
from .evaluation_reporter import EvaluationReporter

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Run step-by-step episode evaluation."""
    
    def __init__(self):
        self.episode_loader = EpisodeLoader()
        self.coordinator = MultiAgentCoordinator()
        self.action_comparator = ActionComparator()
        self.reporter = EvaluationReporter()
    
    def run_episode_evaluation(
        self, 
        episode_examples: List[tf.train.Example],
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete episode evaluation against ground truth.
        
        Args:
            episode_examples: List of tf.train.Example for the episode
            max_steps: Maximum number of steps to evaluate (for memory management)
            
        Returns:
            Episode evaluation result with step-by-step analysis
        """
        try:
            # Load episode with historical context
            episode_data = self.episode_loader.load_episode_with_history(episode_examples)
            
            # Limit steps for memory management
            total_steps = episode_data["episode_length"]
            if max_steps is not None:
                total_steps = min(total_steps, max_steps)
            
            logger.info(f"Starting evaluation for episode {episode_data['episode_id']} ({total_steps} steps)")
            
            step_evaluations = []
            
            # Evaluate each step
            for step_idx in range(total_steps):
                try:
                    step_evaluation = self._evaluate_step(episode_data, step_idx)
                    step_evaluations.append(step_evaluation)
                    
                    logger.info(
                        f"Step {step_idx + 1}/{total_steps}: "
                        f"Score {step_evaluation.evaluation_score:.2f}, "
                        f"Match: {step_evaluation.action_match}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error evaluating step {step_idx}: {e}")
                    # Create failure evaluation
                    step_evaluations.append(
                        StepEvaluationResult(
                            step_number=step_idx,
                            agent_action={"action_type": "ERROR"},
                            ground_truth_action=episode_data["ground_truth_actions"][step_idx],
                            action_match=False,
                            action_type_match=False,
                            evaluation_score=0.0
                        )
                    )
            
            # Calculate overall metrics
            overall_success_rate = (
                sum(step.evaluation_score for step in step_evaluations) / len(step_evaluations)
                if step_evaluations else 0.0
            )
            
            result = {
                "episode_id": episode_data["episode_id"],
                "goal": episode_data["goal"],
                "total_steps": total_steps,
                "step_evaluations": step_evaluations,
                "overall_success_rate": overall_success_rate,
                "successful_steps": sum(1 for step in step_evaluations if step.action_match),
                "failed_steps": sum(1 for step in step_evaluations if not step.action_match)
            }
            
            logger.info(
                f"Episode evaluation complete: {overall_success_rate:.1%} success rate "
                f"({result['successful_steps']}/{total_steps} steps)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running episode evaluation: {e}")
            return {
                "episode_id": "unknown",
                "goal": "unknown",
                "total_steps": 0,
                "step_evaluations": [],
                "overall_success_rate": 0.0,
                "error": str(e)
            }
    
    def _evaluate_step(self, episode_data: Dict[str, Any], step_index: int) -> StepEvaluationResult:
        """
        Evaluate a single step of the episode.
        
        Args:
            episode_data: Complete episode data
            step_index: Index of the step to evaluate
            
        Returns:
            StepEvaluationResult for this step
        """
        try:
            # Create evaluation state for this step with historical context
            state = self.episode_loader.create_evaluation_state_for_step(episode_data, step_index)
            
            # Add any previous actions to the state for context
            if step_index > 0:
                # Add previous step results as historical context
                previous_actions = []
                for i in range(step_index):
                    if i < len(episode_data["ground_truth_actions"]):
                        # For evaluation, we assume previous steps were executed correctly
                        # This simulates the agent having perfect execution up to this point
                        gt_action = episode_data["ground_truth_actions"][i]
                        previous_actions.append(gt_action)
                
                # Convert to planning outputs format (simplified)
                state["action_history"] = []  # Could be enhanced with actual planning outputs
            
            # Run multi-agent system for this step
            logger.debug(f"Running multi-agent system for step {step_index}")
            result = self.coordinator.run(state)
            
            # Extract agent's action output
            execution_output = result.get("execution_output")
            if not execution_output:
                raise ValueError("No execution output from multi-agent system")
            
            # Convert agent output to AiTW format for comparison
            agent_action = self._convert_to_aitw_format(result)
            
            # Get ground truth action
            ground_truth_action = episode_data["ground_truth_actions"][step_index]
            
            # Compare actions
            evaluation = self.action_comparator.compare_actions(
                agent_action, 
                ground_truth_action, 
                step_index
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating step {step_index}: {e}")
            # Return failure evaluation
            return StepEvaluationResult(
                step_number=step_index,
                agent_action={"action_type": "ERROR", "error": str(e)},
                ground_truth_action=episode_data["ground_truth_actions"][step_index],
                action_match=False,
                action_type_match=False,
                evaluation_score=0.0
            )
    
    def _convert_to_aitw_format(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert multi-agent system output to AiTW action format.
        
        Args:
            agent_result: Result from multi-agent coordinator
            
        Returns:
            Action in AiTW format
        """
        try:
            planning_output = agent_result.get("planning_output")
            if not planning_output:
                return {"action_type": "ERROR", "error": "No planning output"}
            
            # Use the action comparator's conversion function
            return self.action_comparator.format_agent_action_for_aitw(planning_output)
            
        except Exception as e:
            logger.error(f"Error converting to AiTW format: {e}")
            return {"action_type": "ERROR", "error": str(e)}
    
    def run_batch_evaluation(
        self, 
        dataset_name: str = 'google_apps', 
        num_episodes: int = 5,
        max_steps_per_episode: int = 10
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple episodes.
        
        Args:
            dataset_name: Name of the dataset to load
            num_episodes: Number of episodes to evaluate
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Batch evaluation results
        """
        try:
            logger.info(f"Starting batch evaluation: {num_episodes} episodes from {dataset_name}")
            
            # Load dataset
            dataset = self.episode_loader.load_dataset(dataset_name)
            
            episode_results = []
            episodes_processed = 0
            
            while episodes_processed < num_episodes:
                try:
                    # Get next episode
                    episode = self.episode_loader.get_episode(dataset)
                    if not episode:
                        logger.warning("No more episodes available in dataset")
                        break
                    
                    # Evaluate episode
                    result = self.run_episode_evaluation(episode, max_steps_per_episode)
                    episode_results.append(result)
                    episodes_processed += 1
                    
                    logger.info(f"Completed episode {episodes_processed}/{num_episodes}")
                    
                except Exception as e:
                    logger.error(f"Error processing episode {episodes_processed + 1}: {e}")
                    # Continue with next episode
                    episodes_processed += 1
            
            # Generate batch report
            batch_result = self.reporter.generate_batch_report(episode_results)
            
            logger.info(f"Batch evaluation complete: {len(episode_results)} episodes processed")
            
            return {
                "batch_summary": {
                    "dataset_name": dataset_name,
                    "episodes_processed": len(episode_results),
                    "episodes_requested": num_episodes
                },
                "episode_results": episode_results,
                "batch_report": batch_result
            }
            
        except Exception as e:
            logger.error(f"Error running batch evaluation: {e}")
            return {
                "batch_summary": {
                    "dataset_name": dataset_name,
                    "episodes_processed": 0,
                    "episodes_requested": num_episodes,
                    "error": str(e)
                },
                "episode_results": [],
                "batch_report": {"error": str(e)}
            }


def main():
    """Main entry point for episode evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AiTW episode evaluation")
    parser.add_argument("--dataset", default="google_apps", help="Dataset name to evaluate")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps per episode")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    runner = EpisodeRunner()
    
    try:
        result = runner.run_batch_evaluation(
            dataset_name=args.dataset,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps
        )
        
        print("Evaluation complete!")
        print(f"Episodes processed: {result['batch_summary']['episodes_processed']}")
        
        if result["episode_results"]:
            avg_success = sum(
                ep.get("overall_success_rate", 0) for ep in result["episode_results"]
            ) / len(result["episode_results"])
            print(f"Average success rate: {avg_success:.1%}")
    
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
