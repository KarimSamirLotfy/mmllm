import logging
import argparse
import sys
import langgraph
from mmllm.agent import run_agent
from mmllm.multi_agent import MultiAgentCoordinator
from mmllm.utils import EpisodeLoader
from mmllm.evaluation import EpisodeRunner
from mmllm.android_in_the_wild import visualization_utils
import tensorflow as tf
import base64
import io
from PIL import Image

from mmllm.utils.prints import print_result
from mmllm.cli.benchmark_command import BenchmarkCommand

logger = logging.getLogger(__name__)


def get_dataset():
    dataset_name = 'google_apps'  #@param ["general", "google_apps", "install", "single", "web_shopping"]

    dataset_directories = {
        'general': 'gs://gresearch/android-in-the-wild/general/*',
        'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
        'install': 'gs://gresearch/android-in-the-wild/install/*',
        'single': 'gs://gresearch/android-in-the-wild/single/*',
        'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
    }
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
    return raw_dataset


def get_episode(dataset):
    """Grabs the first complete episode."""
    episode = []
    episode_id = None
    for d in dataset:
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            break
    return episode


def run_demo_mode():
    """Run original demo functionality."""
    logger.info("=== Multi-Agent Android in the Wild Demo ===")
    
    # Initialize episode loader and coordinator
    episode_loader = EpisodeLoader()
    coordinator = MultiAgentCoordinator()
    
    try:
        # Load sample episode state
        logger.info("Loading sample episode...")
        state = episode_loader.get_sample_episode_state()
        
        logger.info(f"Goal: {state['goal']}")
        logger.info(f"Episode ID: {state.get('episode_id', 'Unknown')}")
        logger.info(f"Episode Length: {state.get('episode_length', 'Unknown')}")
        
        # Run multi-agent system
        logger.info("Running multi-agent coordination...")
        result = coordinator.run(state)
        
        print_result(result)
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


def run_evaluation_mode(dataset_name: str, num_episodes: int, max_steps: int):
    """Run episode evaluation mode."""
    logger.info("=== Episode Evaluation Mode ===")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Episodes to evaluate: {num_episodes}")
    logger.info(f"Max steps per episode: {max_steps}")
    
    try:
        # Initialize episode runner
        runner = EpisodeRunner()
        
        # Run batch evaluation
        result = runner.run_batch_evaluation(
            dataset_name=dataset_name,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps
        )
        
        # Print summary
        batch_summary = result.get("batch_summary", {})
        logger.info(f"Evaluation complete!")
        logger.info(f"Episodes processed: {batch_summary.get('episodes_processed', 0)}")
        
        if result.get("episode_results"):
            avg_success = sum(
                ep.get("overall_success_rate", 0) for ep in result["episode_results"]
            ) / len(result["episode_results"])
            logger.info(f"Average success rate: {avg_success:.1%}")
            
            # Print individual episode results
            for i, episode_result in enumerate(result["episode_results"], 1):
                episode_id = episode_result.get("episode_id", f"episode_{i}")
                success_rate = episode_result.get("overall_success_rate", 0)
                total_steps = episode_result.get("total_steps", 0)
                successful_steps = episode_result.get("successful_steps", 0)
                
                logger.info(f"Episode {i} ({episode_id}): {success_rate:.1%} success ({successful_steps}/{total_steps} steps)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-Agent Android in the Wild System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add benchmark command
    benchmark_cmd = BenchmarkCommand()
    benchmark_cmd.add_parser(subparsers)
    
    # Add original demo/evaluation commands as subcommands
    demo_parser = subparsers.add_parser('demo', help='Run original demo functionality')
    demo_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    demo_parser.set_defaults(func=lambda args: run_demo_mode())
    
    eval_parser = subparsers.add_parser('evaluate', help='Run episode evaluation')
    eval_parser.add_argument(
        '--dataset', 
        default='google_apps',
        choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
        help='Dataset to use for evaluation'
    )
    eval_parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to evaluate')
    eval_parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    eval_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    eval_parser.set_defaults(func=lambda args: run_evaluation_mode(args.dataset, args.episodes, args.max_steps))
    
    # Legacy support - if no subcommand specified, run demo mode
    parser.add_argument(
        '--mode', 
        choices=['demo', 'evaluation'], 
        default='evaluation',
        help='Legacy mode selection (use subcommands instead)'
    )
    parser.add_argument(
        '--dataset', 
        default='google_apps',
        choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
        help='Dataset to use for evaluation mode'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=3,
        help='Number of episodes to evaluate in evaluation mode'
    )
    parser.add_argument(
        '--max-steps', 
        type=int, 
        default=10,
        help='Maximum steps per episode in evaluation mode'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    
    # Route to appropriate function
    if hasattr(args, 'func'):
        # New subcommand-based routing
        return args.func(args)
    elif args.command:
        # Handle case where subcommand was specified but no func set
        logger.error(f"Unknown command: {args.command}")
        return 1
    else:
        # Legacy routing
        logger.info(f"Starting in {args.mode} mode")
        
        if args.mode == 'demo':
            return run_demo_mode()
        elif args.mode == 'evaluation':
            return run_evaluation_mode(args.dataset, args.episodes, args.max_steps)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1


# Legacy main function for backwards compatibility
def legacy_main():
    """Original main function preserved for compatibility."""
    return run_demo_mode()


if __name__ == "__main__":
    main()