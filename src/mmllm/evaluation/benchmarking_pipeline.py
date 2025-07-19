"""Comprehensive benchmarking pipeline for AiTW dataset evaluation."""

import os
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
import jax.numpy as jnp
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from mmllm.agent_ocr.aitw import get_episodes
from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent
from mmllm.utils.episode_loader import EpisodeLoader
from mmllm.evaluation.metrics_calculator import MetricsCalculator
from mmllm.evaluation.csv_exporter import CSVExporter
from mmllm.evaluation.config_manager import ConfigManager
from mmllm.evaluation.benchmark_visualizer import BenchmarkVisualizer
from mmllm.evaluation.types import BenchmarkConfig, StepResult, EpisodeResult, BenchmarkMetrics
from mmllm.android_in_the_wild.action_matching import check_actions_match

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings for AiTW dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'


def process_episode_batch(batch_data: Tuple[List[Any], str, BenchmarkConfig, int]) -> List[EpisodeResult]:
    """
    Worker function to process a batch of episodes in parallel.
    
    Args:
        batch_data: Tuple containing (episode_batch, dataset_name, config, start_idx)
        
    Returns:
        List of EpisodeResult objects
    """
    episode_batch, dataset_name, config, start_idx = batch_data
    
    # Initialize components for this worker process
    agent = SimpleOCRAgent(ocr_module=config.ocr_module)
    episode_loader = EpisodeLoader()
    
    # Set up logging for this worker
    worker_logger = logging.getLogger(f"worker_{mp.current_process().pid}")
    
    results = []
    
    for i, episode_tf in enumerate(episode_batch):
        episode_idx = start_idx + i
        worker_logger.info(f"Worker {mp.current_process().pid} processing episode {episode_idx}")
        
        try:
            episode_result = _process_single_episode(
                episode_tf, dataset_name, episode_idx, agent, episode_loader, config, worker_logger
            )
            if episode_result:
                results.append(episode_result)
        except Exception as e:
            worker_logger.error(f"Error processing episode {episode_idx}: {e}")
            continue
    
    worker_logger.info(f"Worker {mp.current_process().pid} completed batch: {len(results)} episodes processed")
    return results


def _process_single_episode(
    episode_tf: List[tf.train.Example], 
    dataset_name: str, 
    episode_idx: int,
    agent: SimpleOCRAgent,
    episode_loader: EpisodeLoader,
    config: BenchmarkConfig,
    logger: logging.Logger
) -> Optional[EpisodeResult]:
    """
    Process a single episode for evaluation.
    
    Args:
        episode_tf: Episode as list of tf.train.Example objects
        dataset_name: Name of the dataset
        episode_idx: Index of the episode
        agent: SimpleOCRAgent instance
        episode_loader: EpisodeLoader instance
        config: Benchmark configuration
        logger: Logger instance
        
    Returns:
        EpisodeResult or None if processing failed
    """
    try:
        # Load episode with history using episode loader
        episode = episode_loader.load_episode_with_history(episode_tf)
        
        model_actions = []
        step_results = []
        
        episode_id = episode['episode_id']
        goal = episode['goal']
        number_of_steps = len(episode['episode_images'])
        
        # Process steps in the episode
        total_steps = min(config.max_steps_per_episode, number_of_steps)
        steps_completed = 0
        first_error_step = None
        
        # Start from reasonable position in episode (like aitw.py)
        start_step = 0
        logger.info(f"Processing episode {episode_id} with {total_steps} steps (starting from step {start_step})")
        
        for step in range(start_step, min(start_step + total_steps, number_of_steps)):
            try:
                image = episode['episode_images'][step]
                ui_annotations = episode['ui_annotations_list'][step]
                ground_truth_action = episode['ground_truth_actions'][step] if 'ground_truth_actions' in episode else None
                
                if not ground_truth_action:
                    continue
                
                # Run the agent for this step
                result = agent.run_step(
                    image=image,
                    ocr_text=None,
                    ui_description=ui_annotations,
                    goal=goal,
                    thread_id=episode_id
                )
                
                action = result.get('action', {})
                model_actions.append(action)
                
                # Check action matching using the same logic as aitw.py
                annotation_positions = jnp.array([ui_element['position'] for ui_element in ui_annotations])
                
                action_match = check_actions_match(
                    action_1_touch_yx=action.get('coordinates', [0, 0]),
                    action_1_lift_yx=action.get('lift_coordinates', [0, 0]),
                    action_1_action_type=action['action_type'],
                    action_2_touch_yx=ground_truth_action.get('coordinates', [0, 0]),
                    action_2_lift_yx=ground_truth_action.get('lift_coordinates', [0, 0]),
                    action_2_action_type=ground_truth_action['action_type'],
                    annotation_positions=annotation_positions
                )
                
                # Check if action types match
                action_type_match = action['action_type'] == ground_truth_action['action_type']
                
                # Calculate coordinate distance if both have coordinates
                coordinate_distance = None
                if 'coordinates' in action and 'coordinates' in ground_truth_action:
                    coord_diff = jnp.array(action['coordinates']) - jnp.array(ground_truth_action['coordinates'])
                    coordinate_distance = float(jnp.sqrt(jnp.sum(coord_diff ** 2)))
                
                # Check text match if applicable
                text_match = None
                if action.get('text') and ground_truth_action.get('text'):
                    text_match = action['text'].lower() == ground_truth_action['text'].lower()
                
                logger.debug(f"Episode {episode_idx}, Step {step + 1} Action: {json.dumps(action, indent=2)}, Ground Truth: {json.dumps(ground_truth_action, indent=2)} Matches: {action_match}, Type Match: {action_type_match}, Distance: {coordinate_distance}, Text Match: {text_match}")
                
                step_result = StepResult(
                    episode_id=episode_id,
                    step_number=step,
                    action_match=bool(action_match),
                    model_action=action,
                    ground_truth_action=ground_truth_action,
                    task_completed=result.get('task_completed', False),
                    evaluation_score=1.0 if action_match else 0.0,
                    action_type_match=action_type_match,
                    coordinate_distance=coordinate_distance,
                    text_match=text_match
                )
                
                step_results.append(step_result)
                
                if action_match:
                    steps_completed += 1
                elif first_error_step is None:
                    first_error_step = step
                
                # Check if task is completed
                if result.get('task_completed', False):
                    logger.info(f"Episode {episode_idx} - Task completed at step {step}!")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing step {step} in episode {episode_idx}: {e}")
                continue
        
        # Calculate success rate: percentage of steps completed before first error
        if first_error_step is not None:
            success_rate = (first_error_step - start_step) / total_steps * 100.0
        else:
            success_rate = 100.0  # No errors found
            
        # Calculate step accuracy
        step_accuracy = steps_completed / len(step_results) if step_results else 0.0
        
        episode_result = EpisodeResult(
            episode_id=episode_id,
            dataset_name=dataset_name,
            success_rate=success_rate,
            step_accuracy=step_accuracy * 100.0,  # Convert to percentage
            total_steps=len(step_results),
            steps_completed=steps_completed,
            step_results=step_results,
            goal=goal,
            overall_success_rate=success_rate
        )
        
        return episode_result
        
    except Exception as e:
        logger.error(f"Error processing episode {episode_idx}: {e}")
        return None


class BenchmarkingPipeline:
    """Main benchmarking orchestrator."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmarking pipeline."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.agent = SimpleOCRAgent(ocr_module=config.ocr_module)
        self.episode_loader = EpisodeLoader()
        self.metrics_calculator = MetricsCalculator()
        self.csv_exporter = CSVExporter()
        self.config_manager = ConfigManager()
        self.visualizer = BenchmarkVisualizer()
        
        # Dataset directories (from aitw.py pattern)
        self.dataset_directories = {
            'general': 'gs://gresearch/android-in-the-wild/general/*',
            'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
            'install': 'gs://gresearch/android-in-the-wild/install/*',
            'single': 'gs://gresearch/android-in-the-wild/single/*',
            'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info(f'BenchmarkingPipeline initialized with config: {config}')


    def run_benchmark(self) -> List[EpisodeResult]:
        """
        Run comprehensive benchmark across all configured datasets using parallel processing.
        
        Returns:
            List of episode results from all datasets
        """
        self.logger.info(f"Starting parallel benchmark execution with {self.config.max_workers} workers...")
        
        # Save configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(self.config.output_dir, f"config_{timestamp}.json")
        self.config_manager.save_config(self.config, config_path)
        
        all_episode_results = []
        
        for dataset_name in self.config.dataset_names:
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                # Load dataset using TensorFlow pattern from aitw.py
                filenames = tf.io.gfile.glob(self.dataset_directories[dataset_name])
                if not filenames:
                    self.logger.warning(f"No files found for dataset {dataset_name}")
                    continue
                    
                raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
                
                # Collect episodes into batches for parallel processing
                episode_batches = self._prepare_episode_batches(raw_dataset, dataset_name)
                
                if not episode_batches:
                    self.logger.warning(f"No episodes found for dataset {dataset_name}")
                    continue
                
                # Process batches in parallel
                dataset_results = self._process_batches_parallel(episode_batches)
                
                # Flatten results from all batches
                for batch_results in dataset_results:
                    all_episode_results.extend(batch_results)
                
                total_episodes = sum(len(batch_results) for batch_results in dataset_results)
                self.logger.info(f"Completed {dataset_name}: {total_episodes} episodes processed")
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
        
        self.logger.info(f"Parallel benchmark completed: {len(all_episode_results)} total episodes processed")
        return all_episode_results

    def _prepare_episode_batches(self, raw_dataset, dataset_name: str) -> List[Tuple[List[Any], str, BenchmarkConfig, int]]:
        """
        Prepare episode batches for parallel processing.
        
        Args:
            raw_dataset: TensorFlow dataset iterator
            dataset_name: Name of the dataset
            
        Returns:
            List of batch tuples for worker processing
        """
        self.logger.info(f"Preparing episode batches for {dataset_name}...")
        
        episode_batches = []
        current_batch = []
        episode_count = 0
        
        # Collect episodes into batches
        for episode_idx, episode_tf in enumerate(get_episodes(
            raw_dataset, 
            start_episode=self.config.start_episode, 
            end_episode=self.config.end_episode
        )):
            current_batch.append(episode_tf)
            episode_count += 1
            
            # When batch is full, add it to batches list
            if len(current_batch) >= self.config.batch_size:
                start_idx = episode_idx - len(current_batch) + 1 + self.config.start_episode
                batch_data = (current_batch.copy(), dataset_name, self.config, start_idx)
                episode_batches.append(batch_data)
                current_batch = []
        
        # Add remaining episodes as final batch
        if current_batch:
            start_idx = episode_count - len(current_batch) + self.config.start_episode
            batch_data = (current_batch, dataset_name, self.config, start_idx)
            episode_batches.append(batch_data)
        
        self.logger.info(f"Created {len(episode_batches)} batches for {dataset_name} ({episode_count} total episodes)")
        return episode_batches

    def _process_batches_parallel(self, episode_batches: List[Tuple[List[Any], str, BenchmarkConfig, int]]) -> List[List[EpisodeResult]]:
        """
        Process episode batches in parallel using ProcessPoolExecutor.
        
        Args:
            episode_batches: List of batch tuples for processing
            
        Returns:
            List of results from each batch
        """
        self.logger.info(f"Processing {len(episode_batches)} batches with {self.config.max_workers} workers...")
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches to the executor
            future_to_batch = {
                executor.submit(process_episode_batch, batch_data): i 
                for i, batch_data in enumerate(episode_batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.append(batch_results)
                    self.logger.info(f"Completed batch {batch_idx + 1}/{len(episode_batches)}: {len(batch_results)} episodes")
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    all_results.append([])  # Add empty results for failed batch
        
        return all_results

    def run_benchmark_sequential(self) -> List[EpisodeResult]:
        """
        Run benchmark sequentially (fallback for debugging or when parallel processing fails).
        
        Returns:
            List of episode results from all datasets
        """
        self.logger.info("Starting sequential benchmark execution...")
        
        # Save configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(self.config.output_dir, f"config_{timestamp}.json")
        self.config_manager.save_config(self.config, config_path)
        
        all_episode_results = []
        
        for dataset_name in self.config.dataset_names:
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                # Load dataset using TensorFlow pattern from aitw.py
                filenames = tf.io.gfile.glob(self.dataset_directories[dataset_name])
                if not filenames:
                    self.logger.warning(f"No files found for dataset {dataset_name}")
                    continue
                    
                raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
                
                # Process episodes sequentially
                dataset_results = []
                for episode_idx, episode_tf in enumerate(get_episodes(
                    raw_dataset, 
                    start_episode=self.config.start_episode, 
                    end_episode=self.config.end_episode
                )):
                    self.logger.info(f"Processing episode {episode_idx + self.config.start_episode} from {dataset_name}")
                    
                    try:
                        episode_result = _process_single_episode(
                            episode_tf, dataset_name, episode_idx, 
                            self.agent, self.episode_loader, self.config, self.logger
                        )
                        if episode_result:
                            dataset_results.append(episode_result)
                            all_episode_results.append(episode_result)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing episode {episode_idx} from {dataset_name}: {e}")
                        continue
                
                self.logger.info(f"Completed {dataset_name}: {len(dataset_results)} episodes processed")
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
        
        self.logger.info(f"Sequential benchmark completed: {len(all_episode_results)} total episodes processed")
        return all_episode_results

    def generate_comprehensive_report(self, episode_results: List[EpisodeResult]) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report with all outputs.
        
        Args:
            episode_results: List of episode results
            
        Returns:
            Comprehensive report dictionary
        """
        if not episode_results:
            return {"error": "No episode results to report"}
        
        self.logger.info("Generating comprehensive benchmark report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics for each dataset
        dataset_metrics = {}
        for dataset_name in self.config.dataset_names:
            dataset_episodes = [ep for ep in episode_results if ep.dataset_name == dataset_name]
            if dataset_episodes:
                metrics = self.metrics_calculator.calculate_metrics(dataset_episodes, dataset_name, self.config.ocr_module)
                dataset_metrics[dataset_name] = metrics
        
        # Calculate overall metrics
        overall_metrics = self.metrics_calculator.calculate_metrics(episode_results, "overall", self.config.ocr_module)
        
        # Export to CSV
        csv_filename = os.path.join(self.config.output_dir, f"benchmark_results_{timestamp}.csv")
        self.csv_exporter.export_to_csv(episode_results, csv_filename, self.config)
        
        # Generate visualizations
        chart_filename = os.path.join(self.config.output_dir, f"performance_charts_{timestamp}.png")
        # self.visualizer.create_performance_charts(episode_results, dataset_metrics, chart_filename, self.config)
        
        # Create comprehensive report
        report = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_name": self.config.run_name,
                "total_episodes": len(episode_results),
                "datasets": self.config.dataset_names,
                "ocr_module": self.config.ocr_module,
                "evaluation_framework": "mmllm-aitw-benchmarking",
                "parallel_processing": {
                    "max_workers": self.config.max_workers,
                    "batch_size": self.config.batch_size,
                    "enabled": True
                }
            },
            "overall_metrics": {
                "accuracy": overall_metrics.accuracy,
                "success_rate": overall_metrics.success_rate,
                "precision": overall_metrics.precision,
                "recall": overall_metrics.recall,
                "f1_score": overall_metrics.f1_score,
                "total_episodes": overall_metrics.total_episodes,
                "total_steps": overall_metrics.total_steps
            },
            "dataset_metrics": {
                dataset: {
                    "accuracy": metrics.accuracy,
                    "success_rate": metrics.success_rate,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "total_episodes": metrics.total_episodes,
                    "total_steps": metrics.total_steps
                }
                for dataset, metrics in dataset_metrics.items()
            },
            "configuration": {
                "dataset_names": self.config.dataset_names,
                "ocr_module": self.config.ocr_module,
                "start_episode": self.config.start_episode,
                "end_episode": self.config.end_episode,
                "max_steps_per_episode": self.config.max_steps_per_episode,
                "max_workers": self.config.max_workers,
                "batch_size": self.config.batch_size
            },
            "episode_results": [
                {
                    "episode_id": ep.episode_id,
                    "dataset_name": ep.dataset_name,
                    "success_rate": ep.success_rate,
                    "step_accuracy": ep.step_accuracy,
                    "total_steps": ep.total_steps,
                    "steps_completed": ep.steps_completed,
                    "goal": ep.goal
                }
                for ep in episode_results
            ],
            "output_files": {
                "csv_export": csv_filename,
                "visualization": chart_filename,
                "configuration": f"config_{timestamp}.json"
            }
        }
        
        # Save JSON report
        json_filename = os.path.join(self.config.output_dir, f"benchmark_report_{timestamp}.json")
        self.config_manager.save_json(report, json_filename)
        
        self.logger.info(f"Comprehensive report generated:")
        self.logger.info(f"  - JSON report: {json_filename}")
        self.logger.info(f"  - CSV export: {csv_filename}")
        self.logger.info(f"  - Visualizations: {chart_filename}")
        
        return report
