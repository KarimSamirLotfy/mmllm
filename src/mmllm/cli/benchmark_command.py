"""CLI command for comprehensive benchmarking pipeline."""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from mmllm.evaluation.benchmarking_pipeline import BenchmarkingPipeline
from mmllm.evaluation.types import BenchmarkConfig
from mmllm.evaluation.config_manager import ConfigManager
from mmllm.utils.dataset_utils import DatasetUtils

logger = logging.getLogger(__name__)


class BenchmarkCommand:
    """CLI command handler for benchmarking operations."""
    
    def __init__(self):
        """Initialize benchmark command handler."""
        self.config_manager = ConfigManager()
        self.dataset_utils = DatasetUtils()
        self.logger = logging.getLogger(__name__)
    
    def add_parser(self, subparsers) -> argparse.ArgumentParser:
        """
        Add benchmark command parser to the main CLI.
        
        Args:
            subparsers: Subparsers object from main argument parser
            
        Returns:
            Benchmark command parser
        """
        parser = subparsers.add_parser(
            'benchmark',
            help='Run comprehensive benchmarking pipeline',
            description='Execute comprehensive benchmarking across multiple AiTW datasets with configurable parameters'
        )
        
        # Dataset selection
        parser.add_argument(
            '--datasets',
            nargs='+',
            default=['google_apps'],
            choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
            help='Datasets to include in benchmark (default: google_apps)'
        )
        
        # Episode range
        parser.add_argument(
            '--start-episode',
            type=int,
            default=0,
            help='Starting episode index (default: 0)'
        )
        
        parser.add_argument(
            '--end-episode',
            type=int,
            default=None,
            help='Ending episode index (exclusive). If not specified, will process limited episodes per dataset'
        )
        
        parser.add_argument(
            '--episodes',
            type=str,
            default=None,
            help='Episode range in format "start:end" (e.g., "0:5"). Overrides --start-episode and --end-episode'
        )
        
        # Agent configuration
        parser.add_argument(
            '--ocr-module',
            action='store_true',
            default=True,
            help='Enable OCR module in agent (default: True)'
        )
        
        parser.add_argument(
            '--no-ocr',
            action='store_true',
            help='Disable OCR module in agent'
        )
        
        # Processing parameters
        parser.add_argument(
            '--max-steps',
            type=int,
            default=10,
            help='Maximum steps per episode (default: 10)'
        )
        
        # Output configuration
        parser.add_argument(
            '--output-dir',
            type=str,
            default='./benchmark_results',
            help='Output directory for results (default: ./benchmark_results)'
        )
        
        parser.add_argument(
            '--run-name',
            type=str,
            default=None,
            help='Name for this benchmark run (default: auto-generated)'
        )
        
        # Configuration management
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Load configuration from JSON file'
        )
        
        parser.add_argument(
            '--save-config',
            type=str,
            default=None,
            help='Save configuration to JSON file before running'
        )
        
        # Output options
        parser.add_argument(
            '--csv-only',
            action='store_true',
            help='Only generate CSV output (skip visualizations)'
        )
        
        parser.add_argument(
            '--no-visualizations',
            action='store_true',
            help='Skip visualization generation'
        )
        
        # Utility options
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show configuration and time estimates without running benchmark'
        )
        
        parser.add_argument(
            '--recommended',
            choices=['general', 'quick', 'comprehensive', 'ui_focused'],
            help='Use recommended dataset configuration for specific purpose'
        )
        
        parser.add_argument(
            '--estimate-time',
            action='store_true',
            help='Show time estimates for the benchmark configuration'
        )
        
        parser.set_defaults(func=self.run_benchmark)
        
        return parser
    
    def run_benchmark(self, args) -> int:
        """
        Execute benchmark based on command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Load configuration if specified
            if args.config:
                self.logger.info(f"Loading configuration from: {args.config}")
                config = self.config_manager.load_config(args.config)
                
                # Override with command line arguments
                if hasattr(args, 'datasets') and args.datasets != ['google_apps']:
                    config.dataset_names = args.datasets
                if hasattr(args, 'output_dir') and args.output_dir != './benchmark_results':
                    config.output_dir = args.output_dir
                
            else:
                # Create configuration from command line arguments
                config = self._create_config_from_args(args)
            
            # Validate configuration
            self.config_manager.validate_config(config)
            
            # Check dataset availability
            self.logger.info("Checking dataset availability...")
            availability = self.dataset_utils.check_dataset_availability(config.dataset_names)
            unavailable = [name for name, available in availability.items() if not available]
            
            if unavailable:
                self.logger.warning(f"Unavailable datasets: {unavailable}")
                config.dataset_names = [name for name in config.dataset_names if availability.get(name, False)]
                
                if not config.dataset_names:
                    self.logger.error("No datasets are available for benchmarking")
                    return 1
            
            # Show time estimates
            if args.estimate_time or args.dry_run:
                self._show_time_estimates(config)
            
            # Show configuration summary
            self._show_config_summary(config)
            
            # Save configuration if requested
            if args.save_config:
                self.config_manager.save_config(config, args.save_config)
                self.logger.info(f"Configuration saved to: {args.save_config}")
            
            # Dry run - exit without executing
            if args.dry_run:
                self.logger.info("Dry run completed - no benchmark executed")
                return 0
            
            # Execute benchmark
            self.logger.info("Starting benchmark execution...")
            pipeline = BenchmarkingPipeline(config)
            
            # Run benchmark

            episode_results = pipeline.run_benchmark()
            
            if not episode_results:
                self.logger.error("No episode results generated")
                raise RuntimeError("Benchmark did not produce any results")
            
            # Generate comprehensive report
            report = pipeline.generate_comprehensive_report(episode_results)
            
            # Print summary
            self._print_benchmark_summary(report, config)
            
            self.logger.info("Benchmark completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Benchmark interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return 1
    
    def _create_config_from_args(self, args) -> BenchmarkConfig:
        """Create BenchmarkConfig from command line arguments."""
        # Handle dataset recommendations
        if args.recommended:
            datasets = self.dataset_utils.get_recommended_datasets(args.recommended)
            self.logger.info(f"Using recommended datasets for '{args.recommended}': {datasets}")
        else:
            datasets = args.datasets
        
        # Validate datasets
        datasets = self.dataset_utils.validate_dataset_names(datasets)
        
        # Handle episode range
        start_episode = args.start_episode
        end_episode = args.end_episode
        
        if args.episodes:
            try:
                start_str, end_str = args.episodes.split(':')
                start_episode = int(start_str)
                end_episode = int(end_str) if end_str else None
            except ValueError:
                raise ValueError(f"Invalid episode range format: {args.episodes}. Use 'start:end' format")
        
        # Handle OCR module setting
        ocr_module = args.ocr_module and not args.no_ocr
        
        # Generate run name if not provided
        run_name = args.run_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_suffix = "ocr" if ocr_module else "no_ocr"
            dataset_suffix = "_".join(datasets[:2])  # First 2 datasets
            run_name = f"benchmark_{dataset_suffix}_{ocr_suffix}_{timestamp}"
        
        # Create output directory
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = BenchmarkConfig(
            dataset_names=datasets,
            ocr_module=ocr_module,
            start_episode=start_episode,
            end_episode=end_episode,
            output_dir=str(output_dir),
            run_name=run_name,
            max_steps_per_episode=args.max_steps
        )
        
        return config
    
    def _show_time_estimates(self, config: BenchmarkConfig) -> None:
        """Show time estimates for the benchmark configuration."""
        estimates = self.dataset_utils.estimate_processing_time(
            config.dataset_names,
            config.start_episode,
            config.end_episode,
            config.max_steps_per_episode
        )
        
        self.logger.info("=== Time Estimates ===")
        self.logger.info(f"Total episodes: {estimates['total_episodes']}")
        self.logger.info(f"Total expected steps: {estimates['total_expected_steps']}")
        self.logger.info(f"Estimated time: {estimates['total_time_minutes']:.1f} minutes ({estimates['total_time_hours']:.1f} hours)")
        
        if estimates['recommendations']:
            self.logger.info("Recommendations:")
            for rec in estimates['recommendations']:
                self.logger.info(f"  • {rec}")
        
        # Dataset breakdown
        self.logger.info("\nDataset breakdown:")
        for dataset, breakdown in estimates['dataset_breakdown'].items():
            self.logger.info(f"  {dataset}: {breakdown['episodes']} episodes, "
                           f"{breakdown['expected_steps']} steps, "
                           f"{breakdown['estimated_time_minutes']:.1f} min")
    
    def _show_config_summary(self, config: BenchmarkConfig) -> None:
        """Show configuration summary."""
        self.logger.info("=== Benchmark Configuration ===")
        self.logger.info(f"Run name: {config.run_name}")
        self.logger.info(f"Datasets: {', '.join(config.dataset_names)}")
        self.logger.info(f"OCR module: {config.ocr_module}")
        
        episode_range = f"{config.start_episode}-{config.end_episode}" if config.end_episode else f"{config.start_episode}+"
        self.logger.info(f"Episode range: {episode_range}")
        self.logger.info(f"Max steps per episode: {config.max_steps_per_episode}")
        self.logger.info(f"Output directory: {config.output_dir}")
        
        # Dataset information
        dataset_summary = self.dataset_utils.create_dataset_summary(config.dataset_names)
        print(dataset_summary)
    
    def _print_benchmark_summary(self, report: Dict[str, Any], config: BenchmarkConfig) -> None:
        """Print final benchmark summary."""
        self.logger.info("=== Benchmark Results Summary ===")
        
        overall_metrics = report.get('overall_metrics', {})
        self.logger.info(f"Total episodes processed: {overall_metrics.get('total_episodes', 0)}")
        self.logger.info(f"Total steps: {overall_metrics.get('total_steps', 0)}")
        self.logger.info(f"Overall accuracy: {overall_metrics.get('accuracy', 0):.1%}")
        self.logger.info(f"Overall success rate: {overall_metrics.get('success_rate', 0):.1%}")
        self.logger.info(f"Precision: {overall_metrics.get('precision', 0):.3f}")
        self.logger.info(f"Recall: {overall_metrics.get('recall', 0):.3f}")
        self.logger.info(f"F1 Score: {overall_metrics.get('f1_score', 0):.3f}")
        
        # Dataset breakdown
        dataset_metrics = report.get('dataset_metrics', {})
        if dataset_metrics:
            self.logger.info("\nDataset breakdown:")
            for dataset, metrics in dataset_metrics.items():
                self.logger.info(f"  {dataset}: "
                               f"Accuracy={metrics.get('accuracy', 0):.1%}, "
                               f"F1={metrics.get('f1_score', 0):.3f}, "
                               f"Episodes={metrics.get('total_episodes', 0)}")
        
        # Output files
        output_files = report.get('output_files', {})
        self.logger.info("\nGenerated files:")
        for file_type, filepath in output_files.items():
            if filepath:
                self.logger.info(f"  {file_type}: {filepath}")


def add_benchmark_command(parser: argparse.ArgumentParser) -> None:
    """
    Add benchmark command to the main CLI parser.
    
    Args:
        parser: Main argument parser
    """
    subparsers = parser.add_subparser('benchmark')
    benchmark_cmd = BenchmarkCommand()
    benchmark_cmd.add_parser(subparsers)
