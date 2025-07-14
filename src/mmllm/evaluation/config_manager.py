"""Configuration management for benchmarking with JSON save/load functionality."""

import json
import logging
from typing import Dict, Any
from pathlib import Path
from dataclasses import asdict

from mmllm.evaluation.types import BenchmarkConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration save/load for reproducible benchmarking runs."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
    
    def save_config(self, config: BenchmarkConfig, filename: str) -> None:
        """
        Save benchmark configuration to JSON file for reproducibility.
        
        Args:
            config: BenchmarkConfig object to save
            filename: Output JSON filename
        """
        try:
            self.logger.info(f"Saving configuration to: {filename}")
            
            # Convert dataclass to dictionary
            config_dict = asdict(config)
            
            # Add metadata for traceability
            config_dict['_metadata'] = {
                'config_version': '1.0',
                'saved_at': Path(filename).name,
                'config_type': 'BenchmarkConfig'
            }
            
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON with proper formatting
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Configuration saved successfully")
            self._log_config_summary(config)
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def load_config(self, filename: str) -> BenchmarkConfig:
        """
        Load benchmark configuration from JSON file.
        
        Args:
            filename: JSON filename to load
            
        Returns:
            BenchmarkConfig object
        """
        try:
            self.logger.info(f"Loading configuration from: {filename}")
            
            if not Path(filename).exists():
                raise FileNotFoundError(f"Configuration file not found: {filename}")
            
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            # Remove metadata if present
            if '_metadata' in config_dict:
                metadata = config_dict.pop('_metadata')
                self.logger.info(f"Loaded config version: {metadata.get('config_version', 'unknown')}")
            
            # Validate required fields
            self._validate_config_dict(config_dict)
            
            # Create BenchmarkConfig object
            config = BenchmarkConfig(**config_dict)
            
            self.logger.info("Configuration loaded successfully")
            self._log_config_summary(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_json(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save arbitrary JSON data to file.
        
        Args:
            data: Dictionary to save as JSON
            filename: Output filename
        """
        try:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"JSON data saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving JSON data: {e}")
            raise
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            filename: JSON filename to load
            
        Returns:
            Dictionary with loaded data
        """
        try:
            if not Path(filename).exists():
                raise FileNotFoundError(f"JSON file not found: {filename}")
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"JSON data loaded from: {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {e}")
            raise
    
    def validate_config(self, config: BenchmarkConfig) -> bool:
        """
        Validate benchmark configuration parameters.
        
        Args:
            config: BenchmarkConfig to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        try:
            # Basic validation (already done in BenchmarkConfig.__post_init__)
            if not config.dataset_names:
                raise ValueError("At least one dataset must be specified")
            
            valid_datasets = {'general', 'google_apps', 'install', 'single', 'web_shopping'}
            for dataset in config.dataset_names:
                if dataset not in valid_datasets:
                    raise ValueError(f"Invalid dataset: {dataset}")
            
            if config.start_episode < 0:
                raise ValueError("start_episode must be non-negative")
            
            if config.end_episode is not None and config.end_episode <= config.start_episode:
                raise ValueError("end_episode must be greater than start_episode")
            
            if config.max_steps_per_episode <= 0:
                raise ValueError("max_steps_per_episode must be positive")
            
            if not config.output_dir:
                raise ValueError("output_dir must be specified")
            
            if not config.run_name:
                raise ValueError("run_name must be specified")
            
            # Additional validations
            if config.end_episode is not None and (config.end_episode - config.start_episode) > 1000:
                self.logger.warning("Large episode range detected - this may take a very long time")
            
            if config.max_steps_per_episode > 50:
                self.logger.warning("High max_steps_per_episode - this may take a long time per episode")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _validate_config_dict(self, config_dict: Dict[str, Any]) -> None:
        """Validate configuration dictionary before creating BenchmarkConfig."""
        required_fields = [
            'dataset_names', 'ocr_module', 'start_episode', 'output_dir', 'run_name'
        ]
        
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Type validation
        if not isinstance(config_dict['dataset_names'], list):
            raise ValueError("dataset_names must be a list")
        
        if not isinstance(config_dict['ocr_module'], bool):
            raise ValueError("ocr_module must be a boolean")
        
        if not isinstance(config_dict['start_episode'], int):
            raise ValueError("start_episode must be an integer")
        
        if config_dict.get('end_episode') is not None and not isinstance(config_dict['end_episode'], int):
            raise ValueError("end_episode must be an integer or null")
    
    def _log_config_summary(self, config: BenchmarkConfig) -> None:
        """Log summary of configuration for verification."""
        try:
            episode_range = f"{config.start_episode}-{config.end_episode}" if config.end_episode else f"{config.start_episode}+"
            
            self.logger.info("Configuration Summary:")
            self.logger.info(f"  Run name: {config.run_name}")
            self.logger.info(f"  Datasets: {', '.join(config.dataset_names)}")
            self.logger.info(f"  OCR module: {config.ocr_module}")
            self.logger.info(f"  Episode range: {episode_range}")
            self.logger.info(f"  Max steps per episode: {config.max_steps_per_episode}")
            self.logger.info(f"  Output directory: {config.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not log config summary: {e}")
    
    def create_default_config(self, output_dir: str, run_name: str) -> BenchmarkConfig:
        """
        Create a default configuration for quick testing.
        
        Args:
            output_dir: Output directory for results
            run_name: Name for this benchmark run
            
        Returns:
            Default BenchmarkConfig
        """
        config = BenchmarkConfig(
            dataset_names=['google_apps'],
            ocr_module=True,
            start_episode=0,
            end_episode=3,
            output_dir=output_dir,
            run_name=run_name,
            max_steps_per_episode=10
        )
        
        self.logger.info("Created default configuration")
        return config
    
    def merge_configs(self, base_config: BenchmarkConfig, overrides: Dict[str, Any]) -> BenchmarkConfig:
        """
        Merge configuration overrides with base configuration.
        
        Args:
            base_config: Base configuration
            overrides: Dictionary of fields to override
            
        Returns:
            New BenchmarkConfig with overrides applied
        """
        try:
            # Convert to dict and apply overrides
            config_dict = asdict(base_config)
            config_dict.update(overrides)
            
            # Create new config
            new_config = BenchmarkConfig(**config_dict)
            
            # Validate the merged config
            self.validate_config(new_config)
            
            self.logger.info(f"Merged configuration with {len(overrides)} overrides")
            return new_config
            
        except Exception as e:
            self.logger.error(f"Error merging configurations: {e}")
            raise
