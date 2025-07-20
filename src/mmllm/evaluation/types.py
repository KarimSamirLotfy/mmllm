"""Data types for evaluation system."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class StepResult:
    """Result for individual step."""
    episode_id: str
    step_number: int
    action_match: bool
    model_action: Dict[str, Any]
    ground_truth_action: Dict[str, Any]
    task_completed: bool
    evaluation_score: float
    action_type_match: bool
    coordinate_distance: Optional[float] = None
    text_match: Optional[bool] = None


@dataclass
class EpisodeResult:
    """Result for complete episode."""
    episode_id: str
    dataset_name: str
    success_rate: float  # Percentage of steps completed before first error
    step_accuracy: float  # Percentage of correct steps
    total_steps: int
    steps_completed: int
    step_results: List[StepResult]
    goal: str
    overall_success_rate: float


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for analysis."""
    accuracy: float
    success_rate: float
    precision: float
    recall: float
    f1_score: float
    total_episodes: int
    total_steps: int
    dataset_name: str
    ocr_module: bool


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    dataset_names: List[str]  # ['general', 'google_apps', etc.]
    ocr_module: bool
    start_episode: int
    end_episode: Optional[int]
    output_dir: str
    run_name: str
    max_steps_per_episode: int = 10
    max_workers: int = 4  # Number of parallel workers
    batch_size: int = 5   # Episodes per batch for worker processing
    prompt_with_android_tree: bool = False  # Use Android tree prompt instead of default
    add_image_history: bool = False  # Include image history in agent context
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.dataset_names:
            raise ValueError("At least one dataset must be specified")
        
        valid_datasets = {'general', 'google_apps', 'install', 'single', 'web_shopping'}
        for dataset in self.dataset_names:
            if dataset not in valid_datasets:
                raise ValueError(f"Invalid dataset: {dataset}. Must be one of {valid_datasets}")
        
        if self.start_episode < 0:
            raise ValueError("start_episode must be non-negative")
        
        if self.end_episode is not None and self.end_episode <= self.start_episode:
            raise ValueError("end_episode must be greater than start_episode")
