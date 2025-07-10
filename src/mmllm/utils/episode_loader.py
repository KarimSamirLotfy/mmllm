"""Episode loading utilities for Android in the Wild dataset."""

import tensorflow as tf
from typing import List, Dict, Any, Optional
from ..vision.image_processor import ImageProcessor
from ..vision.annotation_parser import AnnotationParser
from ..multi_agent.state import MultiAgentState, AgentPhase


class EpisodeLoader:
    """Loads and processes AiTW dataset episodes."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.annotation_parser = AnnotationParser()
        
        # Dataset configuration
        self.dataset_directories = {
            'general': 'gs://gresearch/android-in-the-wild/general/*',
            'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
            'install': 'gs://gresearch/android-in-the-wild/install/*',
            'single': 'gs://gresearch/android-in-the-wild/single/*',
            'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
        }
    
    def load_dataset(self, dataset_name: str = 'google_apps'):
        """Load raw dataset iterator."""
        if dataset_name not in self.dataset_directories:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        filenames = tf.io.gfile.glob(self.dataset_directories[dataset_name])
        return tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
    
    def get_episode(self, dataset_iterator) -> List[tf.train.Example]:
        """Extract the first complete episode from dataset."""
        episode = []
        episode_id = None
        
        for data in dataset_iterator:
            example = tf.train.Example()
            example.ParseFromString(data)
            
            current_episode_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            
            if episode_id is None:
                episode_id = current_episode_id
                episode.append(example)
            elif current_episode_id == episode_id:
                episode.append(example)
            else:
                break
        
        return episode
    
    def episode_to_multi_agent_state(self, episode: List[tf.train.Example], step_index: int = 0) -> MultiAgentState:
        """Convert AiTW episode step to MultiAgentState."""
        if not episode or step_index >= len(episode):
            raise ValueError("Invalid episode or step index")
        
        example = episode[step_index]
        
        # Extract episode metadata
        episode_info = self.annotation_parser.extract_episode_info(example)
        
        # Process image
        image_array = self.image_processor.decode_aitw_image(example)
        base64_image = self.image_processor.encode_image_for_model(image_array)
        
        # Parse UI annotations
        ui_annotations = self.annotation_parser.parse_ui_annotations(example)
        
        # Create initial state
        state: MultiAgentState = {
            # Core goal and current step
            "goal": episode_info.get("goal_info", "Unknown goal"),
            "current_step": 0,
            "max_steps": min(episode_info.get("episode_length", 10), 10),  # Limit to 10 steps
            
            # Current phase and agent control
            "current_phase": AgentPhase.PLANNING,
            
            # Multimodal inputs
            "current_image": base64_image,
            "ui_annotations": ui_annotations,
            
            # Messages and communication
            "messages": [],
            
            # Agent outputs (initially None)
            "planning_output": None,
            "execution_output": None,
            "reflection_output": None,
            
            # Historical data
            "action_history": [],
            "execution_history": [],
            "reflection_history": [],
            
            # Episode context
            "episode_id": episode_info.get("episode_id"),
            "episode_length": episode_info.get("episode_length"),
            
            # Error handling
            "error_count": 0,
            "last_error": None,
            
            # Success tracking
            "final_result": None
        }
        
        return state
    
    def get_sample_episode_state(self, dataset_name: str = 'google_apps') -> MultiAgentState:
        """Get a sample episode state for testing."""
        try:
            dataset = self.load_dataset(dataset_name)
            episode = self.get_episode(dataset)
            
            if not episode:
                raise ValueError("No episode found in dataset")
            
            return self.episode_to_multi_agent_state(episode, 0)
        
        except Exception as e:
            # Fallback mock state for testing when dataset is not available
            return self._create_mock_state()
    
    def _create_mock_state(self) -> MultiAgentState:
        """Create a mock state for testing when dataset is unavailable."""
        # Create a simple mock base64 image (1x1 pixel PNG)
        mock_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        return {
            "goal": "Find and tap the back button",
            "current_step": 0,
            "max_steps": 5,
            "current_phase": AgentPhase.PLANNING,
            "current_image": mock_image,
            "ui_annotations": [
                {
                    "position": [0.1, 0.1, 0.1, 0.1],
                    "text": "Back",
                    "ui_type": "button",
                    "index": 0
                }
            ],
            "messages": [],
            "planning_output": None,
            "execution_output": None,
            "reflection_output": None,
            "action_history": [],
            "execution_history": [],
            "reflection_history": [],
            "episode_id": "mock_episode_001",
            "episode_length": 5,
            "error_count": 0,
            "last_error": None,
            "final_result": None
        }
