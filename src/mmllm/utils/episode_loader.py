"""Episode loading utilities for Android in the Wild dataset."""

import tensorflow as tf
from typing import List, Dict, Any, Optional
from ..vision.image_processor import ImageProcessor
from ..vision.annotation_parser import AnnotationParser
from ..multi_agent.state import MultiAgentState, AgentPhase, EpisodeEvaluationState
from ..android_in_the_wild.action_type import ActionType


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
    
    def load_episode_with_history(self, episode_examples: List[tf.train.Example]) -> Dict[str, Any]:
        """
        Load episode with full historical context for evaluation.
        
        Args:
            episode_examples: List of tf.train.Example from the episode
            
        Returns:
            Dictionary with episode data including historical context
        """
        if not episode_examples:
            raise ValueError("No episode examples provided")
        
        episode_images = []
        ground_truth_actions = []
        ui_annotations_list = []
        
        # Extract episode metadata from first example
        first_example = episode_examples[0]
        episode_info = self.annotation_parser.extract_episode_info(first_example)
        episode_id = episode_info.get("episode_id", "unknown")
        goal = episode_info.get("goal_info", "Unknown goal")
        
        # Process each step in the episode
        for step_idx, example in enumerate(episode_examples):
            try:
                # Decode and encode image
                image_array = self.image_processor.decode_aitw_image(example)
                base64_image = self.image_processor.encode_image_for_model(image_array)
                episode_images.append(base64_image)
                
                # Extract UI annotations for this step
                ui_annotations = self.annotation_parser.parse_ui_annotations(example)
                ui_annotations_list.append(ui_annotations)
                
                # Extract ground truth action
                ground_truth_action = self._extract_ground_truth_action(example)
                ground_truth_actions.append(ground_truth_action)
                
            except Exception as e:
                # Handle individual step errors gracefully
                print(f"Error processing step {step_idx}: {e}")
                # Use fallback data
                episode_images.append("")
                ui_annotations_list.append([])
                ground_truth_actions.append({"action_type": ActionType.STATUS_TASK_IMPOSSIBLE})
        
        return {
            "episode_id": episode_id,
            "goal": goal,
            "episode_length": len(episode_examples),
            "episode_images": episode_images,
            "ground_truth_actions": ground_truth_actions,
            "ui_annotations_list": ui_annotations_list
        }
    
    def create_evaluation_state_for_step(
        self, 
        episode_data: Dict[str, Any], 
        step_index: int
    ) -> EpisodeEvaluationState:
        """
        Create an EpisodeEvaluationState for a specific step with historical context.
        
        Args:
            episode_data: Episode data from load_episode_with_history()
            step_index: Current step index (0-based)
            
        Returns:
            EpisodeEvaluationState with historical context
        """
        if step_index >= len(episode_data["episode_images"]):
            raise ValueError(f"Step index {step_index} out of range")
        
        # Historical context: all images from start to current step
        historical_images = episode_data["episode_images"][:step_index + 1]
        
        state: EpisodeEvaluationState = {
            # Core goal and step info
            "goal": episode_data["goal"],
            "current_step": step_index,
            "max_steps": min(episode_data["episode_length"], 10),  # Limit for memory
            "current_phase": AgentPhase.PLANNING,
            
            # Current state
            "current_image": episode_data["episode_images"][step_index],
            "ui_annotations": episode_data["ui_annotations_list"][step_index],
            
            # Historical context
            "episode_images": historical_images,
            "episode_id": episode_data["episode_id"],
            "episode_length": episode_data["episode_length"],
            
            # Ground truth for evaluation
            "current_ground_truth": episode_data["ground_truth_actions"][step_index],
            "ground_truth_actions": episode_data["ground_truth_actions"],
            
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
            
            # Error handling
            "error_count": 0,
            "last_error": None,
            
            # Evaluation tracking
            "step_evaluations": None,
            "final_result": None
        }
        
        return state

    def _extract_ground_truth_action(self, example: tf.train.Example) -> Dict[str, Any]:
        """Extract ground truth action from tf.train.Example."""
        try:
            features = example.features.feature
            
            # Extract action type
            if 'action_type' in features:
                action_type = features['action_type'].int64_list.value[0]
            else:
                action_type = ActionType.DUAL_POINT.value  # Default
            
            action = {
                "action_type": action_type
            }
            
            # Extract coordinates for DUAL_POINT actions
            if action_type == ActionType.DUAL_POINT.value:
                if 'touch_point_x' in features and 'touch_point_y' in features:
                    # Coordinates in AiTW format: normalized [y, x]
                    touch_y = features['touch_point_y'].float_list.value[0]
                    touch_x = features['touch_point_x'].float_list.value[0]
                    action["coordinates"] = [touch_y, touch_x]
                    
                    # Check for lift point (swipe)
                    if 'lift_point_x' in features and 'lift_point_y' in features:
                        lift_y = features['lift_point_y'].float_list.value[0]
                        lift_x = features['lift_point_x'].float_list.value[0]
                        action["lift_coordinates"] = [lift_y, lift_x]
            
            # Extract text for TYPE actions
            elif action_type == ActionType.TYPE.value:
                if 'typed_text' in features:
                    text = features['typed_text'].bytes_list.value[0].decode('utf-8')
                    action["text"] = text
            
            return action
            
        except Exception as e:
            print(f"Error extracting ground truth action: {e}")
            # Return fallback action
            return {
                "action_type": ActionType.STATUS_TASK_IMPOSSIBLE.value
            }

    # ...existing methods...
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
