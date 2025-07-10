"""Parse AiTW UI annotations."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class AnnotationParser:
    """Parses UI annotations from Android in the Wild dataset."""
    
    def __init__(self):
        pass
    
    def parse_ui_annotations(self, example: tf.train.Example) -> List[Dict[str, Any]]:
        """
        Parse UI annotations from AiTW dataset example.
        
        Args:
            example: TensorFlow example from AiTW dataset
            
        Returns:
            List of parsed UI annotations
        """
        try:
            # Extract UI annotation data
            positions = self._extract_positions(example)
            texts = self._extract_texts(example)
            ui_types = self._extract_ui_types(example)
            
            # Combine into annotation objects
            annotations = []
            min_length = min(len(positions), len(texts), len(ui_types))
            
            for i in range(min_length):
                annotation = {
                    'position': positions[i],
                    'text': texts[i],
                    'ui_type': ui_types[i],
                    'index': i
                }
                annotations.append(annotation)
            
            return annotations
        
        except Exception as e:
            logger.error(f"Error parsing UI annotations: {e}")
            return []
    
    def _extract_positions(self, example: tf.train.Example) -> List[List[float]]:
        """Extract normalized position coordinates [y, x, height, width]."""
        try:
            # Get flattened positions array
            flattened_positions = np.array(
                example.features.feature['image/ui_annotations_positions'].float_list.value
            )
            
            # Reshape to (n_elements, 4) format
            if len(flattened_positions) % 4 != 0:
                # Trim to nearest multiple of 4
                flattened_positions = flattened_positions[:len(flattened_positions) // 4 * 4]
            
            positions = flattened_positions.reshape(-1, 4)
            
            # Convert to list of lists
            return positions.tolist()
        
        except Exception as e:
            logger.debug(f"Error extracting positions: {e}")
            return []
    
    def _extract_texts(self, example: tf.train.Example) -> List[str]:
        """Extract UI element text content."""
        try:
            texts = []
            for text_bytes in example.features.feature['image/ui_annotations_text'].bytes_list.value:
                text = text_bytes.decode('utf-8', errors='ignore')
                texts.append(text)
            return texts
        
        except Exception as e:
            logger.debug(f"Error extracting texts: {e}")
            return []
    
    def _extract_ui_types(self, example: tf.train.Example) -> List[str]:
        """Extract UI element types."""
        try:
            ui_types = []
            for type_bytes in example.features.feature['image/ui_annotations_ui_types'].bytes_list.value:
                ui_type = type_bytes.decode('utf-8', errors='ignore')
                ui_types.append(ui_type)
            return ui_types
        
        except Exception as e:
            logger.debug(f"Error extracting UI types: {e}")
            return []
    
    def extract_action_info(self, example: tf.train.Example) -> Dict[str, Any]:
        """Extract action information from the example."""
        try:
            action_info = {}
            
            # Action type
            action_type = example.features.feature['results/action_type'].int64_list.value[0]
            action_info['action_type'] = action_type
            
            # Touch coordinates (if applicable)
            if 'results/yx_touch' in example.features.feature:
                touch_coords = example.features.feature['results/yx_touch'].float_list.value
                if len(touch_coords) >= 2:
                    action_info['touch_coordinates'] = [touch_coords[0], touch_coords[1]]
            
            # Lift coordinates (if applicable)
            if 'results/yx_lift' in example.features.feature:
                lift_coords = example.features.feature['results/yx_lift'].float_list.value
                if len(lift_coords) >= 2:
                    action_info['lift_coordinates'] = [lift_coords[0], lift_coords[1]]
            
            # Type action text (if applicable)
            if 'results/type_action' in example.features.feature:
                type_text_bytes = example.features.feature['results/type_action'].bytes_list.value
                if type_text_bytes:
                    action_info['type_text'] = type_text_bytes[0].decode('utf-8', errors='ignore')
            
            return action_info
        
        except Exception as e:
            logger.debug(f"Error extracting action info: {e}")
            return {}
    
    def extract_episode_info(self, example: tf.train.Example) -> Dict[str, Any]:
        """Extract episode metadata."""
        try:
            episode_info = {}
            
            # Episode ID
            episode_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            episode_info['episode_id'] = episode_id
            
            # Episode length
            episode_length = example.features.feature['episode_length'].int64_list.value[0]
            episode_info['episode_length'] = episode_length
            
            # Step ID
            step_id = example.features.feature['step_id'].int64_list.value[0]
            episode_info['step_id'] = step_id
            
            # Goal information
            goal_info = example.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
            episode_info['goal_info'] = goal_info
            
            # Device and system info
            if 'android_api_level' in example.features.feature:
                episode_info['android_api_level'] = example.features.feature['android_api_level'].int64_list.value[0]
            
            if 'device_type' in example.features.feature:
                device_type = example.features.feature['device_type'].bytes_list.value[0].decode('utf-8')
                episode_info['device_type'] = device_type
            
            if 'current_activity' in example.features.feature:
                current_activity = example.features.feature['current_activity'].bytes_list.value[0].decode('utf-8')
                episode_info['current_activity'] = current_activity
            
            return episode_info
        
        except Exception as e:
            logger.debug(f"Error extracting episode info: {e}")
            return {}
    
    def filter_annotations_by_area(self, annotations: List[Dict[str, Any]], 
                                 min_area: float = 0.001) -> List[Dict[str, Any]]:
        """Filter out annotations that are too small."""
        filtered = []
        for annotation in annotations:
            if 'position' in annotation and len(annotation['position']) == 4:
                y, x, height, width = annotation['position']
                area = height * width
                if area >= min_area:
                    filtered.append(annotation)
        return filtered
    
    def filter_annotations_by_text(self, annotations: List[Dict[str, Any]], 
                                 min_text_length: int = 1) -> List[Dict[str, Any]]:
        """Filter out annotations without meaningful text."""
        filtered = []
        for annotation in annotations:
            text = annotation.get('text', '').strip()
            if len(text) >= min_text_length:
                filtered.append(annotation)
        return filtered
