"""UI element detection and analysis."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..android_in_the_wild.action_type import ActionType


class UIAnalyzer:
    """Analyzes UI elements and their properties for action planning."""
    
    def __init__(self):
        self.clickable_ui_types = {
            'button', 'icon', 'text', 'image', 'checkbox', 'switch', 
            'spinner', 'seekbar', 'ratingbar', 'togglebutton'
        }
        self.text_input_types = {
            'edittext', 'textview', 'autocompletetextview', 'multiautocompletetextview'
        }
    
    def analyze_ui_elements(self, ui_annotations: List[Dict[str, Any]], 
                          image_height: int, image_width: int) -> List[Dict[str, Any]]:
        """
        Analyze UI elements from AiTW annotations.
        
        Args:
            ui_annotations: Raw UI annotations from AiTW dataset
            image_height: Height of the screen image
            image_width: Width of the screen image
            
        Returns:
            List of analyzed UI elements with enhanced properties
        """
        analyzed_elements = []
        
        for i, annotation in enumerate(ui_annotations):
            element = self._process_single_element(annotation, i, image_height, image_width)
            if element:
                analyzed_elements.append(element)
        
        return analyzed_elements
    
    def _process_single_element(self, annotation: Dict[str, Any], element_id: int,
                              image_height: int, image_width: int) -> Optional[Dict[str, Any]]:
        """Process a single UI element annotation."""
        try:
            # Extract position information (normalized coordinates)
            if 'position' not in annotation:
                return None
            
            position = annotation['position']
            if len(position) != 4:
                return None
            
            y, x, height, width = position
            
            # Convert to pixel coordinates for analysis
            pixel_y = int(y * image_height)
            pixel_x = int(x * image_width)
            pixel_height = int(height * image_height)
            pixel_width = int(width * image_width)
            
            # Extract text and UI type
            text = annotation.get('text', '').strip()
            ui_type = annotation.get('ui_type', '').lower()
            
            # Determine element capabilities
            capabilities = self._determine_capabilities(ui_type, text)
            
            # Calculate center point for actions
            center_y = y + height / 2
            center_x = x + width / 2
            
            return {
                'id': element_id,
                'position': {
                    'normalized': {'y': y, 'x': x, 'height': height, 'width': width},
                    'pixel': {'y': pixel_y, 'x': pixel_x, 'height': pixel_height, 'width': pixel_width}
                },
                'center': {'y': center_y, 'x': center_x},
                'text': text,
                'ui_type': ui_type,
                'capabilities': capabilities,
                'area': height * width,
                'aspect_ratio': width / height if height > 0 else 0
            }
        
        except Exception as e:
            # Skip malformed annotations
            return None
    
    def _determine_capabilities(self, ui_type: str, text: str) -> Dict[str, bool]:
        """Determine what actions are possible on this UI element."""
        capabilities = {
            'clickable': False,
            'text_input': False,
            'scrollable': False,
            'swipeable': False
        }
        
        # Check if element is clickable
        if (ui_type in self.clickable_ui_types or 
            any(keyword in text.lower() for keyword in ['button', 'click', 'tap', 'select'])):
            capabilities['clickable'] = True
        
        # Check if element accepts text input
        if (ui_type in self.text_input_types or
            any(keyword in text.lower() for keyword in ['input', 'search', 'enter', 'type'])):
            capabilities['text_input'] = True
        
        # Check if element is scrollable/swipeable
        if (ui_type in ['listview', 'recyclerview', 'scrollview', 'viewpager'] or
            any(keyword in text.lower() for keyword in ['scroll', 'list', 'swipe'])):
            capabilities['scrollable'] = True
            capabilities['swipeable'] = True
        
        return capabilities
    
    def find_elements_by_capability(self, elements: List[Dict[str, Any]], 
                                  capability: str) -> List[Dict[str, Any]]:
        """Find elements that have a specific capability."""
        return [elem for elem in elements if elem['capabilities'].get(capability, False)]
    
    def find_elements_by_text(self, elements: List[Dict[str, Any]], 
                            search_text: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Find elements containing specific text."""
        search_text = search_text.lower()
        matching_elements = []
        
        for element in elements:
            element_text = element['text'].lower()
            
            if fuzzy:
                # Fuzzy matching - check if any word in search_text is in element_text
                if any(word in element_text for word in search_text.split()):
                    matching_elements.append(element)
            else:
                # Exact matching
                if search_text in element_text:
                    matching_elements.append(element)
        
        return matching_elements
    
    def suggest_action_targets(self, elements: List[Dict[str, Any]], 
                             goal: str) -> List[Dict[str, Any]]:
        """Suggest UI elements that might be relevant for achieving the goal."""
        goal_lower = goal.lower()
        relevant_elements = []
        
        # Look for elements with text related to the goal
        for element in elements:
            relevance_score = self._calculate_relevance_score(element, goal_lower)
            if relevance_score > 0:
                element_with_score = element.copy()
                element_with_score['relevance_score'] = relevance_score
                relevant_elements.append(element_with_score)
        
        # Sort by relevance score
        relevant_elements.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_elements[:5]  # Return top 5 most relevant
    
    def _calculate_relevance_score(self, element: Dict[str, Any], goal: str) -> float:
        """Calculate how relevant an element is to the goal."""
        score = 0.0
        element_text = element['text'].lower()
        
        # Direct text matches
        goal_words = goal.split()
        for word in goal_words:
            if word in element_text:
                score += 1.0
        
        # Capability bonuses
        if 'click' in goal or 'tap' in goal:
            if element['capabilities']['clickable']:
                score += 0.5
        
        if 'type' in goal or 'enter' in goal or 'input' in goal:
            if element['capabilities']['text_input']:
                score += 0.5
        
        if 'scroll' in goal or 'swipe' in goal:
            if element['capabilities']['scrollable']:
                score += 0.5
        
        # Size bonus (larger elements are often more important)
        if element['area'] > 0.01:  # Larger than 1% of screen
            score += 0.2
        
        return score
    
    def get_recommended_action_type(self, element: Dict[str, Any], goal: str) -> ActionType:
        """Recommend the best action type for an element given the goal."""
        goal_lower = goal.lower()
        capabilities = element['capabilities']
        
        # Text input actions
        if ('type' in goal_lower or 'enter' in goal_lower or 'input' in goal_lower):
            if capabilities['text_input']:
                return ActionType.TYPE
        
        # Scrolling actions
        if ('scroll' in goal_lower or 'swipe' in goal_lower):
            if capabilities['scrollable']:
                return ActionType.DUAL_POINT  # Will be interpreted as swipe
        
        # Default to tap for clickable elements
        if capabilities['clickable']:
            return ActionType.DUAL_POINT  # Will be interpreted as tap
        
        # Fallback to tap
        return ActionType.DUAL_POINT
