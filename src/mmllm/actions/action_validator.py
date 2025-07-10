"""Action validator for feasibility checks."""

from typing import List, Dict, Any, Tuple
import numpy as np
from .action_schemas import TapAction, SwipeAction, TypeAction
from ..android_in_the_wild.action_type import ActionType
from ..multi_agent.state import PlanningOutput


class ActionValidator:
    """Validates action feasibility based on UI state and constraints."""
    
    def __init__(self, tap_distance_threshold: float = 0.14):
        self.tap_distance_threshold = tap_distance_threshold
    
    def validate_action_plan(self, planning_output: PlanningOutput, ui_annotations: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate if a planning output is feasible given current UI state.
        
        Args:
            planning_output: The planning output to validate
            ui_annotations: UI annotations from the current screen
            
        Returns:
            Tuple of (is_valid, reason)
        """
        action_type = planning_output.action_type
        
        if action_type == ActionType.DUAL_POINT:
            return self._validate_dual_point_action(planning_output, ui_annotations)
        elif action_type == ActionType.TYPE:
            return self._validate_type_action(planning_output, ui_annotations)
        else:
            # Navigation and status actions are generally valid
            return True, "Action is valid"
    
    def _validate_dual_point_action(self, planning_output: PlanningOutput, ui_annotations: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Validate dual point action (tap or swipe)."""
        if not planning_output.coordinates:
            return False, "No coordinates provided for dual point action"
        
        y, x = planning_output.coordinates
        
        # Check bounds
        if not (0.0 <= y <= 1.0 and 0.0 <= x <= 1.0):
            return False, "Coordinates outside screen bounds"
        
        # If lift coordinates exist, validate as swipe
        if planning_output.lift_coordinates:
            lift_y, lift_x = planning_output.lift_coordinates
            if not (0.0 <= lift_y <= 1.0 and 0.0 <= lift_x <= 1.0):
                return False, "Lift coordinates outside screen bounds"
            
            # Check swipe distance
            distance = np.sqrt((lift_y - y)**2 + (lift_x - x)**2)
            if distance < 0.04:
                return False, "Swipe distance too short"
            if distance > 0.8:
                return False, "Swipe distance too long"
        
        return True, "Dual point action is valid"
    
    def _validate_type_action(self, planning_output: PlanningOutput, ui_annotations: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Validate type action."""
        if not planning_output.text_input:
            return False, "No text provided for type action"
        
        # Look for text input elements
        text_elements = [
            ann for ann in ui_annotations 
            if ann.get('ui_type', '').lower() in ['text', 'input', 'textbox', 'edittext']
        ]
        
        if not text_elements:
            return False, "No text input elements found on screen"
        
        return True, "Type action is valid"
    
    def _is_tap_near_element(self, tap_y: float, tap_x: float, ui_annotation: Dict[str, Any]) -> bool:
        """Check if tap is near a UI element."""
        # UI annotations format: [y, x, height, width] normalized
        if 'position' not in ui_annotation:
            return False
        
        pos = ui_annotation['position']
        if len(pos) != 4:
            return False
        
        element_y, element_x, element_h, element_w = pos
        
        # Check if tap is within element bounds (with some tolerance)
        tolerance = 0.05  # 5% tolerance
        
        within_x = (element_x - tolerance) <= tap_x <= (element_x + element_w + tolerance)
        within_y = (element_y - tolerance) <= tap_y <= (element_y + element_h + tolerance)
        
        return within_x and within_y
    
    def suggest_improvements(self, planning_output: PlanningOutput, ui_annotations: List[Dict[str, Any]]) -> List[str]:
        """Suggest improvements for a planning output."""
        suggestions = []
        
        if planning_output.action_type == ActionType.DUAL_POINT and planning_output.coordinates:
            # Find closest UI element
            closest_element = self._find_closest_ui_element(planning_output.coordinates, ui_annotations)
            if closest_element:
                suggestions.append(f"Consider targeting UI element: {closest_element.get('text', 'Unknown')}")
        
        if planning_output.confidence < 0.5:
            suggestions.append("Consider gathering more visual context before acting")
        
        return suggestions
    
    def _find_closest_ui_element(self, coordinates: List[float], ui_annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the closest UI element to given coordinates."""
        tap_y, tap_x = coordinates
        closest_element = None
        min_distance = float('inf')
        
        for annotation in ui_annotations:
            if 'position' not in annotation or len(annotation['position']) != 4:
                continue
            
            element_y, element_x, element_h, element_w = annotation['position']
            # Calculate distance to element center
            center_y = element_y + element_h / 2
            center_x = element_x + element_w / 2
            
            distance = np.sqrt((tap_y - center_y)**2 + (tap_x - center_x)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_element = annotation
        
        return closest_element
