"""Action comparison utilities for evaluating agent outputs against AiTW ground truth."""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..android_in_the_wild.action_type import ActionType
from ..multi_agent.state import StepEvaluationResult

logger = logging.getLogger(__name__)


class ActionComparator:
    """Compare agent actions with AiTW ground truth."""
    
    def __init__(self):
        self.tap_distance_threshold = 0.14  # From AiTW paper
        self.swipe_distance_threshold = 0.04
    
    def compare_actions(
        self, 
        agent_action: Dict[str, Any], 
        ground_truth_action: Dict[str, Any],
        step_number: int = 0
    ) -> StepEvaluationResult:
        """
        Compare agent action with ground truth action.
        
        Args:
            agent_action: Action predicted by agent in AiTW format
            ground_truth_action: Ground truth action from dataset
            step_number: Step number in the episode
            
        Returns:
            StepEvaluationResult with detailed comparison metrics
        """
        try:
            # Extract action types
            agent_action_type = self._extract_action_type(agent_action)
            gt_action_type = self._extract_action_type(ground_truth_action)
            
            # Check if action types match
            action_type_match = agent_action_type == gt_action_type
            
            # Initialize evaluation result
            result = StepEvaluationResult(
                step_number=step_number,
                agent_action=agent_action,
                ground_truth_action=ground_truth_action,
                action_match=False,
                action_type_match=action_type_match,
                evaluation_score=0.0
            )
            
            # If action types don't match, return early with low score
            if not action_type_match:
                result.evaluation_score = 0.1  # Small score for attempting an action
                return result
            
            # Compare based on action type
            if agent_action_type == ActionType.DUAL_POINT:
                result = self._compare_dual_point_actions(agent_action, ground_truth_action, result)
            elif agent_action_type == ActionType.TYPE:
                result = self._compare_type_actions(agent_action, ground_truth_action, result)
            elif agent_action_type in [ActionType.PRESS_BACK, ActionType.PRESS_HOME, ActionType.PRESS_ENTER]:
                result = self._compare_navigation_actions(agent_action, ground_truth_action, result)
            elif agent_action_type in [ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE]:
                result = self._compare_status_actions(agent_action, ground_truth_action, result)
            else:
                logger.warning(f"Unknown action type for comparison: {agent_action_type}")
                result.evaluation_score = 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing actions: {e}")
            return StepEvaluationResult(
                step_number=step_number,
                agent_action=agent_action,
                ground_truth_action=ground_truth_action,
                action_match=False,
                action_type_match=False,
                evaluation_score=0.0
            )
    
    def _extract_action_type(self, action: Dict[str, Any]) -> ActionType:
        """Extract ActionType from action dictionary."""
        if "action_type" in action:
            if isinstance(action["action_type"], ActionType):
                return action["action_type"]
            elif isinstance(action["action_type"], int):
                return ActionType(action["action_type"])
            elif isinstance(action["action_type"], str):
                # Handle string representations
                type_mapping = {
                    "DUAL_POINT": ActionType.DUAL_POINT,
                    "TYPE": ActionType.TYPE,
                    "PRESS_BACK": ActionType.PRESS_BACK,
                    "PRESS_HOME": ActionType.PRESS_HOME,
                    "PRESS_ENTER": ActionType.PRESS_ENTER,
                    "STATUS_TASK_COMPLETE": ActionType.STATUS_TASK_COMPLETE,
                    "STATUS_TASK_IMPOSSIBLE": ActionType.STATUS_TASK_IMPOSSIBLE
                }
                return type_mapping.get(action["action_type"], ActionType.DUAL_POINT)
        
        # Fallback: try to infer from action structure
        if "coordinates" in action or "touch_point" in action:
            return ActionType.DUAL_POINT
        elif "text" in action or "text_input" in action:
            return ActionType.TYPE
        else:
            return ActionType.DUAL_POINT  # Default assumption
    
    def _compare_dual_point_actions(
        self, 
        agent_action: Dict[str, Any], 
        ground_truth_action: Dict[str, Any],
        result: StepEvaluationResult
    ) -> StepEvaluationResult:
        """Compare DUAL_POINT actions (taps and swipes)."""
        try:
            # Extract coordinates
            agent_coords = self._extract_coordinates(agent_action)
            gt_coords = self._extract_coordinates(ground_truth_action)
            
            if agent_coords is None or gt_coords is None:
                result.evaluation_score = 0.2  # Some score for matching type
                return result
            
            # Calculate coordinate distance
            agent_start = np.array(agent_coords[:2])  # [y, x]
            gt_start = np.array(gt_coords[:2])
            
            distance = np.linalg.norm(agent_start - gt_start)
            result.coordinate_distance = float(distance)
            
            # Check if coordinates are within threshold
            if distance <= self.tap_distance_threshold:
                result.action_match = True
                result.evaluation_score = 1.0
            else:
                # Partial score based on proximity
                max_distance = 1.0  # Maximum possible distance on normalized screen
                proximity_score = max(0, 1 - (distance / max_distance))
                result.evaluation_score = 0.3 + 0.5 * proximity_score  # Base score + proximity bonus
            
            # For swipes, also compare lift coordinates if available
            if len(agent_coords) > 2 and len(gt_coords) > 2:
                agent_end = np.array(agent_coords[2:4])
                gt_end = np.array(gt_coords[2:4])
                end_distance = np.linalg.norm(agent_end - gt_end)
                
                if end_distance <= self.tap_distance_threshold:
                    result.evaluation_score = min(1.0, result.evaluation_score + 0.2)
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing dual point actions: {e}")
            result.evaluation_score = 0.2
            return result
    
    def _compare_type_actions(
        self, 
        agent_action: Dict[str, Any], 
        ground_truth_action: Dict[str, Any],
        result: StepEvaluationResult
    ) -> StepEvaluationResult:
        """Compare TYPE actions (text input)."""
        try:
            # Extract text content
            agent_text = self._extract_text(agent_action)
            gt_text = self._extract_text(ground_truth_action)
            
            if agent_text is None or gt_text is None:
                result.evaluation_score = 0.3  # Some score for matching type
                return result
            
            # Normalize text for comparison
            agent_text_norm = agent_text.lower().strip()
            gt_text_norm = gt_text.lower().strip()
            
            # Check exact match
            if agent_text_norm == gt_text_norm:
                result.action_match = True
                result.text_match = True
                result.evaluation_score = 1.0
            else:
                # Calculate similarity score
                similarity = self._calculate_text_similarity(agent_text_norm, gt_text_norm)
                result.text_match = similarity > 0.8
                result.evaluation_score = 0.3 + 0.7 * similarity
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing type actions: {e}")
            result.evaluation_score = 0.3
            return result
    
    def _compare_navigation_actions(
        self, 
        agent_action: Dict[str, Any], 
        ground_truth_action: Dict[str, Any],
        result: StepEvaluationResult
    ) -> StepEvaluationResult:
        """Compare navigation actions (back, home, enter)."""
        # For navigation actions, if types match, it's a perfect match
        result.action_match = True
        result.evaluation_score = 1.0
        return result
    
    def _compare_status_actions(
        self, 
        agent_action: Dict[str, Any], 
        ground_truth_action: Dict[str, Any],
        result: StepEvaluationResult
    ) -> StepEvaluationResult:
        """Compare status actions (complete, impossible)."""
        # For status actions, if types match, it's a perfect match
        result.action_match = True
        result.evaluation_score = 1.0
        return result
    
    def _extract_coordinates(self, action: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
        """Extract coordinates from action dictionary."""
        # Try different coordinate field names
        coordinate_fields = ["coordinates", "touch_point", "start_point", "point"]
        
        for field in coordinate_fields:
            if field in action and action[field] is not None:
                coords = action[field]
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    # Ensure coordinates are in [y, x] format and normalized
                    result = [float(coords[0]), float(coords[1])]
                    
                    # Add lift coordinates if available
                    if "lift_coordinates" in action and action["lift_coordinates"]:
                        lift = action["lift_coordinates"]
                        if isinstance(lift, (list, tuple)) and len(lift) >= 2:
                            result.extend([float(lift[0]), float(lift[1])])
                    elif "end_point" in action and action["end_point"]:
                        end = action["end_point"]
                        if isinstance(end, (list, tuple)) and len(end) >= 2:
                            result.extend([float(end[0]), float(end[1])])
                    
                    return tuple(result)
        
        return None
    
    def _extract_text(self, action: Dict[str, Any]) -> Optional[str]:
        """Extract text content from action dictionary."""
        text_fields = ["text", "text_input", "input_text", "content"]
        
        for field in text_fields:
            if field in action and action[field] is not None:
                return str(action[field])
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character-level matching."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def format_agent_action_for_aitw(self, planning_output) -> Dict[str, Any]:
        """Convert PlanningOutput to AiTW action format."""
        action = {
            "action_type": planning_output.action_type.value,
        }
        
        if planning_output.coordinates:
            action["coordinates"] = planning_output.coordinates
        
        if planning_output.lift_coordinates:
            action["lift_coordinates"] = planning_output.lift_coordinates
        
        if planning_output.text_input:
            action["text"] = planning_output.text_input
        
        return action
