"""Action executor for Android interface interactions."""

from typing import Dict, Any, Optional
import logging
from .action_schemas import TapAction, SwipeAction, TypeAction, NavigationAction, StatusAction
from ..android_in_the_wild.action_type import ActionType
from ..multi_agent.state import PlanningOutput


class ActionExecutor:
    """Executes actions on Android interfaces (simulation for AiTW dataset)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_count = 0
    
    def execute_action(self, planning_output: PlanningOutput) -> Dict[str, Any]:
        """
        Execute a planning output and return execution results.
        
        Args:
            planning_output: The planning output to execute
            
        Returns:
            Dictionary with execution results
        """
        self.execution_count += 1
        action_type = planning_output.action_type
        
        try:
            if action_type == ActionType.DUAL_POINT:
                return self._execute_dual_point(planning_output)
            elif action_type == ActionType.TYPE:
                return self._execute_type(planning_output)
            elif action_type in [ActionType.PRESS_BACK, ActionType.PRESS_HOME, ActionType.PRESS_ENTER]:
                return self._execute_navigation(planning_output)
            elif action_type in [ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE]:
                return self._execute_status(planning_output)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                    "execution_id": self.execution_count
                }
        
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_id": self.execution_count
            }
    
    def _execute_dual_point(self, planning_output: PlanningOutput) -> Dict[str, Any]:
        """Execute dual point action (tap or swipe)."""
        if not planning_output.coordinates:
            return {"success": False, "error": "No coordinates provided"}
        
        y, x = planning_output.coordinates
        
        if planning_output.lift_coordinates:
            # This is a swipe
            lift_y, lift_x = planning_output.lift_coordinates
            self.logger.info(f"Executing SWIPE from ({y:.3f}, {x:.3f}) to ({lift_y:.3f}, {lift_x:.3f})")
            action_type = "swipe"
            extra_data = {"end_coordinates": [lift_y, lift_x]}
        else:
            # This is a tap
            self.logger.info(f"Executing TAP at coordinates ({y:.3f}, {x:.3f})")
            action_type = "tap"
            extra_data = {}
        
        return {
            "success": True,
            "action_type": action_type,
            "coordinates": [y, x],
            "reasoning": planning_output.reasoning,
            "confidence": planning_output.confidence,
            "execution_id": self.execution_count,
            "simulation_note": "Simulated execution for AiTW dataset",
            **extra_data
        }
    
    def _execute_type(self, planning_output: PlanningOutput) -> Dict[str, Any]:
        """Execute type action."""
        text = planning_output.text_input or ""
        self.logger.info(f"Executing TYPE: '{text}'")
        
        return {
            "success": True,
            "action_type": "type",
            "text": text,
            "reasoning": planning_output.reasoning,
            "confidence": planning_output.confidence,
            "execution_id": self.execution_count,
            "simulation_note": "Simulated execution for AiTW dataset"
        }
    
    def _execute_navigation(self, planning_output: PlanningOutput) -> Dict[str, Any]:
        """Execute navigation action."""
        action_name = planning_output.action_type.name
        self.logger.info(f"Executing NAVIGATION: {action_name}")
        
        return {
            "success": True,
            "action_type": "navigation",
            "navigation_action": action_name,
            "reasoning": planning_output.reasoning,
            "confidence": planning_output.confidence,
            "execution_id": self.execution_count,
            "simulation_note": "Simulated execution for AiTW dataset"
        }
    
    def _execute_status(self, planning_output: PlanningOutput) -> Dict[str, Any]:
        """Execute status action."""
        action_name = planning_output.action_type.name
        self.logger.info(f"Executing STATUS: {action_name}")
        
        return {
            "success": True,
            "action_type": "status",
            "status_action": action_name,
            "reasoning": planning_output.reasoning,
            "confidence": planning_output.confidence,
            "execution_id": self.execution_count,
            "final_action": True  # Status actions typically end episodes
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.execution_count,
            "executor_status": "active"
        }
