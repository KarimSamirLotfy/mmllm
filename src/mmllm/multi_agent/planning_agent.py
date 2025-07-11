"""Planning agent for multimodal action planning."""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command

from ..model import get_model
from ..android_in_the_wild.action_type import ActionType
from .state import MultiAgentState, PlanningOutput, AgentPhase

logger = logging.getLogger(__name__)


class PlanningAgent:
    """Vision-based action planning agent."""
    
    def __init__(self):
        self.model = get_model()
    
    def plan_action(self, state: MultiAgentState) -> Command:
        """
        Analyze UI screenshot and plan the next action.
        
        Args:
            state: Current multi-agent state
            
        Returns:
            Command to route to execution agent or end
        """
        try:
            # Extract context from state
            goal = state["goal"]
            current_step = state["current_step"]
            current_image = state["current_image"]
            ui_annotations = state.get("ui_annotations", [])
            action_history = state.get("action_history", [])
            reflection_history = state.get("reflection_history", [])
            
            # Add support for historical images if available
            image_history = state.get("image_history", [])
            episode_images = state.get("episode_images", [])
            
            # Build context message
            context_parts = [f"Goal: {goal}"]
            if action_history:
                context_parts.append(f"Previous actions taken: {len(action_history)}")
                last_actions = action_history[-3:]  # Show last 3 actions
                for i, action in enumerate(last_actions):
                    context_parts.append(f"Action {len(action_history)-len(last_actions)+i+1}: {action.reasoning}")
            
            if reflection_history:
                latest_reflection = reflection_history[-1]
                context_parts.append(f"Latest feedback: {latest_reflection.feedback_for_planning}")
            
            # Add historical context if available
            if episode_images and len(episode_images) > 1:
                context_parts.append(f"Episode context: {len(episode_images)} screens processed")
            elif image_history:
                context_parts.append(f"Recent history: {len(image_history)} previous screens")
            
            context = "\n".join(context_parts)
            
            # Prepare UI analysis prompt
            ui_analysis_text = self._build_ui_analysis_prompt(ui_annotations)
            
            # Create multimodal message
            messages = [
                SystemMessage(content=f"""You are an expert mobile UI automation planning agent. Your job is to analyze Android screenshots and plan the next action to achieve the given goal.

CONTEXT:
{context}

IMPORTANT GUIDELINES:
- Coordinates are normalized [y, x] format in range [0, 1]
- Use ActionType.DUAL_POINT for both taps and swipes
- Use ActionType.TYPE for text input
- Use ActionType.PRESS_BACK, PRESS_HOME, PRESS_ENTER for navigation
- Analyze UI elements carefully before choosing actions
- Consider the goal and previous actions when planning
- Be precise with coordinate selection

AVAILABLE ACTIONS:
1. TapAction - Single touch at coordinates
2. SwipeAction - Gesture from start to end coordinates  
3. TypeAction - Text input
4. NavigationAction - Back/Home/Enter buttons
5. StatusAction - Mark task complete/impossible
"""),
                self._build_multimodal_message(
                    goal, current_step, current_image, ui_analysis_text, image_history
                )
            ]
            
            # Get structured planning output
            planning_output = self._get_structured_planning_decision(messages, goal, current_step)
            
            # Determine next phase based on planning result
            if planning_output.action_type in [ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE]:
                next_phase = AgentPhase.COMPLETE
                goto = "END"
            else:
                next_phase = AgentPhase.EXECUTION
                goto = "execution_agent"
            
            # Update state
            new_state = {
                "current_phase": next_phase,
                "planning_output": planning_output,
                "messages": state["messages"] + [
                    AIMessage(content=f"Planned action: {planning_output.reasoning}")
                ]
            }
            
            return Command(goto=goto, update=new_state)
            
        except Exception as e:
            # Handle planning errors
            error_msg = f"Planning error: {str(e)}"
            planning_output = PlanningOutput(
                action_type=ActionType.STATUS_TASK_IMPOSSIBLE,
                reasoning=f"Planning failed due to error: {str(e)}",
                confidence=0.0
            )
            
            new_state = {
                "current_phase": AgentPhase.FAILED,
                "planning_output": planning_output,
                "error_count": state.get("error_count", 0) + 1,
                "last_error": error_msg
            }
            
            return Command(goto="END", update=new_state)

    def _build_multimodal_message(
        self, 
        goal: str, 
        current_step: int, 
        current_image: str, 
        ui_analysis_text: str,
        image_history: Optional[List[str]] = None
    ) -> HumanMessage:
        """Build multimodal message with optional historical context."""
        
        text_content = f"Please analyze this Android interface and plan the next action to achieve the goal.\n\n{ui_analysis_text}\n\nStep {current_step}: What action should be taken next?"
        
        # Start with text and current image
        content = [
            {
                "type": "text",
                "text": text_content
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{current_image}"
                }
            }
        ]
        
        # Add historical images if available (limit to last 2 for memory)
        if image_history:
            recent_history = image_history[-2:]  # Last 2 images
            for i, hist_image in enumerate(recent_history):
                if hist_image:  # Skip empty images
                    content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{hist_image}"
                        }
                    })
        
        return HumanMessage(content=content)
    
    def _build_ui_analysis_prompt(self, ui_annotations: List[Dict[str, Any]]) -> str:
        """Build text description of UI elements."""
        if not ui_annotations:
            return "No UI element annotations available. Please analyze the visual interface carefully."
        
        ui_text = "Detected UI elements:\n"
        for i, element in enumerate(ui_annotations[:10]):  # Limit to first 10 elements
            ui_text += f"- Element {i+1}: {element.get('description', 'Unknown element')}"
            if 'bounds' in element:
                bounds = element['bounds']
                ui_text += f" at coordinates [{bounds.get('y', 0):.3f}, {bounds.get('x', 0):.3f}]"
            ui_text += "\n"
        
        return ui_text
    
    def _get_structured_planning_decision(self, messages: List, goal: str, current_step: int) -> PlanningOutput:
        """Get structured planning decision from the model."""
        try:
            # Use structured output to get the planning decision
            structured_model = self.model.with_structured_output(PlanningOutput)
            result = structured_model.invoke(messages)
            return result
        except Exception as e:
            # Fallback to manual parsing if structured output fails
            logger.debug(f"Structured output failed, using fallback: {e}")
            response = self.model.invoke(messages)
            return self._parse_model_response(response.content, goal, current_step)
    
    def _parse_model_response(self, response_text: str, goal: str, current_step: int) -> PlanningOutput:
        """Parse model response text into planning output."""
        try:
            # Simple parsing logic - look for key indicators in response
            response_lower = response_text.lower()
            
            # Determine action type based on response content
            if "tap" in response_lower or "click" in response_lower:
                action_type = ActionType.DUAL_POINT
                # Try to extract coordinates (basic pattern matching)
                coordinates = self._extract_coordinates(response_text)
                if not coordinates:
                    coordinates = [0.5, 0.5]  # Default center tap
                
                return PlanningOutput(
                    action_type=action_type,
                    coordinates=coordinates,
                    reasoning=f"Planning to tap based on analysis: {response_text[:200]}...",
                    confidence=0.7
                )
            
            elif "swipe" in response_lower or "scroll" in response_lower:
                action_type = ActionType.DUAL_POINT
                start_coords = [0.5, 0.3]  # Default swipe start
                end_coords = [0.5, 0.7]    # Default swipe end
                
                return PlanningOutput(
                    action_type=action_type,
                    coordinates=start_coords,
                    lift_coordinates=end_coords,
                    reasoning=f"Planning to swipe based on analysis: {response_text[:200]}...",
                    confidence=0.6
                )
            
            elif "type" in response_lower or "input" in response_lower or "enter text" in response_lower:
                action_type = ActionType.TYPE
                # Extract text to type if mentioned
                text_to_type = self._extract_text_to_type(response_text)
                
                return PlanningOutput(
                    action_type=action_type,
                    text_input=text_to_type,
                    reasoning=f"Planning to type text based on analysis: {response_text[:200]}...",
                    confidence=0.7
                )
            
            elif "back" in response_lower:
                action_type = ActionType.PRESS_BACK
                return PlanningOutput(
                    action_type=action_type,
                    reasoning=f"Planning to press back button: {response_text[:200]}...",
                    confidence=0.8
                )
            
            elif "home" in response_lower:
                action_type = ActionType.PRESS_HOME
                return PlanningOutput(
                    action_type=action_type,
                    reasoning=f"Planning to press home button: {response_text[:200]}...",
                    confidence=0.8
                )
            
            elif "complete" in response_lower or "done" in response_lower or "finished" in response_lower:
                action_type = ActionType.STATUS_TASK_COMPLETE
                return PlanningOutput(
                    action_type=action_type,
                    reasoning=f"Task appears to be complete: {response_text[:200]}...",
                    confidence=0.9
                )
            
            elif "impossible" in response_lower or "cannot" in response_lower or "unable" in response_lower:
                action_type = ActionType.STATUS_TASK_IMPOSSIBLE
                return PlanningOutput(
                    action_type=action_type,
                    reasoning=f"Task appears impossible: {response_text[:200]}...",
                    confidence=0.8
                )
            
            else:
                # Default to center tap if unclear
                return PlanningOutput(
                    action_type=ActionType.DUAL_POINT,
                    coordinates=[0.5, 0.5],
                    reasoning=f"Default action - center tap: {response_text[:200]}...",
                    confidence=0.4
                )
                
        except Exception as e:
            # Fallback planning
            return PlanningOutput(
                action_type=ActionType.STATUS_TASK_IMPOSSIBLE,
                reasoning=f"Failed to parse model response: {str(e)}",
                confidence=0.1
            )
    
    def _extract_coordinates(self, text: str) -> Optional[List[float]]:
        """Extract coordinates from text using basic pattern matching."""
        import re
        
        # Look for patterns like [0.5, 0.3] or (0.5, 0.3) or 0.5,0.3
        patterns = [
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',
            r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)',
            r'(\d+\.?\d*),\s*(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    y, x = float(match.group(1)), float(match.group(2))
                    # Ensure coordinates are in valid range
                    y = max(0.0, min(1.0, y))
                    x = max(0.0, min(1.0, x))
                    return [y, x]
                except Exception:
                    continue
        
        return None
    
    def _extract_text_to_type(self, text: str) -> str:
        """Extract text that should be typed from the response."""
        # Look for quoted text or text after "type"
        import re
        
        # Look for quoted strings
        quote_match = re.search(r'"([^"]*)"', text)
        if quote_match:
            return quote_match.group(1)
        
        # Look for text after "type"
        type_match = re.search(r'type\s+([^\n.!?]+)', text, re.IGNORECASE)
        if type_match:
            return type_match.group(1).strip()
        
        # Default
        return "sample text"


def planning_node(state: MultiAgentState) -> Command:
    """Planning agent node for LangGraph."""
    agent = PlanningAgent()
    return agent.plan_action(state)
