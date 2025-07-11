"""Enhanced planning agent with historical context support."""

import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command

from .planning_agent import PlanningAgent
from .state import EpisodeEvaluationState, PlanningOutput, AgentPhase
from ..android_in_the_wild.action_type import ActionType

logger = logging.getLogger(__name__)


class EnhancedPlanningAgent(PlanningAgent):
    """Enhanced planning agent with historical context processing."""
    
    def __init__(self):
        super().__init__()
        self.max_historical_images = 3  # Limit for memory management
    
    def plan_action_with_history(self, state: EpisodeEvaluationState) -> Command:
        """
        Analyze UI screenshot with historical context and plan the next action.
        
        Args:
            state: Episode evaluation state with historical context
            
        Returns:
            Command to route to execution agent or end
        """
        try:
            # Extract context from state
            goal = state["goal"]
            current_step = state["current_step"]
            current_image = state["current_image"]
            ui_annotations = state.get("ui_annotations", [])
            episode_images = state.get("episode_images", [])
            action_history = state.get("action_history", [])
            reflection_history = state.get("reflection_history", [])
            
            # Build enhanced context message with historical information
            context_parts = [f"Goal: {goal}"]
            context_parts.append(f"Current step: {current_step + 1}")
            
            # Add historical context
            if len(episode_images) > 1:
                context_parts.append(f"Episode history: {len(episode_images)} steps completed")
                context_parts.append("Previous screens have been provided for context.")
            
            if action_history:
                context_parts.append(f"Previous actions taken: {len(action_history)}")
                # Show last few actions for context
                last_actions = action_history[-2:]  # Last 2 actions
                for i, action in enumerate(last_actions):
                    step_num = len(action_history) - len(last_actions) + i + 1
                    context_parts.append(f"Step {step_num}: {action.reasoning}")
            
            if reflection_history:
                latest_reflection = reflection_history[-1]
                context_parts.append(f"Latest feedback: {latest_reflection.feedback_for_planning}")
            
            context = "\n".join(context_parts)
            
            # Prepare UI analysis prompt
            ui_analysis_text = self._build_ui_analysis_prompt(ui_annotations)
            
            # Build historical context prompt
            historical_context_prompt = self._build_historical_context_prompt(episode_images, current_step)
            
            # Create enhanced multimodal message with historical images
            messages = [
                SystemMessage(content=f"""You are an expert mobile UI automation planning agent with access to historical context. Your job is to analyze Android screenshots and plan the next action to achieve the given goal.

CONTEXT:
{context}

{historical_context_prompt}

IMPORTANT GUIDELINES:
- Coordinates are normalized [y, x] format in range [0, 1]
- Use ActionType.DUAL_POINT for both taps and swipes
- Use ActionType.TYPE for text input
- Use ActionType.PRESS_BACK, PRESS_HOME, PRESS_ENTER for navigation
- Analyze UI elements carefully before choosing actions
- Consider the goal, previous actions, and historical screens when planning
- Be precise with coordinate selection
- Learn from the progression shown in previous screens

AVAILABLE ACTIONS:
1. TapAction - Single touch at coordinates
2. SwipeAction - Gesture from start to end coordinates  
3. TypeAction - Text input
4. NavigationAction - Back/Home/Enter buttons
5. StatusAction - Mark task complete/impossible
"""),
                self._build_multimodal_message_with_history(
                    goal, current_step, current_image, episode_images, ui_analysis_text
                )
            ]
            
            # Get enhanced structured planning output
            planning_output = self._get_enhanced_planning_decision(messages, state)
            
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
                    AIMessage(content=f"Planned action with historical context: {planning_output.reasoning}")
                ]
            }
            
            return Command(goto=goto, update=new_state)
            
        except Exception as e:
            # Handle planning errors
            error_msg = f"Enhanced planning error: {str(e)}"
            logger.error(error_msg)
            
            planning_output = PlanningOutput(
                action_type=ActionType.STATUS_TASK_IMPOSSIBLE,
                reasoning=f"Enhanced planning failed due to error: {str(e)}",
                confidence=0.0,
                historical_context_used=["error_occurred"]
            )
            
            new_state = {
                "current_phase": AgentPhase.FAILED,
                "planning_output": planning_output,
                "error_count": state.get("error_count", 0) + 1,
                "last_error": error_msg,
                "messages": state["messages"] + [AIMessage(content=error_msg)]
            }
            
            return Command(goto="END", update=new_state)
    
    def _build_historical_context_prompt(self, episode_images: List[str], current_step: int) -> str:
        """Build prompt text describing the historical context."""
        if len(episode_images) <= 1:
            return "HISTORICAL CONTEXT: This is the first step of the episode."
        
        historical_steps = len(episode_images) - 1  # Exclude current image
        
        prompt = f"""HISTORICAL CONTEXT:
You have access to {historical_steps} previous screen(s) from this episode, showing the progression of the interface.
The images are ordered chronologically, with the most recent screens first.
Use this historical context to:
1. Understand what actions have been taken previously
2. See how the interface has changed over time
3. Make more informed decisions about the next action
4. Avoid repeating unsuccessful patterns
5. Build upon previous progress toward the goal

The current screen (step {current_step + 1}) shows the latest state after all previous actions."""
        
        return prompt
    
    def _build_multimodal_message_with_history(
        self, 
        goal: str, 
        current_step: int, 
        current_image: str,
        episode_images: List[str],
        ui_analysis_text: str
    ) -> HumanMessage:
        """Build multimodal message including historical images."""
        
        # Start with text content
        text_content = f"""Please analyze this Android interface sequence and plan the next action to achieve the goal.

GOAL: {goal}

{ui_analysis_text}

CURRENT STEP: {current_step + 1}

The images show the progression of screens in chronological order:
"""
        
        # Add description of image sequence
        if len(episode_images) > 1:
            for i, _ in enumerate(episode_images[:-1]):  # All except current
                text_content += f"- Image {i+1}: Previous screen (step {i+1})\n"
            text_content += f"- Image {len(episode_images)}: Current screen (step {current_step + 1}) - PLAN ACTION FOR THIS SCREEN\n"
        else:
            text_content += f"- Image 1: Current screen (step {current_step + 1}) - PLAN ACTION FOR THIS SCREEN\n"
        
        text_content += "\nWhat action should be taken next to progress toward the goal?"
        
        # Build content list with text and images
        content = [
            {
                "type": "text",
                "text": text_content
            }
        ]
        
        # Add historical images (limit for memory management)
        images_to_include = episode_images[-self.max_historical_images:]
        
        for i, image in enumerate(images_to_include):
            if image:  # Skip empty images
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                })
        
        return HumanMessage(content=content)
    
    def _get_enhanced_planning_decision(
        self, 
        messages: List, 
        state: EpisodeEvaluationState
    ) -> PlanningOutput:
        """Get enhanced structured planning decision from the model."""
        try:
            # Use structured output to get the planning decision
            structured_model = self.model.with_structured_output(PlanningOutput)
            result = structured_model.invoke(messages)
            
            # Enhance with historical context tracking
            if not result.historical_context_used:
                historical_items = []
                if len(state.get("episode_images", [])) > 1:
                    historical_items.append(f"{len(state['episode_images']) - 1}_previous_screens")
                if state.get("action_history"):
                    historical_items.append(f"{len(state['action_history'])}_previous_actions")
                
                result.historical_context_used = historical_items
            
            return result
            
        except Exception as e:
            logger.debug(f"Structured output failed, using fallback: {e}")
            # Fallback to manual parsing if structured output fails
            response = self.model.invoke(messages)
            parsed_result = self._parse_model_response(
                response.content, 
                state["goal"], 
                state["current_step"]
            )
            
            # Add historical context tracking to fallback
            parsed_result.historical_context_used = [
                f"{len(state.get('episode_images', []))}_images_processed",
                "fallback_parsing_used"
            ]
            
            return parsed_result
    
    def plan_action(self, state) -> Command:
        """
        Override base plan_action to support both regular and enhanced states.
        
        This maintains compatibility with existing multi-agent system while adding
        enhanced functionality for episode evaluation.
        """
        # Check if this is an enhanced evaluation state
        if isinstance(state, dict) and "episode_images" in state:
            # Cast to evaluation state type for enhanced processing
            eval_state = state  # Type hint: this should be EpisodeEvaluationState
            return self.plan_action_with_history(eval_state)
        else:
            # Use base implementation for regular states
            return super().plan_action(state)


def enhanced_planning_node(state) -> Command:
    """Enhanced planning agent node for LangGraph with historical context."""
    agent = EnhancedPlanningAgent()
    
    # Use enhanced planning if historical context is available
    if isinstance(state, dict) and "episode_images" in state:
        return agent.plan_action_with_history(state)
    else:
        return agent.plan_action(state)
