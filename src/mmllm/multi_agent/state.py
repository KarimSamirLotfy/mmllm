"""Shared state definitions for the multi-agent system."""

from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from enum import Enum

# Import action types from AiTW
from ..android_in_the_wild.action_type import ActionType


class AgentPhase(str, Enum):
    """Phases of the multi-agent workflow."""
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    COMPLETE = "complete"
    FAILED = "failed"


class PlanningOutput(BaseModel):
    """Output from the planning agent."""
    action_type: ActionType = Field(description=f"Type of action to perform: {', '.join([str(action) for action in ActionType])}")
    coordinates: Optional[List[float]] = Field(
        default=None, description="Normalized coordinates [y, x] for touch actions"
    )
    lift_coordinates: Optional[List[float]] = Field(
        default=None, description="Normalized coordinates [y, x] for lift point in swipes"
    )
    text_input: Optional[str] = Field(
        default=None, description="Text to type for TYPE actions"
    )
    reasoning: str = Field(description="Explanation of why this action was chosen")
    confidence: float = Field(description="Confidence score between 0 and 1")
    # ui_elements_detected: List[dict] = Field(
    #     default_factory=list, 
    #     description="UI elements detected in the image",
    #     json_schema_extra={"additionalProperties": False}
    # )
    


class ExecutionOutput(BaseModel):
    """Output from the execution agent."""
    action_executed: bool = Field(description="Whether the action was successfully executed")
    execution_details: str = Field(description="Details about the execution")
    error_message: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )
    state_change_detected: bool = Field(
        default=False, description="Whether a state change was detected"
    )


class ReflectionOutput(BaseModel):
    """Output from the reflection agent."""
    goal_achieved: bool = Field(description="Whether the goal was achieved")
    progress_assessment: str = Field(description="Assessment of progress towards goal")
    next_strategy: str = Field(description="Strategy for next iteration")
    should_continue: bool = Field(description="Whether to continue or stop")
    feedback_for_planning: str = Field(
        description="Feedback to improve future planning"
    )


class MultiAgentState(TypedDict):
    """Shared state for the multi-agent system."""
    # Core goal and current step
    goal: str
    current_step: int
    max_steps: int
    
    # Current phase and agent control
    current_phase: AgentPhase
    
    # Multimodal inputs
    current_image: str  # Base64 encoded image
    ui_annotations: List[Dict[str, Any]]  # UI element annotations
    
    # Messages and communication
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Agent outputs
    planning_output: Optional[PlanningOutput]
    execution_output: Optional[ExecutionOutput]
    reflection_output: Optional[ReflectionOutput]
    
    # Historical data
    action_history: List[PlanningOutput]
    execution_history: List[ExecutionOutput]
    reflection_history: List[ReflectionOutput]
    
    # Episode context
    episode_id: Optional[str]
    episode_length: Optional[int]
    
    # Error handling
    error_count: int
    last_error: Optional[str]
    
    # Success tracking
    final_result: Optional[Dict[str, Any]]
