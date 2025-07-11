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
    # Historical context usage tracking
    historical_context_used: Optional[List[str]] = Field(
        default=None, description="List of historical context items used in planning"
    )


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
    # AiTW format compliance
    aitw_action_format: Optional[Dict[str, Any]] = Field(
        default=None, description="Action in AiTW dataset format for comparison"
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


class StepEvaluationResult(BaseModel):
    """Evaluation result for a single step."""
    step_number: int = Field(description="Step number in the episode")
    agent_action: Dict[str, Any] = Field(description="Action predicted by agent")
    ground_truth_action: Dict[str, Any] = Field(description="Ground truth action from dataset")
    action_match: bool = Field(description="Whether actions match")
    coordinate_distance: Optional[float] = Field(
        default=None, description="Euclidean distance between coordinates for spatial actions"
    )
    action_type_match: bool = Field(description="Whether action types match")
    text_match: Optional[bool] = Field(
        default=None, description="Whether text inputs match for TYPE actions"
    )
    evaluation_score: float = Field(description="Score between 0-1 for this step")


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
    
    # Historical context for episodes
    episode_images: Optional[List[str]]  # All previous images in episode
    image_history: Optional[List[str]]  # Recent image history for context
    
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
    
    # Episode evaluation context
    current_ground_truth: Optional[Dict[str, Any]]  # Ground truth for current step
    ground_truth_actions: Optional[List[Dict[str, Any]]]  # All ground truth actions
    
    # Error handling
    error_count: int
    last_error: Optional[str]
    
    # Success tracking
    final_result: Optional[Dict[str, Any]]


class EpisodeEvaluationState(TypedDict):
    """Extended state for episode evaluation with historical context."""
    # Inherit all from MultiAgentState
    goal: str
    current_step: int
    max_steps: int
    current_phase: AgentPhase
    current_image: str
    ui_annotations: List[Dict[str, Any]]
    
    # Enhanced episode context
    episode_images: List[str]  # All images from episode start to current
    episode_id: str
    episode_length: int
    
    # Ground truth context
    current_ground_truth: Dict[str, Any]
    ground_truth_actions: List[Dict[str, Any]]
    
    # Messages and outputs
    messages: Annotated[Sequence[BaseMessage], add_messages]
    planning_output: Optional[PlanningOutput]
    execution_output: Optional[ExecutionOutput]
    reflection_output: Optional[ReflectionOutput]
    
    # Historical data
    action_history: List[PlanningOutput]
    execution_history: List[ExecutionOutput]
    reflection_history: List[ReflectionOutput]
    
    # Error handling
    error_count: int
    last_error: Optional[str]
    
    # Evaluation tracking
    step_evaluations: Optional[List[StepEvaluationResult]]
    final_result: Optional[Dict[str, Any]]
