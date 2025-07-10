"""Pydantic models for all Android in the Wild action types."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from ..android_in_the_wild.action_type import ActionType


class BaseAction(BaseModel):
    """Base action schema."""
    action_type: ActionType = Field(description="Type of action from AiTW ActionType enum")
    reasoning: str = Field(description="Explanation for choosing this action")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class TapAction(BaseAction):
    """Tap action with single touch point."""
    action_type: ActionType = ActionType.DUAL_POINT
    coordinates: List[float] = Field(
        description="Normalized coordinates [y, x] for tap location"
    )
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Coordinates must be [y, x]")
        if not all(0.0 <= coord <= 1.0 for coord in v):
            raise ValueError("Coordinates must be normalized between 0 and 1")
        return v


class SwipeAction(BaseAction):
    """Swipe action with start and end points."""
    action_type: ActionType = ActionType.DUAL_POINT
    start_coordinates: List[float] = Field(
        description="Normalized start coordinates [y, x]"
    )
    end_coordinates: List[float] = Field(
        description="Normalized end coordinates [y, x]"
    )
    
    @validator('start_coordinates', 'end_coordinates')
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Coordinates must be [y, x]")
        if not all(0.0 <= coord <= 1.0 for coord in v):
            raise ValueError("Coordinates must be normalized between 0 and 1")
        return v


class TypeAction(BaseAction):
    """Type text action."""
    action_type: ActionType = ActionType.TYPE
    text: str = Field(description="Text to type")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class NavigationAction(BaseAction):
    """Navigation button actions (back, home, enter)."""
    action_type: ActionType = Field(
        description="Must be PRESS_BACK, PRESS_HOME, or PRESS_ENTER"
    )
    
    @validator('action_type')
    def validate_navigation_type(cls, v):
        valid_types = {ActionType.PRESS_BACK, ActionType.PRESS_HOME, ActionType.PRESS_ENTER}
        if v not in valid_types:
            raise ValueError(f"Action type must be one of {valid_types}")
        return v


class StatusAction(BaseAction):
    """Task status actions (complete or impossible)."""
    action_type: ActionType = Field(
        description="Must be STATUS_TASK_COMPLETE or STATUS_TASK_IMPOSSIBLE"
    )
    completion_reason: str = Field(
        description="Reason for marking task as complete or impossible"
    )
    
    @validator('action_type')
    def validate_status_type(cls, v):
        valid_types = {ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE}
        if v not in valid_types:
            raise ValueError(f"Action type must be one of {valid_types}")
        return v


class ActionPlan(BaseModel):
    """Complete action plan with UI analysis."""
    action: BaseAction = Field(description="The specific action to execute")
    ui_elements_detected: List[dict] = Field(
        default_factory=list,
        description="UI elements detected in the current screen"
    )
    alternative_actions: List[BaseAction] = Field(
        default_factory=list,
        description="Alternative actions if primary fails"
    )
    expected_outcome: str = Field(
        description="Expected result after executing this action"
    )
