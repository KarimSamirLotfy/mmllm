"""Execution agent for action execution and error handling."""

from typing import Dict, Any
from langgraph.types import Command
from langchain_core.messages import AIMessage
from ..actions.action_executor import ActionExecutor
from ..actions.action_validator import ActionValidator
from .state import MultiAgentState, ExecutionOutput, AgentPhase
from ..android_in_the_wild.action_type import ActionType


class ExecutionAgent:
    """Executes planned actions and handles errors."""
    def __init__(self):
        self.executor = ActionExecutor()
        self.validator = ActionValidator()

    def execute_action(self, state: MultiAgentState) -> Command:
        try:
            plan = state.get("planning_output")
            ui_annotations = state.get("ui_annotations", [])
            if not plan:
                raise ValueError("No planning output found in state.")
            # Validate action
            is_valid, reason = self.validator.validate_action_plan(plan, ui_annotations)
            if not is_valid:
                execution_output = ExecutionOutput(
                    action_executed=False,
                    execution_details=reason,
                    error_message=reason,
                    state_change_detected=False
                )
                new_state = {
                    "execution_output": execution_output,
                    "current_phase": AgentPhase.REFLECTION,
                    "messages": state["messages"] + [AIMessage(content=f"Execution failed: {reason}")],
                    "error_count": state.get("error_count", 0) + 1,
                    "last_error": reason
                }
                return Command(goto="reflection_agent", update=new_state)
            # Execute action
            result = self.executor.execute_action(plan)
            execution_output = ExecutionOutput(
                action_executed=result.get("success", False),
                execution_details=result.get("simulation_note", "Executed action."),
                error_message=result.get("error"),
                state_change_detected=result.get("success", False)
            )
            new_state = {
                "execution_output": execution_output,
                "current_phase": AgentPhase.REFLECTION,
                "messages": state["messages"] + [AIMessage(content=f"Executed action: {result}")]
            }
            return Command(goto="reflection_agent", update=new_state)
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            execution_output = ExecutionOutput(
                action_executed=False,
                execution_details=error_msg,
                error_message=error_msg,
                state_change_detected=False
            )
            new_state = {
                "execution_output": execution_output,
                "current_phase": AgentPhase.REFLECTION,
                "messages": state["messages"] + [AIMessage(content=error_msg)],
                "error_count": state.get("error_count", 0) + 1,
                "last_error": error_msg
            }
            return Command(goto="reflection_agent", update=new_state)

def execution_node(state: MultiAgentState) -> Command:
    agent = ExecutionAgent()
    return agent.execute_action(state)
