"""Reflection agent for outcome analysis and feedback."""

from langgraph.types import Command
from langchain_core.messages import AIMessage
from .state import MultiAgentState, ReflectionOutput, AgentPhase

class ReflectionAgent:
    """Analyzes execution outcomes and provides feedback for planning."""
    def __init__(self):
        pass

    def reflect(self, state: MultiAgentState) -> Command:
        try:
            execution_output = state.get("execution_output")
            current_step = state["current_step"]
            max_steps = state["max_steps"]
            error_count = state.get("error_count", 0)
            # Simple logic: if action executed and no error, check if goal achieved
            if execution_output and execution_output.action_executed:
                # For demo, assume goal is achieved if action executed and step >= max_steps
                goal_achieved = (current_step + 1 >= max_steps)
                progress = f"Step {current_step+1}/{max_steps}. Action executed successfully."
                next_strategy = "Continue" if not goal_achieved else "Stop"
                should_continue = not goal_achieved
                feedback = "Good progress."
            else:
                goal_achieved = False
                progress = f"Step {current_step+1}/{max_steps}. Action failed."
                next_strategy = "Replan"
                should_continue = True
                feedback = execution_output.error_message if execution_output else "Unknown error."
            reflection_output = ReflectionOutput(
                goal_achieved=goal_achieved,
                progress_assessment=progress,
                next_strategy=next_strategy,
                should_continue=should_continue,
                feedback_for_planning=feedback
            )
            # Decide next phase
            if goal_achieved:
                next_phase = AgentPhase.COMPLETE
                goto = "END"
            elif error_count > 3:
                next_phase = AgentPhase.FAILED
                goto = "END"
            elif should_continue:
                next_phase = AgentPhase.PLANNING
                goto = "planning_agent"
            else:
                next_phase = AgentPhase.COMPLETE
                goto = "END"
            new_state = {
                "reflection_output": reflection_output,
                "current_phase": next_phase,
                "messages": state["messages"] + [AIMessage(content=f"Reflection: {progress}")],
                "current_step": state["current_step"] + 1
            }
            return Command(goto=goto, update=new_state)
        except Exception as e:
            error_msg = f"Reflection error: {str(e)}"
            reflection_output = ReflectionOutput(
                goal_achieved=False,
                progress_assessment=error_msg,
                next_strategy="Replan",
                should_continue=True,
                feedback_for_planning=error_msg
            )
            new_state = {
                "reflection_output": reflection_output,
                "current_phase": AgentPhase.FAILED,
                "messages": state["messages"] + [AIMessage(content=error_msg)],
                "error_count": state.get("error_count", 0) + 1,
                "last_error": error_msg
            }
            return Command(goto="END", update=new_state)

def reflection_node(state: MultiAgentState) -> Command:
    agent = ReflectionAgent()
    return agent.reflect(state)
