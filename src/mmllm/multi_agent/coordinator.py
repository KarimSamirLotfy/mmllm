"""Multi-agent orchestration using LangGraph StateGraph."""

from langgraph.graph import StateGraph, START, END
from .state import MultiAgentState, AgentPhase
from .planning_agent import planning_node
from .execution_agent import execution_node
from .reflection_agent import reflection_node
from ..android_in_the_wild.action_type import ActionType


def build_multi_agent_graph():
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("planning_agent", planning_node)
    graph_builder.add_node("execution_agent", execution_node)
    graph_builder.add_node("reflection_agent", reflection_node)

    # Conditional routing after planning
    def route_after_planning(state: MultiAgentState):
        planning_output = state.get("planning_output")
        if planning_output and planning_output.action_type in [
            ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE
        ]:
            return "END"
        return "execution_agent"

    # Conditional routing after execution
    def route_after_execution(state: MultiAgentState):
        return "reflection_agent"

    # Conditional routing after reflection
    def route_after_reflection(state: MultiAgentState):
        reflection_output = state.get("reflection_output")
        if reflection_output and reflection_output.goal_achieved:
            return "END"
        if state.get("error_count", 0) > 3:
            return "END"
        if state.get("current_step", 0) >= state.get("max_steps", 10):
            return "END"
        return "planning_agent"

    graph_builder.add_conditional_edges(
        "planning_agent", route_after_planning, {"execution_agent": "execution_agent", "END": END}
    )
    graph_builder.add_conditional_edges(
        "execution_agent", route_after_execution, {"reflection_agent": "reflection_agent"}
    )
    graph_builder.add_conditional_edges(
        "reflection_agent", route_after_reflection, {"planning_agent": "planning_agent", "END": END}
    )
    graph_builder.add_edge(START, "planning_agent")
    return graph_builder.compile()

class MultiAgentCoordinator:
    def __init__(self):
        self.graph = build_multi_agent_graph()

    def run(self, initial_state: MultiAgentState):
        return self.graph.invoke(initial_state)
