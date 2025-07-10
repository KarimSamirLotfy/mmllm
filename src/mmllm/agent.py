import logging
from typing import TypedDict, Annotated, Sequence, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import dotenv
import os
import os
from openai import AzureOpenAI

from .model import get_model

logger = logging.getLogger(__name__)

# 1. Define the agent state
class AgentState(TypedDict):
    goal: str
    next_step: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    multimodal_input: Dict[str, Any]  # e.g., {"text": ..., "image": ...}

# 2. Define the output schema for coordinates
class ClickCoordinates(TypedDict):
    x: int
    y: int
    description: str

# 3. Define the agent node
def multimodal_agent(state: AgentState) -> Command[Literal[END]]:
    model = get_model()
    prompt = [
        {"role": "system", "content": f"Goal: {state['goal']}"},
        {"role": "system", "content": f"Current step: {state['next_step']}"},
       {"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": state['multimodal_input']['text'],
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": state['multimodal_input']['image'],  # Assuming image is base64 encoded
                    "mime_type": "image/jpeg",
                },
            ],
        }
        # For image, pass as appropriate for your model
        # e.g., {"role": "user", "content": state['multimodal_input']['image']}
    ]

    # Ask the model to output the next coordinates to click on
    response = model.with_structured_output(ClickCoordinates).invoke(prompt)
    coords = response  # Should be a dict with x, y, description

    # Update state for next step if needed
    new_state = {
        "goal": state["goal"],
        "next_step": f"Clicked at ({coords['x']}, {coords['y']})",
        "messages": state["messages"],
        "multimodal_input": state["multimodal_input"],
    }
    return Command(goto=END, update=new_state)

# 4. Build the graph
def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("multimodal_agent", multimodal_agent)
    graph_builder.add_edge(START, "multimodal_agent")
    graph = graph_builder.compile()
    return graph
# 5. Example invocation
def run_agent(goal: str, next_step: str, text: str, image: Any):
    initial_state = {
        "goal": goal,
        "next_step": next_step,
        "messages": [],
        "multimodal_input": {"text": text, "image": image},
    }
    graph = build_graph()
    result = graph.invoke(initial_state)
    return result

if __name__ == "__main__":
    # Example usage
    goal = "Find the back button"
    next_step = "Identify the button location"
    text_input = "Please find the submit button."
    image_input = None  # Replace with actual image data if available

    result = run_agent(goal, next_step, text_input, image_input)
    logger.info(result)