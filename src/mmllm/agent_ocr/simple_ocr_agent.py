

from typing import Annotated, TypedDict, List, Dict, Any, Optional
from PIL import Image
from dataclasses import dataclass, asdict
import json
import base64
from io import BytesIO

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from mmllm.agent_ocr.prompt import DATASET_PROMPT
from mmllm.model import get_model


class ActionOutput(BaseModel):
    """Structured output schema for agent actions."""
    action_type: int = Field(description="Action type: 1=tap, 2=long_press, 3=swipe, 4=drag, 5=type_text, 0=task_complete")
    coordinates: List[float] = Field(description="x,y coordinates (0-1 normalized)", default=[0, 0])
    lift_coordinates: Optional[List[float]] = Field(default=[0, 0], description="lift coordinates for drag/swipe actions")
    text: Optional[str] = Field(default=None, description="text to type for type_text actions")
    task_done: bool = Field(description="true if the overall goal is achieved")


@dataclass
class HistoryItem:
    """Single interaction history item containing UI state and action taken"""
    image: Any  # PIL Image or image path
    ocr: str  # OCR text extracted from the image
    ui_description: str  # String describing the current UI state
    goal: str  # Current goal/task description
    action_taken: Optional[Dict[str, Any]] = None  # Action that was taken
    task_done: bool = False  # Whether the task is completed


class AgentState(TypedDict):
    """State schema for the LangGraph agent"""
    history: Annotated[List[HistoryItem], lambda x, y: x + y]  # History of past interactions
    current_image: Any  # Current UI screenshot
    current_ocr: str  # Current OCR text
    current_ui_description: str  # Current UI description
    goal: str  # Overall goal/task
    action: Optional[Dict[str, Any]]  # Current action to take
    task_completed: bool  # Whether the overall task is done
    # explanation: Optional[str]  # Explanation of the action taken

class SimpleOCRAgent:
    """LangGraph-based OCR agent for UI automation"""
    
    def __init__(self):
        """
        Initialize the agent with a language model
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for model responses
        """
        self.llm =get_model()
        
        # Create the state graph
        self.graph_builder = StateGraph(AgentState)
        
        # Add single model node
        self.graph_builder.add_node("model", self._model_node)
        
        # Add conditional edges - either continue waiting for next episode or end
        self.graph_builder.add_conditional_edges(
            "model",
            self._should_continue,
            {
                "continue": END,  # Wait for next user input
                "end": END
            }
        )
        
        # Set entry point
        self.graph_builder.set_entry_point("model")
        
        # Compile with memory
        memory = MemorySaver()
        self.graph = self.graph_builder.compile(checkpointer=memory)
    
    def _image_to_base64(self, image: Any) -> Optional[str]:
        """Convert PIL Image to base64 string."""
        if image is None:
            return None
        
        try:
            if isinstance(image, str):
                # If it's already a base64 string, return it
                return image
            elif hasattr(image, 'save'):
                # PIL Image
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                return None
        except Exception:
            return None
    
    def _build_multimodal_message(
        self, 
        goal: str, 
        current_step: int, 
        current_image: Any, 
        current_ocr: str,
        current_ui_description: str,
        history: Optional[List[HistoryItem]] = None
    ) -> HumanMessage:
        """Build multimodal message with historical context."""
        
        # Build text content with context
        text_content = f"""
        {DATASET_PROMPT}
        
        Goal: {goal}
        
        Step {current_step}: Analyze the current UI and decide the next action.
        
        Current UI State:
        - OCR Text: {current_ocr}
        - UI Description: {current_ui_description}
        """
        
        # Add recent history context
        if history:
            recent_history = history[-3:]  # Last 3 interactions
            if recent_history:
                text_content += "\n\nRecent History:\n"
                for i, item in enumerate(recent_history):
                    text_content += f"\nStep {len(history) - len(recent_history) + i + 1}:\n"
                    text_content += f"- OCR: {item.ocr}\n"
                    text_content += f"- UI: {item.ui_description}\n"
                    text_content += f"- Action: {item.action_taken}\n"
        
        text_content += """
        
        You must respond with a JSON object containing:
        {
            "action_type": <int>,  // 1=tap, 2=long_press, 3=swipe, 4=drag, 5=type_text
            "coordinates": [<float>, <float>],  // x,y coordinates (0-1 normalized)
            "lift_coordinates": [<float>, <float>],  // for drag/swipe actions
            "text": "<string>",  // for type_text actions
            "task_done": <bool>  // true if the overall goal is achieved
        }
        
        Examples:
        - Tap a button: {"action_type": 1, "coordinates": [0.5, 0.7], "task_done": false}
        - Type text: {"action_type": 5, "coordinates": [0.5, 0.3], "text": "hello world", "task_done": false}
        - Drag: {"action_type": 4, "coordinates": [0.2, 0.8], "lift_coordinates": [0.8, 0.8], "task_done": false}
        - Task complete: {"action_type": 0, "coordinates": [0, 0], "task_done": true}
        
        Respond only with the JSON object:
        """
        
        # Start with text content
        content = [
            {
                "type": "text",
                "text": text_content
            }
        ]
        
        # Add current image
        current_image_b64 = self._image_to_base64(current_image)
        if current_image_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{current_image_b64}"
                }
            })
        
        # Add historical images (limit to last 2 for memory)
        if history:
            recent_history = history[-2:]  # Last 2 images
            for item in recent_history:
                hist_image_b64 = self._image_to_base64(item.image)
                if hist_image_b64:
                    content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{hist_image_b64}"
                        }
                    })
        
        return HumanMessage(content=content)
    
    def _model_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Single model node that analyzes UI, decides action, and updates history
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with action and history
        """
        
        # Get current step number
        current_step = len(state.get('history', [])) + 1
        
        # Build multimodal message with all context
        message = self._build_multimodal_message(
            goal=state['goal'],
            current_step=current_step,
            current_image=state.get('current_image'),
            current_ocr=state.get('current_ocr', ''),
            current_ui_description=state.get('current_ui_description', ''),
            history=state.get('history')
        )
        
        # Invoke the model with multimodal content using structured output
        try:
            structured_model = self.llm.with_structured_output(ActionOutput)
            action_output = structured_model.invoke([message])
            
            # Convert Pydantic model to dict for compatibility
            action_data = action_output.model_dump()
            
        except Exception as e:
            # Fallback to regular text parsing if structured output fails
            print(f"Structured output failed, using fallback: {e}")
            response = self.llm.invoke([message])
            
            try:
                # Parse the JSON response
                action_data = json.loads(response.content.strip())
                
                # Validate the action format
                required_fields = ["action_type", "coordinates", "task_done"]
                if not all(field in action_data for field in required_fields):
                    raise ValueError("Missing required fields in action")
                    
            except (json.JSONDecodeError, ValueError) as parse_error:
                # Final fallback action if both parsing methods fail
                print(f"Error parsing action: {parse_error}")
                action_data = {
                    "action_type": 0,
                    "coordinates": [0, 0],
                    "task_done": True,
                    "error": f"Failed to parse action: {str(e)} -> {str(parse_error)}"
                }
        
        # Create new history item
        new_history_item = HistoryItem(
            image=state.get('current_image'),
            ocr=state.get('current_ocr', ''),
            ui_description=state.get('current_ui_description', ''),
            goal=state.get('goal', ''),
            action_taken=action_data,
            task_done=action_data.get("task_done", False)
        )
        
        return {
            "action": action_data,
            "task_completed": action_data.get("task_done", False),
            "history": [new_history_item]  # This will be appended to existing history
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine whether to continue (wait for next episode) or end
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" to wait for next user input or "end" to finish
        """
        if state.get('task_completed', False):
            return "end"
        
        # Check if we've hit a maximum number of steps
        if len(state.get('history', [])) >= 10:  # Max 10 steps
            return "end"
            
        # Continue means wait for next episode from user
        return "continue"
    
    def run_step(self, 
                 image: Any,
                 ocr_text: str, 
                 ui_description: str, 
                 goal: str,
                 thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent for one step/episode
        
        Args:
            image: Current UI screenshot
            ocr_text: OCR text from the image
            ui_description: Description of the UI state
            goal: Overall goal/task description
            thread_id: Thread ID for state persistence
            
        Returns:
            Action to take and updated state
        """
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Update current state for this step
        step_state = {
            "current_image": image,
            "current_ocr": ocr_text,
            "current_ui_description": ui_description,
            "goal": goal,
        }
        
        # Run the graph for one step
        final_state = None
        for event in self.graph.stream(step_state, config, stream_mode="values"):
            final_state = event
            
        # 
        return final_state
    
    def run_agent(self, 
                  image: Any,
                  ocr_text: str, 
                  ui_description: str, 
                  goal: str,
                  thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent for one interaction (backwards compatibility)
        
        Args:
            image: Current UI screenshot
            ocr_text: OCR text from the image
            ui_description: Description of the UI state
            goal: Overall goal/task description
            thread_id: Thread ID for state persistence
            
        Returns:
            Action to take and updated state
        """
        return self.run_step(image, ocr_text, ui_description, goal, thread_id)
    
    def get_state(self, thread_id: str = "default") -> Dict[str, Any]:
        """
        Get the current state of the agent
        
        Args:
            thread_id: Thread ID to get state for
            
        Returns:
            Current agent state
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        return snapshot.values


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = SimpleOCRAgent()
    
    # Example interaction
    result = agent.run_agent(
        image=None,  # Would be a PIL Image
        ocr_text="Login button Settings Menu",
        ui_description="Login screen with username field, password field, and login button",
        goal="Log into the application",
        thread_id="session_1"
    )
    
    print("Agent result:")
    print(json.dumps(result.get('action', {}), indent=2))