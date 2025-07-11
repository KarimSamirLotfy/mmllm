# LangGraph OCR Agent

This directory contains a simplified LangGraph-based agent for UI automation using OCR and vision capabilities.

## Overview

The `SimpleOCRAgent` is built using LangGraph with a **single-node architecture** where:

1. **State Management**: The agent maintains a history of past interactions, each containing:
   - Current UI screenshot (image)
   - OCR text extracted from the image
   - UI description string
   - Current goal/task
   - Action taken
   - Task completion status

2. **Agent Flow**: The agent follows this simple workflow:
   ```
   start → model → (wait for next episode) → model → ... → end
   ```

3. **Episode-Based Execution**: 
   - The model processes one episode at a time
   - Outputs an action and checks if the task is complete
   - If not complete, waits for the user to provide the next episode
   - If complete, ends the session

## Features

- **Single Node Design**: One model node handles analysis, decision making, and history updates
- **Episode-Based**: Processes one UI state at a time, waiting for user input between steps
- **Persistent State**: Uses LangGraph's MemorySaver to maintain conversation history across episodes
- **Structured Actions**: Outputs structured JSON actions compatible with UI automation frameworks
- **History Tracking**: Maintains a complete history of interactions for context-aware decision making

## Usage

### Basic Usage

```python
from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent

# Initialize the agent
agent = SimpleOCRAgent()

# Run a single episode/step
result = agent.run_step(
    image=your_screenshot,  # PIL Image object
    ocr_text="Login button Settings Menu",
    ui_description="Login screen with username and password fields",
    goal="Log into the application",
    thread_id="session_1"
)

# Get the action to perform
action = result.get('action', {})
print(action)

# Check if task is complete
if result.get('task_completed', False):
    print("Task completed!")
else:
    print("Continue to next episode...")
```

### Step-by-Step Workflow

```python
# Episode 1: Initial state
result1 = agent.run_step(
    image=screenshot1,
    ocr_text="Username: [field] Password: [field] [Login Button]",
    ui_description="Login screen with empty fields",
    goal="Log in with credentials",
    thread_id="login_flow"
)

# User executes the action from result1, then provides next episode
# Episode 2: After executing first action
result2 = agent.run_step(
    image=screenshot2,
    ocr_text="Username: john Password: [field] [Login Button]",
    ui_description="Login screen with username filled",
    goal="Log in with credentials",
    thread_id="login_flow"  # Same thread_id maintains context
)

# Continue until result.get('task_completed') == True
```

### Action Format

The agent outputs actions in this format:

```json
{
    "action_type": 1,
    "coordinates": [0.5, 0.7],
    "lift_coordinates": [0.8, 0.8],
    "text": "hello world",
    "task_done": false
}
```

**Action Types:**
- `1`: Tap/Click
- `2`: Long press
- `3`: Swipe
- `4`: Drag
- `5`: Type text

**Coordinates:** Normalized (0-1) x,y coordinates relative to screen dimensions

### State Structure

The agent maintains this state structure:

```python
class AgentState(TypedDict):
    history: List[HistoryItem]  # Past interactions
    current_image: Any  # Current screenshot
    current_ocr: str  # Current OCR text
    current_ui_description: str  # Current UI description
    goal: str  # Overall task goal
    action: Optional[Dict]  # Current action to take
    task_completed: bool  # Task completion status

class HistoryItem:
    image: Any  # Screenshot
    ocr: str  # OCR text
    ui_description: str  # UI description
    goal: str  # Goal at this step
    action_taken: Optional[Dict]  # Action that was taken
    task_done: bool  # Step completion status
```

### Example Scenarios

#### Login Flow (Episode-based)
```python
thread_id = "login_session"
goal = "Log in with username 'john' and password 'secret123'"

# Episode 1: Initial login screen
result1 = agent.run_step(
    image=screenshot1,
    ocr_text="Username: [field] Password: [field] [Login Button]",
    ui_description="Login screen with empty username and password fields",
    goal=goal,
    thread_id=thread_id
)
# User executes action, captures new screenshot

# Episode 2: After entering username
result2 = agent.run_step(
    image=screenshot2,
    ocr_text="Username: john Password: [field] [Login Button]",
    ui_description="Login screen with username filled, password field empty",
    goal=goal,
    thread_id=thread_id
)
# User executes action, captures new screenshot

# Episode 3: After entering password
result3 = agent.run_step(
    image=screenshot3,
    ocr_text="Username: john Password: ****** [Login Button]",
    ui_description="Login screen with both fields filled",
    goal=goal,
    thread_id=thread_id
)
# Continue until task_completed == True
```

#### Single Episode Example
```python
result = agent.run_step(
    image=main_menu_screenshot,
    ocr_text="Home Profile Settings Help Logout",
    ui_description="Main menu with navigation options",
    goal="Navigate to Settings page",
    thread_id="navigation"
)
```

## Advanced Features

### Thread Management
Use different `thread_id` values to maintain separate conversation contexts:

```python
# User session 1
agent.run_agent(..., thread_id="user_1_session")

# User session 2  
agent.run_agent(..., thread_id="user_2_session")
```

### State Inspection
Get the current state and history:

```python
current_state = agent.get_state("session_1")
history = current_state.get('history', [])

for item in history:
    print(f"OCR: {item.ocr}")
    print(f"Action: {item.action_taken}")
    print(f"Completed: {item.task_done}")
```

### Custom Models
Use different language models:

```python
# Using GPT-4
agent = SimpleOCRAgent(model_name="gpt-4o", temperature=0.1)

# Using GPT-3.5
agent = SimpleOCRAgent(model_name="gpt-3.5-turbo", temperature=0.2)
```

## Integration

To integrate with your UI automation framework:

1. **Initialize Agent**: Create a `SimpleOCRAgent` instance
2. **Start Session**: Begin with the first episode using a unique `thread_id`
3. **Episode Loop**:
   - Capture screenshot of current UI state
   - Extract OCR text from the image
   - Generate or provide a description of the UI elements
   - Call `agent.run_step()` with the current state
   - Execute the returned action on your target system
   - Capture the new UI state
   - Repeat until `task_completed` is True

### Example Integration Loop

```python
agent = SimpleOCRAgent()
thread_id = "automation_session_1"
goal = "Complete the user registration process"

episode = 1
while True:
    # Capture current state
    screenshot = capture_screenshot()
    ocr_text = extract_ocr(screenshot)
    ui_description = describe_ui(screenshot)
    
    # Get agent decision
    result = agent.run_step(
        image=screenshot,
        ocr_text=ocr_text,
        ui_description=ui_description,
        goal=goal,
        thread_id=thread_id
    )
    
    # Check if done
    if result.get('task_completed', False):
        print("Task completed successfully!")
        break
    
    # Execute the action
    action = result.get('action', {})
    execute_action(action)  # Your action execution function
    
    episode += 1
    if episode > 20:  # Safety limit
        print("Max episodes reached")
        break
```

## Dependencies

- `langgraph>=0.5.2`
- `langchain-openai`
- `PIL` (Pillow)
- `typing_extensions`

## Environment Setup

Make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Examples

See `step_by_step_example.py` for complete working examples of:
- Episode-based login flow automation
- Interactive mode for testing
- State management and history tracking
- Multi-step task completion with user control

Run the examples:
```bash
# Automated demo
python -m mmllm.agent_ocr.step_by_step_example

# Interactive mode
python -m mmllm.agent_ocr.step_by_step_example
# Then choose option 2
```
