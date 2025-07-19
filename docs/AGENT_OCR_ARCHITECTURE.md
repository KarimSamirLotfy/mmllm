# SimpleOCRAgent System Architecture

## Overview

The `SimpleOCRAgent` is a LangGraph-based multimodal UI automation system designed for Android interface interaction using the Android in the Wild (AiTW) dataset. It employs a single-node architecture with episode-based execution, persistent state management, and OCR-enhanced vision capabilities.

## System Architecture

### Core Components

#### 1. **SimpleOCRAgent** (`simple_ocr_agent.py`)
The main agent class that orchestrates UI automation through LangGraph's StateGraph.

**Key Features:**
- **Single-node LangGraph architecture** with a model node for decision making
- **Episode-based execution** - processes one UI state at a time
- **Persistent state management** using LangGraph's MemorySaver
- **Structured output** using Pydantic models
- **OCR module integration** for enhanced UI element detection

#### 2. **State Management System**

**AgentState (TypedDict):**
```python
class AgentState:
    history: List[HistoryItem]           # Past interactions
    current_image: Any                   # Current UI screenshot  
    current_ocr: str                     # Current OCR text
    current_ui_description: str          # Current UI description
    goal: str                           # Overall task goal
    action: Optional[Dict[str, Any]]     # Current action to take
    task_completed: bool                # Task completion status
```

**HistoryItem (Dataclass):**
```python
class HistoryItem:
    image: Any                          # Screenshot
    ocr: str                           # OCR text
    ui_description: str                 # UI description  
    goal: str                          # Goal at this step
    action_taken: Optional[Dict]        # Action that was taken
    task_done: bool                    # Step completion status
```

#### 3. **Action Schema** (`ActionOutput`)
Structured Pydantic model defining agent outputs:

```python
class ActionOutput:
    action_type: Literal[3,4,5,6,7,10,11]  # AiTW action types
    coordinates: List[float]                # Normalized [x,y] coordinates
    lift_coordinates: List[float]           # For drag/swipe actions
    text: Optional[str]                     # Text for type actions
    task_done: bool                        # Task completion flag
    explanation: str                       # Action reasoning
```

**Action Types:**
- `3`: TYPE - Send text to emulator
- `4`: DUAL_POINT - Gestures (tap, swipe, pinch, zoom)
- `5`: PRESS_BACK - Back button
- `6`: PRESS_HOME - Home button  
- `7`: PRESS_ENTER - Enter key
- `10`: STATUS_TASK_COMPLETE - Task completed
- `11`: STATUS_TASK_IMPOSSIBLE - Task impossible

## Technical Components

### 1. **Image Processing Pipeline** (`extract.py`, `process.py`)

#### **OCR Extraction** (`extract.py`)
- **Tesseract OCR** integration with preprocessing
- **UI element detection** with bounding boxes
- **Confidence-based filtering** (>50% confidence threshold)
- **Coordinate normalization** to [0,1] range
- **Visual output** with bounding box overlays

```python
def extract_ui_elements(pil_image, use_preprocess=True, normalize=False):
    # Returns: (elements_list, visualization_image)
```

#### **Grid Overlay System** (`process.py`)
- **Dynamic grid generation** based on image dimensions
- **Anchor point marking** for coordinate reference
- **Visual calibration aids** for precise targeting
- **Transparency control** for overlay visibility

```python
def add_grid_with_anchors(image, grid_spacing_percent=0.1):
    # Adds calibration grid with anchor points
```

### 2. **LangGraph Architecture**

#### **Graph Structure:**
```
START → model_node → conditional_edge → {continue: END, end: END}
```

#### **Model Node** (`_model_node`)
Single processing node that:
1. **Builds multimodal messages** with image + text context
2. **Invokes LLM** with structured output schema
3. **Creates history items** for state persistence
4. **Returns action decisions** with metadata

#### **Conditional Routing** (`_should_continue`)
Determines execution flow:
- `"continue"` → Wait for next episode (user input)
- `"end"` → Task completed or max steps reached

#### **Memory Management**
- **MemorySaver checkpointer** for persistent state
- **Thread-based isolation** for multiple sessions
- **Historical context** in multimodal messages

### 3. **Multimodal Message Construction**

#### **Context Building:**
- **Current state**: Image, OCR text, UI description, goal
- **Historical context**: Last 3 interactions for decision context
- **Visual elements**: Base64-encoded images with proper MIME types
- **Structured prompting**: Dataset-specific instructions

#### **Message Format:**
```python
HumanMessage(content=[
    {"type": "text", "text": context_prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
])
```

### 4. **OCR Module Integration**

When `ocr_module=True`:
1. **Extracts UI elements** using Tesseract
2. **Adds grid overlays** for coordinate reference
3. **Enhances UI descriptions** with detected elements
4. **Provides location hints** for element targeting

### 5. **Execution Flow**

#### **Episode-Based Processing:**
```python
# Episode 1: Initial state
result1 = agent.run_step(image=screenshot1, goal=goal, thread_id="session")

# Episode 2: After action execution  
result2 = agent.run_step(image=screenshot2, goal=goal, thread_id="session")

# Continue until task_completed == True
```

#### **Action Execution Pipeline:**
1. **State preparation** with current UI context
2. **LLM invocation** with structured output
3. **Action validation** and fallback handling
4. **History update** with new interaction
5. **Coordinate transformation** for ground truth matching

### 6. **Error Handling & Fallbacks**

#### **Structured Output Fallback:**
```python
try:
    structured_model = self.llm.with_structured_output(ActionOutput)
    action_output = structured_model.invoke([message])
except Exception:
    # Fallback to JSON parsing
    response = self.llm.invoke([message])
    action_data = json.loads(response.content)
```

#### **Final Fallback Action:**
If all parsing fails, returns safe default action with error metadata.

## Architecture Patterns

### 1. **Single-Node Simplicity**
- **Reduced complexity** compared to multi-agent systems
- **Centralized decision making** in one model node
- **Simplified state management** with direct updates

### 2. **Episode-Based Execution**
- **User-controlled progression** between UI states
- **Natural interaction flow** mimicking human behavior
- **State persistence** across episodes

### 3. **Structured Output Enforcement**
- **Type safety** with Pydantic schemas
- **Validation** of action format
- **Consistent API** for downstream systems

### 4. **Memory-Aware Processing**
- **Historical context** for decision making
- **Thread isolation** for concurrent sessions
- **Persistent state** across interactions

## Visual Architecture Diagrams

### 1. **Overall System Architecture**

```OK
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SimpleOCRAgent System                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Input Layer   │    │ Processing Layer│    │  Output Layer   │             │
│  │                 │    │                 │    │                 │             │
│  │ • Screenshots   │────▶ • OCR Module    │────▶ • Structured    │             │
│  │ • OCR Text      │    │ • Grid Overlay  │    │   Actions       │             │
│  │ • UI Desc       │    │ •               │    │ • Coordinates   │             │
│  │ • Goal          │    │ • LLM           │    │ • Task Status   │             │
│  │ •          │    │      • Memory        │    │ • Explanations  │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│           │                       │                       │                    │
│           │              ┌─────────────────┐               │                    │
│           │              │  State Manager  │               │                    │
│           │              │                 │               │                    │
│           └──────────────▶ • AgentState    │◀──────────────┘                    │
│                          │ • HistoryItems  │                                    │
│                          │                 │                                    │
│                          │                 │                                    │
│                          └─────────────────┘                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. **LangGraph Node Architecture**

```OK
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LangGraph Execution Flow                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    START                                                                        │
│      │                                                                         │
│      ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          MODEL NODE                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │ Message Builder │  │  LLM Invoker   │  │ History Updater │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• Multimodal msg │──▶• Structured out │──▶• Create history │         │   │
│  │  │• Context build  │  │• Error handling │  │• Update state   │         │   │
│  │  │• Image encoding │  │• Fallback logic │  │• Return action  │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│                          ┌─────────────────┐                                  │
│                          │ CONDITIONAL EDGE│                                  │
│                          │ (_should_continue)│                                │
│                          │  model choose   │                                  │
│                          │ task_completed? │                                  │
│                          │ max_steps?      │                                  │
│                          └─────────────────┘                                  │
│                             │           │                                     │
│                    ┌────────┘           └────────┐                            │
│                    ▼                             ▼                            │
│                ┌─────────┐                  ┌─────────┐                       │
│                │"continue"│                  │  "end"  │                       │
│                │   END   │                  │   END   │                       │
│                │(wait for│                  │(task    │                       │
│                │ next    │                  │complete)│                       │
│                │episode) │                  │         │                       │
│                └─────────┘                  └─────────┘                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3. **State Management & Memory System**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         State Management Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          AgentState (Current)                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │current_image    │  │current_ocr      │  │current_ui_desc  │         │   │
│  │  │(PIL/base64)     │  │(string)         │  │(string)         │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │goal (string)    │  │action (dict)    │  │task_completed   │         │   │
│  │  │                 │  │                 │  │(boolean)        │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        History (List[HistoryItem])                     │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │   Episode 1     │  │   Episode 2     │  │   Episode N     │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• image          │  │• image          │  │• image          │         │   │
│  │  │• ocr            │  │• ocr            │  │• ocr            │         │   │
│  │  │• ui_description │  │• ui_description │  │• ui_description │         │   │
│  │  │• goal           │  │• goal           │  │• goal           │         │   │
│  │  │• action_taken   │  │• action_taken   │  │• action_taken   │         │   │
│  │  │• task_done      │  │• task_done      │  │• task_done      │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        MemorySaver (Persistent)                        │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │   Thread 1      │  │   Thread 2      │  │   Thread N      │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• session_1_hist │  │• login_flow     │  │• automation_x   │         │   │
│  │  │• checkpoints    │  │• checkpoints    │  │• checkpoints    │         │   │
│  │  │• state_snapshots│  │• state_snapshots│  │• state_snapshots│         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. **OCR Module Integration Flow**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OCR Module Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐                                                           │
│  │  Input Image    │                                                           │
│  │  (PIL/numpy)    │                                                           │
│  └─────────────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐        │
│  │ Tesseract OCR   │      │ Grid Overlay    │      │ UI Enhancement  │        │
│  │                 │      │                 │      │                 │        │
│  │• Text extract   │      │• Add grid lines │      │• Enhance desc   │        │
│  │• Confidence >50%│◀─────│• Anchor points  │◀─────│• Location hints │        │
│  │• Bounding boxes │      │• Visual cal     │      │• Element refs   │        │
│  │• Coordinates    │      │• Transparency   │      │• JSON format    │        │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘        │
│           │                        │                        │                 │
│           ▼                        ▼                        ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          Multimodal Message                            │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │Enhanced OCR Text│  │Grid-Overlaid Img│  │Context Prompt   │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• Element list   │  │• Base64 encoded │  │• Task goal      │         │   │
│  │  │• Click points   │  │• Visual aids    │  │• UI description │         │   │
│  │  │• Confidence     │  │• Calibration    │  │• History context│         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│                            ┌─────────────────┐                                │
│                            │ LLM Processing  │                                │
│                            │ (Vision + Text) │                                │
│                            └─────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. **Episode-Based Execution Timeline**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Episode-Based Execution Flow                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Time:     t0          t1          t2          t3          t4                  │
│           ┌──┐        ┌──┐        ┌──┐        ┌──┐        ┌──┐                 │
│  Episode: │ 1│───────▶│ 2│───────▶│ 3│───────▶│ 4│───────▶│ N│                 │
│           └──┘        └──┘        └──┘        └──┘        └──┘                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Episode Lifecycle                              │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │1. State Input   │  │2. Agent Process │  │3. Action Output │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• Screenshot     │──▶• Build message  │──▶• Structured act │         │   │
│  │  │• OCR text       │  │• LLM invoke     │  │• Coordinates    │         │   │
│  │  │• UI description │  │• Parse response │  │• Explanation    │         │   │
│  │  │• Goal context   │  │• Update history │  │• Task status    │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     User/System Action Execution                       │   │
│  │                                                                         │   │
│  │  • Execute action on target system                                     │   │
│  │  • Capture new UI state                                                │   │
│  │  • Extract new OCR/UI description                                      │   │
│  │  • Prepare next episode input                                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│                          ┌─────────────────┐                                  │
│                          │Wait for next    │                                  │
│                          │episode OR       │                                  │
│                          │task complete    │                                  │
│                          └─────────────────┘                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. **Action Output Schema Flow**

``` OK
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Action Processing Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐                                                           │
│  │ LLM Response    │                                                           │
│  │ (Raw Output)    │                                                           │
│  └─────────────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       Structured Output Parsing                        │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │   Primary Try   │  │  Fallback Try   │  │   Final Try     │         │   │
│  │  │                 │  │                 │  │                 │         │   │
│  │  │• Pydantic model │  │• JSON parsing   │  │• Safe defaults  │         │   │
│  │  │• with_structured│  │• Manual validation│ │• Error metadata │         │   │
│  │  │• ActionOutput   │  │• Required fields│  │• Graceful fail  │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        ActionOutput (Validated)                        │   │
│  │                                                                         │   │
│  │  action_type: [3,4,5,6,7,10,11]  ┌─ 3: TYPE (text input)              │   │
│  │  coordinates: [x,y] (0-1 norm)   ├─ 4: DUAL_POINT (tap/swipe)          │   │
│  │  lift_coordinates: [x,y]         ├─ 5: PRESS_BACK                      │   │
│  │  text: Optional[str]             ├─ 6: PRESS_HOME                      │   │
│  │  task_done: bool                 ├─ 7: PRESS_ENTER                     │   │
│  │  explanation: str                ├─ 10: STATUS_TASK_COMPLETE           │   │
│  │                                  └─ 11: STATUS_TASK_IMPOSSIBLE         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       Coordinate Transformation                        │   │
│  │                                                                         │   │
│  │  • Flip coordinates: [x,y] → [y,x] for ground truth matching          │   │
│  │  • Apply to both coordinates and lift_coordinates                     │   │
│  │  • Maintain normalization [0-1] range                                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. **Dataset Integration** (`aitw.py`)
- **AiTW episode loading** and processing
- **Ground truth comparison** for evaluation
- **Action matching** validation

### 2. **Visualization Integration**
- **Episode plotting** with action overlays
- **Grid visualization** for coordinate reference
- **UI element highlighting** from OCR

### 3. **Model Integration** (`model.py`)
- **Azure OpenAI** configuration
- **Multimodal capabilities** for vision + text
- **Structured output** support

## Usage Patterns

### 1. **Basic Episode Processing**
```python
agent = SimpleOCRAgent(ocr_module=True)
result = agent.run_step(
    image=screenshot,
    ocr_text=extracted_text,
    ui_description=ui_state,
    goal="Complete user registration",
    thread_id="session_1"
)
```

### 2. **Automated Evaluation**
```python
# Process multiple episodes for benchmarking
for episode_idx, episode_tf in enumerate(episodes):
    episode = episode_loader.load_episode_with_history(episode_tf)
    for step in range(len(episode['episode_images'])):
        result = agent.run_step(
            image=episode['episode_images'][step],
            ui_description=episode['ui_annotations_list'][step],
            goal=episode['goal'],
            thread_id=episode['episode_id']
        )
```

### 3. **Interactive Testing**
```python
# Step-by-step interaction with user control
while not task_completed:
    result = agent.run_step(current_ui_state, thread_id=session)
    execute_action(result['action'])
    current_ui_state = capture_new_state()
    task_completed = result.get('task_completed', False)
```

## Performance Characteristics

### **Strengths:**
- **Simple architecture** - easy to understand and debug
- **Persistent memory** - context-aware decisions
- **Structured output** - reliable action format
- **OCR enhancement** - improved element detection
- **Episode flexibility** - user-controlled progression

### **Considerations:**
- **Single model dependency** - all decisions through one LLM
- **Sequential processing** - no parallel analysis
- **Memory growth** - history accumulates over time
- **Coordinate accuracy** - depends on OCR and visual reasoning



This architecture provides a robust foundation for multimodal UI automation with clear separation of concerns, persistent state management, and flexible integration capabilities.
