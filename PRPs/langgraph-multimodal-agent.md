name: "LangGraph Multimodal Agent for Android in the Wild Dataset"
description: |
  Create a comprehensive multi-agent system using LangGraph for vision-centric mobile device control on the Android in the Wild dataset.

## Goal
Build a LangGraph-based multimodal agent system that can autonomously interact with Android interfaces using the Android in the Wild (AiTW) dataset. The system should implement a three-phase architecture: Planning, Execution, and Reflection, with each phase handled by separate agents that communicate via feedback loops.

## Why
- **Research Value**: Advance multimodal AI research in mobile device automation
- **Integration with AiTW**: Leverage Google's large-scale dataset for mobile device control
- **Multi-Agent Architecture**: Demonstrate sophisticated agent orchestration patterns
- **Vision-Language Understanding**: Combine visual GUI understanding with action planning

## What
A multi-agent system that can:
- Analyze Android UI screenshots using vision-language models
- Plan action sequences based on natural language goals
- Execute actions with proper error handling and state management
- Reflect on outcomes and adjust strategies
- Handle all AiTW action types: tap, swipe, type, navigation buttons

### Success Criteria
- [ ] Three distinct agents (Planning, Execution, Reflection) working in coordination
- [ ] Successful processing of AiTW dataset examples
- [ ] Structured output for all action types defined in AiTW
- [ ] Feedback loop between agents with state persistence
- [ ] Comprehensive error handling and recovery mechanisms
- [ ] Integration tests using real AiTW episodes

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/langchain-ai/langgraph
  why: Latest LangGraph patterns for multi-agent systems and StateGraph implementation
  critical: Command-based routing, multi-agent orchestration, state management

- url: https://arxiv.org/abs/2307.10088
  why: Android in the Wild paper - understand dataset structure and action types
  critical: Action matching, UI annotation format, evaluation metrics

- file: src/mmllm/agent.py
  why: Existing single-agent implementation pattern to extend
  critical: Shows current StateGraph usage, structured output with Pydantic

- file: src/mmllm/android_in_the_wild/action_type.py
  why: Defines all supported action types in AiTW dataset
  critical: ACTION_TYPE enumeration, dual_point vs other actions

- file: src/mmllm/android_in_the_wild/visualization_utils.py
  why: Image processing and UI annotation handling utilities
  critical: _decode_image function, annotation position processing

- file: src/mmllm/android_in_the_wild/demo.ipynb
  why: Shows how to load and process AiTW dataset examples
  critical: Episode extraction, feature access patterns

- doc: https://python.langchain.com/docs/concepts/structured_outputs
  section: Pydantic models with with_structured_output
  critical: Vision model integration, multimodal input handling

- docfile: src/mmllm/model.py
  why: Azure OpenAI configuration and model initialization
  critical: Multimodal model setup for vision tasks
```

### Current Codebase Structure
```bash
src/
├── mmllm/
│   ├── __init__.py
│   ├── agent.py                    # Current single-agent implementation
│   ├── main.py                     # Dataset loading and simple agent usage
│   ├── model.py                    # Azure OpenAI model configuration
│   └── android_in_the_wild/        # AiTW dataset utilities
│       ├── __init__.py
│       ├── action_matching.py      # Action comparison utilities
│       ├── action_type.py          # Action type enumeration
│       ├── demo.ipynb             # Dataset exploration notebook
│       ├── example.episode.md     # Sample episode structure
│       ├── README.md              # Dataset documentation
│       ├── requirements.txt       # AiTW specific dependencies
│       └── visualization_utils.py # Image and annotation processing
PRPs/
├── commands/
│   └── generate-prp.md
└── templates/
    └── prp_base.md
pyproject.toml                     # Project dependencies
```

### Desired Codebase Structure with New Files
```bash
src/
├── mmllm/
│   ├── multi_agent/               # NEW: Multi-agent system
│   │   ├── __init__.py
│   │   ├── state.py              # Shared state definitions
│   │   ├── planning_agent.py     # Vision-based action planning
│   │   ├── execution_agent.py    # Action execution with error handling
│   │   ├── reflection_agent.py   # Outcome analysis and feedback
│   │   └── coordinator.py        # Multi-agent orchestration
│   ├── actions/                   # NEW: Action handling
│   │   ├── __init__.py
│   │   ├── action_schemas.py     # Pydantic models for all action types
│   │   ├── action_executor.py    # Execute actions on Android interface
│   │   └── action_validator.py   # Validate action feasibility
│   ├── vision/                    # NEW: Vision processing
│   │   ├── __init__.py
│   │   ├── image_processor.py    # Image preprocessing for multimodal models
│   │   ├── ui_analyzer.py        # UI element detection and analysis
│   │   └── annotation_parser.py  # Parse AiTW UI annotations
│   └── utils/                     # NEW: Utilities
│       ├── __init__.py
│       ├── episode_loader.py     # AiTW episode loading utilities
│       └── state_manager.py      # Agent state persistence
tests/                             # NEW: Test suite
├── __init__.py
├── test_multi_agent.py
├── test_actions.py
├── test_vision.py
└── fixtures/
    └── sample_episodes.py
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: LangGraph 0.5.2 requires specific Command usage patterns
# Use Command(goto=agent_name, update=state_dict) for agent routing
# StateGraph requires TypedDict for state management

# CRITICAL: Azure OpenAI multimodal requires specific message format
# Images must be base64 encoded with proper MIME type specification
# Use "image_url" type with data:image/jpeg;base64,{data} format

# CRITICAL: AiTW dataset uses normalized coordinates [0,1]
# Touch coordinates are in format [y, x] not [x, y]
# Action type 4 (DUAL_POINT) covers both taps and swipes

# CRITICAL: TensorFlow dataset requires tf.train.Example parsing
# Images are encoded as raw bytes and need tf.io.decode_raw
# UI annotations are flattened arrays that need reshaping

# CRITICAL: Pydantic v2 syntax changes
# Use Field(..., description="...") not Field(description="...")
# BaseModel.model_validate() not BaseModel.parse_obj()
```

## Implementation Blueprint

### Data Models and Structure

Create comprehensive Pydantic models for type safety and validation:

```python
# Core state management
class MultiAgentState(TypedDict):
    episode_id: str
    goal: str
    current_step: int
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_screenshot: Dict[str, Any]  # base64 image + metadata
    ui_annotations: List[Dict[str, Any]]
    action_history: List[Dict[str, Any]]
    execution_feedback: Optional[Dict[str, Any]]
    reflection_insights: List[str]
    plan: Optional[Dict[str, Any]]
    
# Action schemas for all AiTW action types
class TapAction(BaseModel):
    action_type: Literal["tap"] = "tap"
    coordinates: Tuple[float, float]  # normalized [y, x]
    description: str
    confidence: float
    
class SwipeAction(BaseModel):
    action_type: Literal["swipe"] = "swipe"
    start_coordinates: Tuple[float, float]
    end_coordinates: Tuple[float, float]
    description: str
    confidence: float
    
class TypeAction(BaseModel):
    action_type: Literal["type"] = "type"
    text: str
    target_element: Optional[str]
    description: str
    
class NavigationAction(BaseModel):
    action_type: Literal["navigation"] = "navigation"
    button: Literal["back", "home", "enter"]
    description: str
```

### List of Tasks to Complete (in order)

```yaml
Task 1: Create Core State Management
CREATE src/mmllm/multi_agent/state.py:
  - DEFINE MultiAgentState TypedDict with all required fields
  - IMPLEMENT state validation and serialization helpers
  - PATTERN: Follow existing AgentState in src/mmllm/agent.py

Task 2: Build Action Schema System  
CREATE src/mmllm/actions/action_schemas.py:
  - DEFINE Pydantic models for all AiTW action types
  - IMPLEMENT Union type for polymorphic action handling
  - MIRROR pattern from existing ClickCoordinates in agent.py
  - REFERENCE action_type.py for complete action enumeration

Task 3: Implement Vision Processing Pipeline
CREATE src/mmllm/vision/image_processor.py:
  - IMPLEMENT base64 encoding from TensorFlow images
  - MIRROR pattern from main.py image processing
  - ADD image resizing and format normalization
  
CREATE src/mmllm/vision/ui_analyzer.py:
  - PARSE UI annotations from AiTW format
  - IMPLEMENT bounding box processing
  - REFERENCE visualization_utils.py patterns

Task 4: Build Planning Agent
CREATE src/mmllm/multi_agent/planning_agent.py:
  - IMPLEMENT multimodal LLM integration for GUI analysis
  - GENERATE action plans using vision + text input
  - USE structured output with action schemas
  - PATTERN: Extend multimodal_agent from agent.py

Task 5: Build Execution Agent  
CREATE src/mmllm/multi_agent/execution_agent.py:
  - IMPLEMENT action execution with error handling
  - VALIDATE actions against UI state
  - PROVIDE feedback to planning and reflection agents
  - HANDLE all AiTW action types

Task 6: Build Reflection Agent
CREATE src/mmllm/multi_agent/reflection_agent.py:
  - ANALYZE execution outcomes and errors
  - GENERATE insights for future planning
  - IMPLEMENT feedback loop mechanisms
  - MAINTAIN action history and learning

Task 7: Create Multi-Agent Coordinator
CREATE src/mmllm/multi_agent/coordinator.py:
  - IMPLEMENT LangGraph StateGraph with three agents
  - DEFINE conditional routing based on agent outputs
  - HANDLE inter-agent communication and state updates
  - PATTERN: Use Command-based routing from LangGraph docs

Task 8: Build Episode Loading System
CREATE src/mmllm/utils/episode_loader.py:
  - IMPLEMENT AiTW dataset loading and parsing
  - EXTRACT episodes with proper image/annotation processing
  - MIRROR patterns from main.py and demo.ipynb
  - ADD caching and batch processing capabilities

Task 9: Create Action Execution Engine
CREATE src/mmllm/actions/action_executor.py:
  - IMPLEMENT action validation against UI state
  - SIMULATE action execution for testing
  - GENERATE execution feedback and success metrics
  - HANDLE error cases and recovery strategies

Task 10: Build Integration Layer
MODIFY src/mmllm/main.py:
  - INTEGRATE multi-agent system with existing dataset loading
  - REPLACE single agent with multi-agent coordinator
  - ADD comprehensive logging and monitoring
  - PRESERVE existing model configuration patterns
```

### Per Task Pseudocode

```python
# Task 4: Planning Agent Implementation
class PlanningAgent:
    def __init__(self, model_provider):
        self.model = model_provider.get_model()
        self.vision_processor = ImageProcessor()
        
    def plan_actions(self, state: MultiAgentState) -> Command:
        # PATTERN: Multimodal input like agent.py but enhanced
        screenshot = state["current_screenshot"]
        ui_annotations = state["ui_annotations"]
        goal = state["goal"]
        
        # CRITICAL: Proper multimodal message format for Azure OpenAI
        messages = [
            {"role": "system", "content": f"Goal: {goal}"},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this Android interface and plan the next action"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{screenshot['data']}"
                }}
            ]}
        ]
        
        # PATTERN: Use structured output with action schemas
        structured_model = self.model.with_structured_output(ActionPlan)
        plan = structured_model.invoke(messages)
        
        return Command(
            goto="execution_agent",
            update={"plan": plan, "messages": state["messages"] + [plan_message]}
        )

# Task 7: Multi-Agent Coordinator
def build_multi_agent_graph():
    # PATTERN: StateGraph with conditional routing like LangGraph examples
    graph_builder = StateGraph(MultiAgentState)
    
    graph_builder.add_node("planning_agent", planning_node)
    graph_builder.add_node("execution_agent", execution_node) 
    graph_builder.add_node("reflection_agent", reflection_node)
    
    # CRITICAL: Use conditional edges for dynamic routing
    graph_builder.add_conditional_edges(
        "planning_agent",
        route_after_planning,  # Returns "execution_agent" or END
        {"execute": "execution_agent", "complete": END}
    )
    
    graph_builder.add_conditional_edges(
        "execution_agent", 
        route_after_execution,  # Returns "reflection_agent" or "planning_agent"
        {"reflect": "reflection_agent", "replan": "planning_agent"}
    )
    
    graph_builder.add_edge("reflection_agent", "planning_agent")
    graph_builder.add_edge(START, "planning_agent")
    
    return graph_builder.compile()
```

### Integration Points
```yaml
DATASET:
  - integration: Load AiTW episodes from TensorFlow format
  - pattern: "tf.data.TFRecordDataset with GZIP compression"
  - location: main.py and new episode_loader.py

MODEL:
  - integration: Azure OpenAI multimodal model
  - pattern: "AzureChatOpenAI with vision capabilities"
  - location: model.py (extend existing configuration)

VISION:
  - integration: TensorFlow image processing to base64
  - pattern: "tf.io.decode_raw + PIL + base64 encoding"
  - location: visualization_utils.py patterns

ACTIONS:
  - integration: Map AiTW action types to agent actions
  - pattern: "ActionType enum to Pydantic action schemas"
  - location: action_type.py mapping to new action schemas
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/mmllm/multi_agent/ --fix
ruff check src/mmllm/actions/ --fix  
ruff check src/mmllm/vision/ --fix
mypy src/mmllm/multi_agent/
mypy src/mmllm/actions/
mypy src/mmllm/vision/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE tests/test_multi_agent.py with these test cases:
def test_planning_agent_generates_valid_plan():
    """Planning agent produces valid action plans"""
    planning_agent = PlanningAgent(mock_model)
    state = create_test_state_with_screenshot()
    result = planning_agent.plan_actions(state)
    assert isinstance(result.update["plan"], ActionPlan)

def test_execution_agent_handles_all_action_types():
    """Execution agent processes all AiTW action types"""
    execution_agent = ExecutionAgent()
    for action_type in [TapAction, SwipeAction, TypeAction, NavigationAction]:
        action = create_mock_action(action_type)
        result = execution_agent.execute_action(action, mock_state)
        assert result.success or result.error_message

def test_reflection_agent_provides_feedback():
    """Reflection agent analyzes outcomes and provides insights"""
    reflection_agent = ReflectionAgent(mock_model)
    state_with_execution = create_state_with_execution_results()
    result = reflection_agent.reflect(state_with_execution)
    assert len(result.update["reflection_insights"]) > 0

def test_multi_agent_coordination():
    """Multi-agent graph routes correctly between agents"""
    graph = build_multi_agent_graph()
    initial_state = create_test_episode_state()
    results = list(graph.stream(initial_state))
    # Verify planning -> execution -> reflection flow
    assert any("planning_agent" in chunk for chunk in results)
    assert any("execution_agent" in chunk for chunk in results)

def test_aitw_episode_processing():
    """System processes real AiTW episodes correctly"""
    episode_loader = EpisodeLoader()
    episode = episode_loader.load_sample_episode()
    graph = build_multi_agent_graph()
    result = graph.invoke(episode_to_state(episode))
    assert result["current_step"] > 0
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_multi_agent.py -v
uv run pytest tests/test_actions.py -v
uv run pytest tests/test_vision.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real AiTW dataset
uv run python -m src.mmllm.main

# Expected output showing multi-agent coordination:
# Planning Agent: Analyzing screenshot for goal "turn off javascript in chrome"
# Execution Agent: Executing tap action at coordinates (0.85, 0.12)
# Reflection Agent: Action successful, UI state changed as expected
# Final result: Task completed successfully
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Multi-agent coordination works: test with sample AiTW episode
- [ ] All AiTW action types supported: tap, swipe, type, navigation
- [ ] Vision processing handles UI annotations correctly
- [ ] Feedback loops between agents function properly
- [ ] Error handling and recovery mechanisms work
- [ ] State persistence across agent transitions

## Anti-Patterns to Avoid
- ❌ Don't use single-agent patterns for multi-agent coordination
- ❌ Don't ignore AiTW coordinate normalization (0-1 range, y-x order)
- ❌ Don't skip vision processing validation - images must be properly encoded
- ❌ Don't hardcode action coordinates - use UI annotation analysis
- ❌ Don't bypass error handling in action execution
- ❌ Don't skip state validation between agent transitions
- ❌ Don't ignore structured output schema validation

---

## Quality Score: 9/10

This PRP provides comprehensive context for implementing a sophisticated multi-agent system with:
- Complete documentation and reference patterns
- Detailed task breakdown with specific implementation guidance  
- Proper validation gates and error handling
- Integration with existing codebase patterns
- Coverage of all AiTW dataset requirements

The high confidence score reflects the thorough research, detailed implementation blueprint, and comprehensive validation approach that should enable successful one-pass implementation.
