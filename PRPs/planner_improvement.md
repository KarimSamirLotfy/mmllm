name: "LangGraph Planner Agent Improvements for Episode Evaluation"
description: |
  Enhance the existing multi-agent system's planning agent to improve episode-by-episode evaluation against AiTW ground truth with comprehensive historical context and structured output comparison.

## Goal
Improve the existing LangGraph-based planning agent to enable step-by-step episode evaluation where:
- The planner gets access to all previous images in the episode history
- Each step is evaluated against ground truth actions from AiTW dataset
- Output format matches AiTW dataset structure for easy comparison
- Comprehensive episode evaluation with detailed step comparison reporting

## Why
- **Research Validation**: Enable precise evaluation of multi-agent performance against AiTW ground truth
- **Model Improvement**: Provide detailed feedback for agent performance optimization
- **Historical Context**: Leverage episode history to improve planning decisions
- **Structured Evaluation**: Standardize output format for automated comparison with dataset

## What
Enhanced planner agent functionality that includes:
- Historical image context processing for planning decisions
- Episode runner with step-by-step ground truth comparison
- Structured output matching AiTW dataset format
- Comprehensive evaluation reporting with success metrics
- Improved planning agent with multi-image context processing

### Success Criteria
- [ ] Planning agent processes all historical episode images
- [ ] Episode runner evaluates each step against ground truth
- [ ] Agent outputs match AiTW dataset action format exactly
- [ ] Comprehensive episode evaluation report generated
- [ ] Planning decisions improve with historical context
- [ ] Integration tests pass with real AiTW episodes

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/langchain-ai/langgraph
  why: Latest LangGraph Command patterns and multi-agent coordination
  critical: Command routing, state management, multi-agent orchestration

- url: https://arxiv.org/abs/2307.10088
  why: Android in the Wild paper - action format and evaluation metrics
  critical: Action matching, coordinate normalization, evaluation methodology

- file: src/mmllm/multi_agent/planning_agent.py
  why: Current planning agent implementation to extend
  critical: Vision processing patterns, structured output, error handling

- file: src/mmllm/multi_agent/state.py
  why: State management and data models for multi-agent system
  critical: MultiAgentState structure, PlanningOutput schema

- file: src/mmllm/android_in_the_wild/action_type.py
  why: Complete action type enumeration and format specification
  critical: ActionType enum values, action structure matching

- file: src/mmllm/android_in_the_wild/action_matching.py
  why: Action comparison and validation utilities
  critical: Ground truth comparison logic, coordinate normalization

- file: src/mmllm/utils/episode_loader.py
  why: Episode loading and state conversion patterns
  critical: AiTW episode processing, state initialization

- file: src/mmllm/vision/image_processor.py
  why: Image processing for multimodal models
  critical: Base64 encoding, image format handling

- file: src/mmllm/main.py
  why: Current usage patterns and integration approach
  critical: Multi-agent coordination, logging patterns
```

### Current Codebase Structure
```bash
src/
├── mmllm/
│   ├── multi_agent/               # Existing multi-agent system
│   │   ├── planning_agent.py     # Current planning agent - NEEDS ENHANCEMENT
│   │   ├── execution_agent.py    # Execution with action validation
│   │   ├── reflection_agent.py   # Outcome analysis
│   │   ├── coordinator.py        # LangGraph orchestration
│   │   └── state.py              # State management and schemas
│   ├── actions/                   # Action handling system
│   │   ├── action_schemas.py     # Pydantic models for actions
│   │   ├── action_executor.py    # Action execution simulation
│   │   └── action_validator.py   # Action validation logic
│   ├── vision/                    # Vision processing pipeline
│   │   ├── image_processor.py    # Image encoding/decoding
│   │   ├── ui_analyzer.py        # UI element analysis
│   │   └── annotation_parser.py  # AiTW annotation parsing
│   ├── utils/                     # Utilities
│   │   ├── episode_loader.py     # AiTW episode loading
│   │   └── state_manager.py      # State persistence
│   └── android_in_the_wild/       # AiTW dataset utilities
│       ├── action_matching.py    # Action comparison logic
│       ├── action_type.py        # Action type definitions
│       └── visualization_utils.py # Image/annotation processing
tests/
├── test_multi_agent.py           # Existing test patterns
└── fixtures/                     # Test data
```

### Desired Enhancements with New Files
```bash
src/
├── mmllm/
│   ├── evaluation/               # NEW: Evaluation system
│   │   ├── __init__.py
│   │   ├── episode_runner.py    # Step-by-step episode evaluation
│   │   ├── action_comparator.py # Ground truth comparison
│   │   └── evaluation_reporter.py # Comprehensive reporting
│   ├── multi_agent/
│   │   ├── enhanced_planning_agent.py # NEW: Historical context planning
│   │   └── planning_agent.py    # MODIFY: Add historical image support
│   └── utils/
│       └── episode_loader.py    # MODIFY: Add historical context loading
tests/
├── test_episode_evaluation.py   # NEW: Episode evaluation tests
└── fixtures/
    └── sample_episodes.py       # NEW: Test episode data
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: LangGraph 0.5.2 Command usage patterns
# Use Command(goto=agent_name, update=state_dict) for routing
# StateGraph requires TypedDict for state management

# CRITICAL: Azure OpenAI multimodal requires specific format
# Images must be base64 encoded with "data:image/jpeg;base64,{data}" format
# Multiple images in context require proper message structuring

# CRITICAL: AiTW dataset coordinate format
# Coordinates are normalized [0,1] in [y, x] format (not [x, y])
# Actions use ActionType enum with specific parameter requirements

# CRITICAL: AiTW action comparison requirements
# ExecutionOutput must match exact AiTW action format for comparison
# Coordinate precision should match dataset (3-4 decimal places)

# CRITICAL: TensorFlow dataset processing
# Images are encoded as raw bytes requiring tf.io.decode_raw
# UI annotations need proper reshaping from flattened arrays

# CRITICAL: Memory management for historical images
# Multiple base64 images in state can cause memory issues
# Consider image compression or selective history for long episodes

# CRITICAL: Pydantic v2 syntax for schemas
# Use Field(description="...") not Field(..., description="...")
# Optional fields need proper default values
```

## Implementation Blueprint

### Data Models and Structure

Enhanced state management for historical context and evaluation:

```python
# Enhanced state for episode evaluation
class EpisodeEvaluationState(TypedDict):
    """Enhanced state for episode-by-episode evaluation."""
    # Existing MultiAgentState fields
    goal: str
    current_step: int
    max_steps: int
    current_phase: AgentPhase
    current_image: str
    ui_annotations: List[Dict[str, Any]]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # NEW: Historical context for planning
    episode_images: List[str]  # All images from episode start
    episode_actions: List[Dict[str, Any]]  # All previous actions taken
    episode_ground_truth: List[Dict[str, Any]]  # Ground truth actions
    
    # NEW: Evaluation tracking
    step_evaluations: List[Dict[str, Any]]  # Per-step comparison results
    current_ground_truth: Optional[Dict[str, Any]]  # Current step ground truth
    
    # Enhanced planning output for evaluation
    planning_output: Optional[EnhancedPlanningOutput]

class EnhancedPlanningOutput(BaseModel):
    """Enhanced planning output matching AiTW format exactly."""
    action_type: ActionType
    coordinates: Optional[List[float]] = None  # [y, x] normalized
    lift_coordinates: Optional[List[float]] = None  # For swipes
    text_input: Optional[str] = None
    reasoning: str
    confidence: float
    
    # NEW: AiTW format compliance
    aitw_action_format: Dict[str, Any]  # Exact AiTW action structure
    historical_context_used: List[str]  # Which historical elements influenced decision

class StepEvaluationResult(BaseModel):
    """Evaluation result for a single step."""
    step_number: int
    agent_action: Dict[str, Any]
    ground_truth_action: Dict[str, Any]
    action_match: bool
    coordinate_distance: Optional[float]  # For spatial actions
    action_type_match: bool
    text_match: Optional[bool]  # For TYPE actions
    evaluation_score: float  # 0-1 score for this step
```

### List of Tasks to Complete (in order)

```yaml
Task 1: Enhance Episode Loader for Historical Context
MODIFY src/mmllm/utils/episode_loader.py:
  - ADD method load_episode_with_history()
  - EXTRACT all images and ground truth actions from episode
  - PRESERVE existing episode_to_multi_agent_state() patterns
  - ADD historical context to state initialization

Task 2: Create Enhanced Planning Agent
CREATE src/mmllm/multi_agent/enhanced_planning_agent.py:
  - INHERIT from existing PlanningAgent patterns
  - ADD multi-image processing for historical context
  - MODIFY prompt to include episode history
  - ENSURE output matches AiTW action format exactly
  - PRESERVE existing structured output patterns

Task 3: Create Episode Evaluation Runner
CREATE src/mmllm/evaluation/episode_runner.py:
  - IMPLEMENT step-by-step episode processing
  - LOAD episodes with historical context
  - COORDINATE multi-agent system for each step
  - COLLECT agent outputs vs ground truth
  - GENERATE step evaluation results

Task 4: Create Action Comparator
CREATE src/mmllm/evaluation/action_comparator.py:
  - IMPLEMENT action format comparison
  - HANDLE coordinate distance calculation
  - VALIDATE action type matching
  - SUPPORT text input comparison
  - REFERENCE existing action_matching.py patterns

Task 5: Create Evaluation Reporter
CREATE src/mmllm/evaluation/evaluation_reporter.py:
  - GENERATE comprehensive episode reports
  - CALCULATE success metrics and statistics
  - PROVIDE step-by-step comparison details
  - EXPORT results in multiple formats (JSON, markdown)

Task 6: Modify Planning Agent for Historical Context
MODIFY src/mmllm/multi_agent/planning_agent.py:
  - ADD historical image processing to plan_action()
  - MODIFY prompt construction for episode context
  - ENHANCE vision message building for multiple images
  - PRESERVE existing error handling patterns

Task 7: Create Integration Tests
CREATE tests/test_episode_evaluation.py:
  - TEST episode runner with sample episodes
  - VALIDATE action comparison accuracy
  - VERIFY historical context processing
  - ENSURE output format compliance

Task 8: Update Main Entry Point
MODIFY src/mmllm/main.py:
  - ADD episode evaluation mode
  - INTEGRATE enhanced planning agent
  - PRESERVE existing demo functionality
  - ADD evaluation result logging
```

### Per Task Pseudocode

```python
# Task 1: Enhanced Episode Loader
class EpisodeLoader:
    def load_episode_with_history(self, episode_examples: List[tf.train.Example]) -> Dict[str, Any]:
        """Load episode with full historical context."""
        # PATTERN: Process all images in episode
        episode_images = []
        ground_truth_actions = []
        
        for step, example in enumerate(episode_examples):
            # CRITICAL: Use existing image processing patterns
            image_array = self.image_processor.decode_aitw_image(example)
            base64_image = self.image_processor.encode_image_for_model(image_array)
            episode_images.append(base64_image)
            
            # CRITICAL: Extract ground truth action in AiTW format
            action_data = self.annotation_parser.extract_action_data(example)
            ground_truth_actions.append(action_data)
        
        return {
            "episode_images": episode_images,
            "ground_truth_actions": ground_truth_actions,
            "episode_length": len(episode_examples)
        }

# Task 2: Enhanced Planning Agent
class EnhancedPlanningAgent(PlanningAgent):
    def plan_action_with_history(self, state: EpisodeEvaluationState) -> Command:
        """Plan action using full episode history."""
        # PATTERN: Build multimodal messages with historical context
        messages = [
            SystemMessage(content=self._build_historical_context_prompt(state)),
            HumanMessage(content=[
                {"type": "text", "text": f"Goal: {state['goal']}\nStep {state['current_step']}"},
                # CRITICAL: Include current image
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['current_image']}"}},
                # NEW: Include previous images for context
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} 
                  for img in state['episode_images'][-3:]]  # Last 3 images for context
            ])
        ]
        
        # PATTERN: Use structured output with enhanced schema
        planning_output = self._get_enhanced_planning_decision(messages, state)
        
        return Command(
            goto="execution_agent",
            update={"planning_output": planning_output}
        )

# Task 3: Episode Runner
class EpisodeRunner:
    def run_episode_evaluation(self, episode_data: List[tf.train.Example]) -> Dict[str, Any]:
        """Run step-by-step episode evaluation."""
        # PATTERN: Initialize multi-agent coordinator
        coordinator = MultiAgentCoordinator()
        step_evaluations = []
        
        for step_idx in range(len(episode_data)):
            # CRITICAL: Build state with historical context
            state = self._build_state_with_history(episode_data, step_idx)
            
            # Execute multi-agent system for this step
            result = coordinator.run(state)
            
            # Compare agent output with ground truth
            evaluation = self.action_comparator.compare_actions(
                result["execution_output"].aitw_action_format,
                state["current_ground_truth"]
            )
            
            step_evaluations.append(evaluation)
        
        return {
            "episode_id": episode_data[0].features.feature['episode_id'].bytes_list.value[0].decode('utf-8'),
            "total_steps": len(episode_data),
            "step_evaluations": step_evaluations,
            "overall_success_rate": sum(e["evaluation_score"] for e in step_evaluations) / len(step_evaluations)
        }
```

### Integration Points
```yaml
DATASET:
  - pattern: Load complete episodes with tf.data.TFRecordDataset
  - enhancement: Process all episode steps for historical context
  - location: utils/episode_loader.py enhancements

MODEL:
  - pattern: Azure OpenAI multimodal with multiple images
  - enhancement: Support historical image context in planning
  - location: multi_agent/enhanced_planning_agent.py

ACTIONS:
  - pattern: Map agent outputs to exact AiTW action format
  - enhancement: Ensure format compliance for comparison
  - location: evaluation/action_comparator.py

EVALUATION:
  - pattern: Step-by-step comparison with ground truth
  - enhancement: Comprehensive metrics and reporting
  - location: evaluation/episode_runner.py
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/mmllm/evaluation/ --fix
ruff check src/mmllm/multi_agent/ --fix
mypy src/mmllm/evaluation/
mypy src/mmllm/multi_agent/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE tests/test_episode_evaluation.py with these test cases:
def test_enhanced_planning_agent_with_history():
    """Enhanced planning agent uses historical context"""
    agent = EnhancedPlanningAgent()
    state = create_test_state_with_history()
    result = agent.plan_action_with_history(state)
    assert "episode_images" in result.update
    assert len(result.update["planning_output"].historical_context_used) > 0

def test_episode_runner_step_by_step():
    """Episode runner evaluates each step correctly"""
    runner = EpisodeRunner()
    mock_episode = load_mock_episode()
    result = runner.run_episode_evaluation(mock_episode)
    assert "step_evaluations" in result
    assert len(result["step_evaluations"]) == len(mock_episode)

def test_action_comparator_aitw_format():
    """Action comparator handles AiTW format correctly"""
    comparator = ActionComparator()
    agent_action = create_mock_agent_action()
    ground_truth = create_mock_ground_truth()
    result = comparator.compare_actions(agent_action, ground_truth)
    assert "action_match" in result
    assert "evaluation_score" in result

def test_historical_context_processing():
    """Historical context improves planning decisions"""
    loader = EpisodeLoader()
    episode_data = loader.load_episode_with_history(mock_episodes)
    assert len(episode_data["episode_images"]) > 1
    assert len(episode_data["ground_truth_actions"]) > 1
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_episode_evaluation.py -v
uv run pytest tests/test_multi_agent.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real AiTW dataset
uv run python -m src.mmllm.evaluation.episode_runner --dataset google_apps --episodes 3

# Expected output:
# Episode 1/3: episode_12345
# Step 1/5: Planning with historical context...
# Step 1/5: Agent action: DUAL_POINT at [0.123, 0.456]
# Step 1/5: Ground truth: DUAL_POINT at [0.125, 0.458]
# Step 1/5: Evaluation score: 0.95
# ...
# Episode evaluation complete: 85% success rate

# Comprehensive evaluation report
uv run python -c "
from src.mmllm.evaluation.episode_runner import EpisodeRunner
from src.mmllm.utils.episode_loader import EpisodeLoader

loader = EpisodeLoader()
runner = EpisodeRunner()
dataset = loader.load_dataset('google_apps')
episode = loader.get_episode(dataset)
result = runner.run_episode_evaluation(episode)
print(f'Success rate: {result[\"overall_success_rate\"]:.2%}')
"
```

## Final Validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Episode runner works with real AiTW data
- [ ] Action outputs match AiTW format exactly
- [ ] Historical context improves planning decisions
- [ ] Evaluation reports are comprehensive and accurate
- [ ] Memory usage is reasonable for long episodes
- [ ] Integration with existing multi-agent system preserved

## Anti-Patterns to Avoid
- ❌ Don't break existing multi-agent coordination patterns
- ❌ Don't ignore AiTW coordinate format ([y, x] not [x, y])
- ❌ Don't skip action format validation - must match exactly
- ❌ Don't load entire episodes into memory without compression
- ❌ Don't bypass existing error handling in multi-agent system
- ❌ Don't ignore historical context in planning decisions
- ❌ Don't create new state schemas without TypedDict validation
- ❌ Don't skip ground truth comparison for any action type

---

## Quality Score: 9/10

This PRP provides comprehensive context for enhancing the planning agent with:
- Complete understanding of existing multi-agent architecture
- Detailed historical context processing requirements
- Exact AiTW format compliance specifications
- Comprehensive evaluation framework design
- Proper validation gates and testing approach
- Integration with existing codebase patterns
- Coverage of all evaluation requirements from the feature spec

The high confidence score reflects thorough research, detailed task breakdown, and comprehensive validation approach that should enable successful implementation while preserving existing functionality.
