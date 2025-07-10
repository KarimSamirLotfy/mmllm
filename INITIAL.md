## FEATURE:

Create a langgraph multimodal agent. the agent should run on the AiTW (android in the wild dataset). 

The Agent should have this structure. 

## Agent Requirements for MMLLM-based Vision-Centric Multi-Agent System

### 1. Planning Phase
- **Inputs:** Multimodal (vision, OCR, structured UI, icon context)
- **Tasks:**
  - Interpret GUI using multimodal LLMs.
  - Plan actions using both visual and textual cues.
  - Maintain memory of past actions for chain-of-action reasoning.
- **Output:** Action plan for execution agents.

### 2. Execution Phase
- **Inputs:** Action plan, current GUI state (vision/OCR)
- **Tasks:**
  - Execute planned actions step-by-step.
  - Support stateful interaction (e.g., focus, type, recover from errors).
  - Provide feedback to planning and reflection phases.

### 3. Reflection Phase
- **Inputs:** Execution feedback, vision/OCR, action history
- **Tasks:**
  - Analyze outcomes and errors.
  - Adjust future plans for efficiency and robustness.
  - Enable vision-grounded planning improvements.

### 4. Multi-Agent Compatibility
- Each phase can be handled by separate agents (planning, execution, reflection).
- Agents communicate via feedback loops, sharing multimodal context.

---

### Markdown Graph

```mermaid
flowchart TD
    A[Planning Phase<br/>(Multimodal LLM, OCR, UI)] --> B[Execution Phase<br/>(Stateful, Vision Input)]
    B --> C[Reflection Phase<br/>(Vision-Grounded, Feedback)]
    C -->|Feedback Loop| A
    subgraph Agents
        A
        B
        C
    end
```

---

**Multimodal Integration:**
- Planning and reflection phases leverage vision, OCR, and UI context.
- Execution phase operates on real-time GUI state and provides actionable feedback.

---

### References & Implementation Pointers

- See `mmllm/src/mmllm/main.py` for a simple agent implementation.
- The official AiTW demo is in `src/mmllm/android_in_the_wild/demo.ipynb`.
- Core agent logic is in `src/mmllm/agent.py`.

### Documentation

- Libraries: `langgraph`, `langchain`
- Dataset: AiTW (`android_in_the_wild` folder)
- Use Context7 MCP server for up-to-date library documentation.

### Development Notes

- Project uses `uv` for installation; import modules using relative paths (e.g., `mmllm.android_in_the_wild.visualization_utils.py`).
- Do not use `print` statements; use the configured logger:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```
- Logging is set up in `src/mmllm/__init__.py`.
