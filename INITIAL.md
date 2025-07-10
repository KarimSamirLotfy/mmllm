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

**Key Multimodal Usage:**
- Multimodal input (vision, OCR, UI) is used in the Planning and Reflection phases.
- Execution phase uses current GUI state (vision/OCR) for real-time action and feedback.


## EXAMPLES:

mmllm/src/mmllm/main.py 
Has a Full implementaion of a simpler agent.

src/mmllm/android_in_the_wild/demo.ipynb
has the offical AiTW Demo. 

src/mmllm/agent.py
Has the agent


## DOCUMENTATION:

langgraph (python libary)
langchain (python libary)
AiTW (folder android_in_the_wild)

## OTHER CONSIDERATIONS:

Use Context7 MCP server to search for most recent documentaion on any libary