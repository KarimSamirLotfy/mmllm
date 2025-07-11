## FEATURE:

Edit this langgraph MultiModal agent to improve it. 

## Agent Requirements for MMLLM-based Vision-Centric Multi-Agent System

### 1. Planning Phase
- **Inputs:** Multimodal (vision, OCR, structured UI, icon context)
- **Tasks:**
  - gets to look at all the past images of the episode.
  - gets a current_state which is the image of the current state
  - Outputs in written form. what the agent should do. like click on X or swipe from left. in text form
- **Output:** Action Plan for the Next Step. Text Form

### 2. Execution Phase
- **Inputs:** Takes the current image and the Action plan
- **Tasks:**
  - Chooses the action to be taken and the corrdinates and all configuration to execute correctly.

### 3. Reflection Phase
- **Inputs:** Gets the current state 
- **Tasks:**
  - Assess if the task is finished or not

### 4. Multi-Agent Compatibility
- Each phase can be handled by separate agents (planning, execution, reflection).
- Agents communicate via feedback loops, sharing multimodal context.

### Integration with AiTW
* To evaluate how the model is doing. create a demo episode runner. the model gets the episode 1 step at a time. we evaluate the answer it gives on each step. 

* on each step the previous steps are given to the planner as if it was able to execute them just like the ground truth. 

* it gives the execution agent the plan, the exceution agent looks at the plan executes the next step. then the next step is compared with the true next step in the dataset. make sure the ouput of the exceution agent is the same format as the output for the dataset. this way the comparison is easy

* at the end print the episode data, each step and what the true action was and the agent action. 
---

### Markdown Graph

```mermaid
flowchart TD
    A[Planning Phase<br/>(Multimodal LLM, UI, previous steps)] --> B[Execution Phase<br/>(Vision Input, plan)]
    B --> C[Reflection Phase<br/>(Vision-Grounded, Feedback)]
    C -->|Feedback Loop| A
    subgraph Agents
        A
        B
        C
    end
```

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

- imports: do not use relative imports all imports should be done mmllm.<path to file> as this is a package.