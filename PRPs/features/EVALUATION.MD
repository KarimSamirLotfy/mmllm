## FEATURE:

Create a Benchmarking pipeline.
# Requirements
1. Runs the model on the dataset.
2. keeps track of the model preformance and logs it
3. creates a csv of the runs
4. creates a report json file
5. Has a visualsation script that shows the graphs needed
6. is configurable, allows easily to run on subset of datasets, specifc datasets, and allow configuring the run. saves the configuraiton to make sure we can reproduce the results

* It needs to run the agent on the AiTW datasets. allowing configuration for the run like which datasets specifcally. 
* it should create evaluation metrics based on action matching. 
1. accuracy for step. 
2. success rate. if a model does the first mistake on the 4th step out of 10 then it gets a 40%
3. also show the F1 score, recall and other metrics 
4. save the entire run in the csv, save all fields you will need for the visulastion 
5. the scrip should show, model charts on differet datasets, different metrics.
6. the model has ocr_module variable as a configurable paramter to compare.  

### References & Implementation Pointers

- See `/home/kiko/mmllm/src/mmllm/agent_ocr/aitw.py` , it has everything needed, it has the dataloader, the matching function, the ploting function and the model. 

### Documentation

- Libraries: `langgraph`, `matplotlib`
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

- imports: do not use relative imports all imports should be done mmllm.<path to file> as this is a package. so from mmllm.agent_orc.simple_ocr_agent import SimpleOcrAgent This is how all imports should work