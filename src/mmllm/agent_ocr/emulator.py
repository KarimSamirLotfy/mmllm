import json
import time
from matplotlib.pyplot import step
import tensorflow as tf
from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent
from emulator_bridge import run_adb_shell, get_screen, take_screenshot
from mmllm.utils.visualization import plot_episode
from PIL import Image
import io

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
import base64
from io import BytesIO

import json
import re

def convert_dataset_to_emulator_action(dataset_action, screen_width=1080, screen_height=2400):
    # If input is a list, process each action and return a list of commands
    action_type = dataset_action.get("action_type")

    if action_type == 3:  # TYPE
        return f'input type "{dataset_action["text"]}"'

    elif action_type == 4:  # DUAL_POINT
        x1, y1 = dataset_action["coordinates"]
        x2, y2 = dataset_action["lift_coordinates"]

        px1, py1 = int(x1 * screen_width), int(y1 * screen_height)
        px2, py2 = int(x2 * screen_width), int(y2 * screen_height)

        if (px1, py1) == (px2, py2):
            return f"input tap {px1} {py1}"
        else:
            return f"input swipe {px1} {py1} {px2} {py2}"

    elif action_type == 5:
        return "input keyevent 4"

    elif action_type == 6:
        return "input keyevent 3"

    elif action_type == 7:
        return "input keyevent 6"

    elif action_type == 10:
        return "# Task complete"

    elif action_type == 11:
        return "# Task impossible"

    elif action_type in (0, 1, 2):  # Assume tap-like click
        x1, y1 = dataset_action["coordinates"]
        px1, py1 = int(x1 * screen_width), int(y1 * screen_height)
        return f"input tap {px1} {py1}"

    else:
        return "# Unknown or unsupported action"

def pil_image_to_base64(pil_image, format="PNG", include_prefix=True):
    """Convert a PIL Image to base64 string."""
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    if include_prefix:
        return f"data:image/{format.lower()};base64,{base64_str}"
    else:
        return base64_str


agent = SimpleOCRAgent(ocr_module=True)

# input from commandline for the task
while True:
    task_description = input("Enter the task description (or 'exit' to quit): ").strip()
    if task_description.lower() == 'exit':
        print("Exiting...")
        break
    thread_id = "0"
    step = 0
    while True:
        time.sleep(3)  # Simulate waiting for the next step
        elements, local_path, (width, height) = take_screenshot(step)
        image = Image.open(local_path)

        # Resize image to fit the model input size
        #image = image.resize((540, 1200))

        #(width, height) = image.size  # Get image size

        image_base64 = pil_image_to_base64(image, include_prefix=False)

        result = agent.run_step(
            image=image_base64,  # In real usage, this would be a PIL Image
            ocr_text="",
            ui_description=elements,
            goal=task_description,
            thread_id=thread_id
        )
        print(f"Step {step} Action:")
        # Print the action decided by the agent
        command = result.get('action', {})["actions"]
        print(json.dumps(command, indent=2))   
        # Execute the command in the emulator
        for action in command:
            emulator_command = convert_dataset_to_emulator_action(action, width, height)
            if action.get('task_done', True):
                print("Task completed!")
                break
            else:
                if emulator_command:
                    print(f"Executing command: {emulator_command}")
                    run_adb_shell(emulator_command)
                else:
                    print("No valid emulator command generated.")

