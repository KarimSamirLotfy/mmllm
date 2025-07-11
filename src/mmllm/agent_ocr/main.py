import subprocess
from pathlib import Path
from PIL import Image
from time import sleep
from mmllm.agent_ocr.process import overlay_grid_with_anchors
from mmllm.agent_ocr.extract import extract_ui_elements
import cv2
import json


# Define $ADB path for WSL
ADB = "/mnt/c/Users/krono/AppData/Local/Android/Sdk/platform-tools/adb.exe"

def run_adb_shell(command: str):
    """Run a shell command via ADB."""
    subprocess.run([ADB, "shell"] + command.strip().split(), check=False)

def take_screenshot(index=1):
    """Take a screenshot, return local image path and (width, height)."""
    filename = f"ss/screen.png"
    filename2 = f"ss/screen_p.jpg"

    local_path = Path.cwd() / filename
    local_path2 = Path.cwd() / filename2

    remote_path = "/sdcard/screen.png"

    # Take and pull screenshot
    print("📸 Capturing screenshot...")
    subprocess.run([ADB, "shell", "screencap", "-p", remote_path], check=False)
    subprocess.run([ADB, "pull", remote_path, str(local_path)], check=False)
    subprocess.run([ADB, "shell", "rm", remote_path], check=False)

    # Load image and get size
    image = Image.open(local_path)
    elements, processed_image = extract_ui_elements(image)

    cv2.imwrite(str(local_path2), processed_image)  # Save the processed image with bounding boxes
    with open('elements.json', 'w') as f:
        json.dump(elements, f, indent=2)

    width, height = image.size
    print(f"🖼️ Screenshot size: {width} x {height} pixels")
    image.show()

    return local_path, (width, height)

def main():
    print("🎮 ADB Interactive Shell with Screenshot Viewer (type 'exit' to quit)\n")
    index = 1
    while True:
        image_path, (width, height) = take_screenshot(index)
        overlay_grid_with_anchors(
            image_path=image_path,
            output_path=image_path,
            grid_spacing=60,
            grid_color=(255, 0, 0),
            alpha=0.3,
            anchor_color=(255, 255, 0),
            anchor_radius=2,
            anchor_labels=True
        )
        index += 1

        cmd = input("\n📥 Enter ADB shell command (e.g., 'input tap 300 400'): ").strip()
        if cmd.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        if cmd:
            print(f"▶️ $ADB shell {cmd}")
            run_adb_shell(cmd)

        sleep(2)

if __name__ == "__main__":
    main()
