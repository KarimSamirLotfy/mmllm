import cv2
from matplotlib import pyplot as plt
import pytesseract
import json
from pytesseract import Output
import numpy as np
from PIL import Image, ImageOps

def invert_pil_image(image: Image.Image) -> Image.Image:
    
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_rgb = ImageOps.invert(rgb_image)
        return Image.merge('RGBA', (*inverted_rgb.split(), a))
    elif image.mode == 'RGB':
        return ImageOps.invert(image)
    else:
        # For grayscale or others, convert to L first
        return ImageOps.invert(image.convert('L'))


def extract_ui_elements(pil_image: Image.Image, use_preprocess: bool = True, normalize: bool = False) -> (list, np.ndarray):
    """Extract UI elements via Tesseract OCR, draw boxes, and emit JSON list."""
    # Choose image for OCR
    if use_preprocess:
        ocr_image = invert_pil_image(pil_image)
    else:
        ocr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # Run Tesseract with data output
    data = pytesseract.image_to_data(
        ocr_image,
        output_type=Output.DICT,
        config='--psm 6'
    )

    # For visualization, convert to BGR color
    vis = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # width and height of the image
    width, height = vis.shape[1], vis.shape[0]
    elements = []

    n = len(data['level'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text and conf > 50:
            x, y, w, h = (
                data['left'][i], data['top'][i],
                data['width'][i], data['height'][i]
            )
            cx, cy = x + w // 2, y + h // 2
            # Draw box & label
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
            if normalize:
                # Normalize coordinates to [0, 1] range
                x /= width
                y /= height
                w /= width
                h /= height
                cx /= width
                cy /= height
            elements.append({
                'text': text,
                'confidence': conf,
                'bounding_box': {'left': x, 'top': y, 'width': w, 'height': h},
                'click_point': {'x': cx, 'y': cy}
            })


    return elements, vis


if __name__ == '__main__':
    # Load screenshot
    screenshot = Image.open('/home/kiko/mmllm/episode_plot.png')
    elems, vis = extract_ui_elements(screenshot, use_preprocess=True)
    cv2.imwrite('ocr.png', vis)
    print(json.dumps(elems, indent=2))
    print(f"Extracted {len(elems)} elements. JSON and bb.png saved.")
