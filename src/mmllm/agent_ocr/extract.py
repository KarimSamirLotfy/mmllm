import cv2
import pytesseract
import json
from pytesseract import Output
import numpy as np
from PIL import Image


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to grayscale and apply adaptive thresholding."""
    # Convert PIL to OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Convert to gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Reduce noise and smooth
    blur = cv2.medianBlur(gray, 3)
    # Adaptive threshold for crisp text
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 9
    )
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Invert back for OCR
    processed = cv2.bitwise_not(cleaned)
    return processed


def extract_ui_elements(pil_image: Image.Image, use_preprocess: bool = True):
    """Extract UI elements via Tesseract OCR, draw boxes, and emit JSON list."""
    # Choose image for OCR
    #if use_preprocess:
    #    ocr_image = preprocess_image(pil_image)
    #else:
    #    ocr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # Run Tesseract with data output
    data = pytesseract.image_to_data(
        pil_image,
        output_type=Output.DICT,
        config='--psm 6'
    )

    # For visualization, convert to BGR color
    vis = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
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
            elements.append({
                'text': text,
                'confidence': conf,
                'bounding_box': {'left': x, 'top': y, 'width': w, 'height': h},
                'click_point': {'x': cx, 'y': cy}
            })
            # Draw box & label
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

    return elements, vis


if __name__ == '__main__':
    # Load screenshot
    screenshot = Image.open('screenshot.png')
    elems = extract_ui_elements(screenshot, use_preprocess=True)
    print(f"Extracted {len(elems)} elements. JSON and bb.png saved.")
