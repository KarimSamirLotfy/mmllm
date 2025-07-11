from PIL import Image, ImageDraw, ImageFont

def overlay_grid_with_anchors(
    image_path,
    output_path,
    grid_spacing=100,
    grid_color=(255, 0, 0),
    alpha=0.3,
    anchor_color=(255, 255, 0),
    anchor_radius=8,
    anchor_labels=True
):
    """
    Overlay a 100px grid with semi-transparent lines and visible anchor point markers on a screenshot.

    Parameters:
    - image_path: Path to the input screenshot
    - output_path: Path to save the result
    - grid_spacing: Spacing between grid lines (default 100)
    - grid_color: RGB tuple for the grid lines
    - alpha: Transparency of grid lines (0.0 to 1.0)
    - anchor_color: RGB color for marker circles and text
    - anchor_radius: Radius of anchor point circle markers
    - anchor_labels: Whether to draw coordinate labels
    """
    # Load screenshot
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    # Prepare transparent grid layer
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba_color = grid_color + (int(255 * alpha),)

    # Draw grid lines
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=rgba_color, width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=rgba_color, width=1)

    # Define anchor points
    anchors = {
        "(0,0)": (0, 0),
        "(540,1200)": (540, 1200),
        f"({width},0)": (width - 1, 0),
        f"(0,{height})": (0, height - 1),
        f"({width},{height})": (width - 1, height - 1)
    }

    # Draw anchor circles and labels
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for label, (x, y) in anchors.items():
        # Draw circle
        draw.ellipse([
            (x - anchor_radius, y - anchor_radius),
            (x + anchor_radius, y + anchor_radius)
        ], outline=anchor_color + (255,), width=2)

        # Draw label text
        if anchor_labels:
            draw.text((x + 10, y + 5), label, fill=anchor_color + (255,), font=font)

    # Combine screenshot and overlay
    combined = Image.alpha_composite(image, overlay)
    combined.save(output_path)
    print(f"Saved: {output_path}")

def add_grid_with_anchors(
    image,
    grid_spacing_percent=0.1,
    grid_color=(255, 0, 0),
    alpha=0.3,
    anchor_color=(255, 255, 0),
    anchor_radius=8,
    anchor_labels=True
):
    """
    Add a grid overlay with anchor point markers to a PIL image.

    Parameters:
    - image: PIL Image object
    - grid_spacing_percent: Grid spacing as percentage of image dimensions (default 0.1 = 10%)
    - grid_color: RGB tuple for the grid lines
    - alpha: Transparency of grid lines (0.0 to 1.0)
    - anchor_color: RGB color for marker circles and text
    - anchor_radius: Radius of anchor point circle markers
    - anchor_labels: Whether to draw coordinate labels

    Returns:
    - PIL Image with grid and anchors overlaid
    """
    # Ensure image is in RGBA mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    width, height = image.size

    # Calculate dynamic grid spacing based on image dimensions
    grid_spacing_x = int(width * grid_spacing_percent)
    grid_spacing_y = int(height * grid_spacing_percent)

    # Prepare transparent grid layer
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba_color = grid_color + (int(255 * alpha),)

    # Draw grid lines
    for x in range(0, width, grid_spacing_x):
        draw.line([(x, 0), (x, height)], fill=rgba_color, width=1)
    for y in range(0, height, grid_spacing_y):
        draw.line([(0, y), (width, y)], fill=rgba_color, width=1)

    # Calculate anchor points dynamically from image dimensions
    center_x = width // 2
    center_y = height // 2
    
    anchors = {
        "(0,0)": (0, 0),
        f"({center_x},{center_y})": (center_x, center_y),
        f"({width-1},0)": (width - 1, 0),
        f"(0,{height-1})": (0, height - 1),
        f"({width-1},{height-1})": (width - 1, height - 1)
    }

    # Draw anchor circles and labels
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for label, (x, y) in anchors.items():
        # Draw circle
        draw.ellipse([
            (x - anchor_radius, y - anchor_radius),
            (x + anchor_radius, y + anchor_radius)
        ], outline=anchor_color + (255,), width=2)

        # Draw label text
        if anchor_labels:
            draw.text((x + 10, y + 5), label, fill=anchor_color + (255,), font=font)

    # Combine original image and overlay
    combined = Image.alpha_composite(image, overlay)
    return combined