
import base64
from io import BytesIO

import numpy as np
from mmllm.android_in_the_wild.visualization_utils import _plot_action, _plot_dual_point
from matplotlib import pyplot as plt
from PIL import Image
def base64_to_image(base64_string):
    """Convert base64 string to PIL Image or numpy array for matplotlib."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image from bytes
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed (matplotlib prefers RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array for matplotlib
        numpy_image = np.array(pil_image)
        
        return numpy_image
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None
def base64_to_pil_image(base64_string):
    """Convert base64 string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image from bytes
        pil_image = Image.open(BytesIO(image_bytes))
        
        return pil_image
    except Exception as e:
        print(f"Error converting base64 to PIL image: {e}")
        return None
def plot_example(
        image,
        touch_x,
        touch_y,
        lift_x,
        lift_y,
        action_type,
        ax=None,
    ):
    
    """Plots the example's action on the given matplotlib axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    img = base64_to_image(image)
    ax.imshow(img)
    ax = _plot_action(
            ex_action_type=action_type,
            screen_height= img.shape[0],
            screen_width= img.shape[1],
            touch_x= touch_x,
            touch_y= touch_y,
            lift_x= lift_x,
            lift_y= lift_y,
            action_text=None,  # Assuming no text for this example
            ax=ax,
    )
    return ax
def plot_episode(
    episode,
    show_actions=True,
    model_actions={},
):
    """Plots a visualization of the given episode.

    Args:
        episode: A dictionary containing episode data with keys like 'episode_images', 
                'ground_truth_actions', 'goal', etc.
        show_actions: Whether to plot the actions for each episode step.
        model_actions: Dictionary with model actions for each step (same structure as ground_truth_actions)
    """
    _NUM_EXS_PER_ROW = 4  # Number of examples per row
    
    # Get episode length from the images
    ep_len = len(episode['episode_images'])
    
    # Calculate grid dimensions
    n_cols = min(ep_len, _NUM_EXS_PER_ROW)
    n_rows = 1 + (ep_len - 1) // n_cols
    
    # Create subplot grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 7))
    
    # Handle single subplot case
    if ep_len == 1:
        axs = [axs]
    elif n_rows == 1:
        axs = axs if hasattr(axs, '__len__') else [axs]
    else:
        axs = axs.flatten()
    
    goal = episode.get('goal', 'No goal specified')
    
    # Plot each step
    for i in range(ep_len):
        image = episode['episode_images'][i]
        ground_truth_action = episode.get('ground_truth_actions', [{}])[i] if i < len(episode.get('ground_truth_actions', [])) else {}
        model_action = model_actions.get(i, {}) if isinstance(model_actions, dict) else (model_actions[i] if i < len(model_actions) else {})
        
        # Convert image for display
        img = base64_to_image(image) if isinstance(image, str) else image
        if img is not None:
            axs[i].imshow(img)
        
        if show_actions:
            # Plot ground truth action in default color (usually red/blue)
            if ground_truth_action :
                touch_x, touch_y = ground_truth_action.get('coordinates', (0, 0))
                lift_x, lift_y = ground_truth_action.get('lift_coordinates', (touch_x, touch_y))
                action_type = ground_truth_action.get('action_type', 0)
                
                _plot_action(
                    ex_action_type=action_type,
                    screen_height=img.shape[0],
                    screen_width=img.shape[1],
                    touch_x=touch_x,
                    touch_y=touch_y,
                    lift_x=lift_x,
                    lift_y=lift_y,
                    action_text=ground_truth_action.get('text', None),
                    ax=axs[i],
                )
            
            # Plot model action in green color
            if model_action and 'action_type' in model_action:
                _plot_action(
                    ex_action_type=model_action['action_type'],
                    screen_height=img.shape[0],
                    screen_width=img.shape[1],
                    touch_x=model_action.get('coordinates', (0, 0))[0],
                    touch_y=model_action.get('coordinates', (0, 0))[1],
                    lift_x=model_action.get('lift_coordinates', (0, 0))[0],
                    lift_y=model_action.get('lift_coordinates', (0, 0))[1],
                    action_text=model_action.get('text', None),
                    ax=axs[i],
                    color='green',  # Use green for model actions
                )
            # if model_action and 'action_type' in model_action and model_action['action_type'] == 4:
            #     model_touch_x, model_touch_y = model_action.get('coordinates', (0, 0))
            #     model_lift_x, model_lift_y = model_action.get('lift_coordinates', (model_touch_x, model_touch_y))
            #     model_action_type = model_action.get('action_type', 0)
                
            #     # Plot model action with green color
            #     axs[i] =  _plot_dual_point(
            #                 touch_x=model_touch_x,
            #                 touch_y=model_touch_y,
            #                 lift_x=model_lift_x,
            #                 lift_y=model_lift_y,
            #                 screen_height=img.shape[0],
            #                 screen_width=img.shape[1],
            #                 ax=axs[i],
            #                 color='green',
            #             )


        
        # Add step title with action comparison
        title = f'Step {i + 1}'
        if show_actions and ground_truth_action and model_action:
            gt_coords = ground_truth_action.get('coordinates', [0, 0])  
            model_coords = model_action.get('coordinates', [0, 0])
            if len(model_coords) == 0:
                model_coords = [0, 0]
            # Calculate distance between actions
            distance = np.sqrt((gt_coords[0] - model_coords[0])**2 + (gt_coords[1] - model_coords[1])**2)
            title += f'\nDist: {distance:.3f}'
        
        # axs[i].set_title(title, fontsize=12)
        # axs[i].axis('off')  # Remove axis ticks
    
    # Remove unused subplots
    for j in range(ep_len, len(axs)):
        if j < len(axs):
            axs[j].remove()
    
    # Add legend
    if show_actions:
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = []
        if any(episode.get('ground_truth_actions', [])):
            legend_elements.append(Patch(facecolor='red', label='Ground Truth'))
        if model_actions:
            legend_elements.append(Patch(facecolor='green', label='Model Prediction'))
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    # Add overall title
    fig.suptitle(
        goal,
        size=16,
        y=0.98,
    )
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('episode_plot.png', bbox_inches='tight', dpi=150)
    
    return fig, axs

def _plot_action_colored(
    ex_action_type,
    screen_height,
    screen_width,
    touch_x,
    touch_y,
    lift_x,
    lift_y,
    action_text=None,
    ax=None,
    color='green',
    fliped_xy=True,
):
    """Plot action with custom color. Modified version of _plot_action."""
    if fliped_xy:
        temp = touch_x
        touch_x = touch_y
        touch_y = temp
        temp = lift_x
        lift_x = lift_y
        lift_y = temp
    # Convert normalized coordinates to pixel coordinates
    touch_x_px = touch_x * screen_width
    touch_y_px = touch_y * screen_height
    lift_x_px = lift_x * screen_width
    lift_y_px = lift_y * screen_height
    
    # Plot based on action type
    if ex_action_type == 1:  # tap
        ax.scatter(touch_x_px, touch_y_px, c=color, s=100, marker='o', alpha=0.7)
        ax.annotate('TAP', (touch_x_px, touch_y_px), xytext=(5, 5), 
                   textcoords='offset points', color=color, fontweight='bold')
    
    elif ex_action_type == 2:  # long_press
        ax.scatter(touch_x_px, touch_y_px, c=color, s=150, marker='s', alpha=0.7)
        ax.annotate('LONG PRESS', (touch_x_px, touch_y_px), xytext=(5, 5),
                   textcoords='offset points', color=color, fontweight='bold')
    
    elif ex_action_type in [3, 4]:  # swipe or drag
        ax.annotate('', xy=(lift_x_px, lift_y_px), xytext=(touch_x_px, touch_y_px),
                   arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.7))
        action_name = 'SWIPE' if ex_action_type == 3 else 'DRAG'
        ax.annotate(action_name, (touch_x_px, touch_y_px), xytext=(5, 5),
                   textcoords='offset points', color=color, fontweight='bold')
    
    elif ex_action_type == 5:  # type_text
        ax.scatter(touch_x_px, touch_y_px, c=color, s=100, marker='^', alpha=0.7)
        text_label = f'TYPE: {action_text}' if action_text else 'TYPE'
        ax.annotate(text_label, (touch_x_px, touch_y_px), xytext=(5, 5),
                   textcoords='offset points', color=color, fontweight='bold')
    
    return ax