import langgraph
from mmllm.agent import run_agent
from mmllm.multi_agent import MultiAgentCoordinator
from mmllm.utils import EpisodeLoader

from mmllm.android_in_the_wild import visualization_utils
import tensorflow as tf
import base64
import io
from PIL import Image

from mmllm.utils.prints import print_result

def get_dataset():
    dataset_name = 'google_apps'  #@param ["general", "google_apps", "install", "single", "web_shopping"]

    dataset_directories = {
        'general': 'gs://gresearch/android-in-the-wild/general/*',
        'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
        'install': 'gs://gresearch/android-in-the-wild/install/*',
        'single': 'gs://gresearch/android-in-the-wild/single/*',
        'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
    }
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
    return raw_dataset

def get_episode(dataset):
  """Grabs the first complete episode."""
  episode = []
  episode_id = None
  for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id
      episode.append(ex)
    elif ep_id == episode_id:
      episode.append(ex)
    else:
      break
  return episode
def main():
    print("=== Multi-Agent Android in the Wild Demo ===")
    
    # Initialize episode loader and coordinator
    episode_loader = EpisodeLoader()
    coordinator = MultiAgentCoordinator()
    
    try:
        # Try to load real episode
        print("Loading episode from dataset...")
        initial_state = episode_loader.get_sample_episode_state('google_apps')
        print(f"Loaded episode: {initial_state['episode_id']}")
        print(f"Goal: {initial_state['goal']}")
        
    except Exception as e:
        print(f"Could not load real episode ({e}), using mock data...")
        initial_state = episode_loader._create_mock_state()
        print(f"Using mock episode: {initial_state['episode_id']}")
        print(f"Goal: {initial_state['goal']}")
    
    # Run multi-agent system
    print("\n=== Starting Multi-Agent Execution ===")
    try:
        final_state = coordinator.run(initial_state)
        
        print("\n=== Multi-Agent Results ===")
        print(f"Final phase: {final_state.get('current_phase', 'unknown')}")
        print(f"Steps completed: {final_state.get('current_step', 0)}")
        print(f"Errors encountered: {final_state.get('error_count', 0)}")
        
        if final_state.get('reflection_output'):
            reflection = final_state['reflection_output']
            print(f"Goal achieved: {reflection.goal_achieved}")
            print(f"Progress: {reflection.progress_assessment}")
        
        # Show action history
        action_history = final_state.get('action_history', [])
        if action_history:
            print(f"\nActions taken ({len(action_history)}):")
            for i, action in enumerate(action_history):
                print(f"  {i+1}. {action.reasoning} (confidence: {action.confidence:.2f})")
        
        print("\n=== Single Agent Comparison (Legacy) ===")
        # Run the old single agent for comparison
        try:
            single_agent_demo(initial_state)
        except Exception as e:
            print(f"Single agent demo failed: {e}")
            
    except Exception as e:
        print(f"Multi-agent execution failed: {e}")
        import traceback
        traceback.print_exc()


def single_agent_demo(initial_state):
    """Run the original single agent for comparison."""
    # Get first example from original demo code
    raw_dataset = get_dataset()
    ep = get_episode(raw_dataset)
    
    if not ep:
        print("No episodes available for single agent demo")
        return
        
    for i, ex in enumerate(ep):
        example = ex
        print(f'Single Agent Example {i}:')
        goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
        print(f'  Goal: {goal}')
        image_height = example.features.feature['image/height'].int64_list.value[0]
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_channels = example.features.feature['image/channels'].int64_list.value[0]
        image = visualization_utils._decode_image(example, image_height, image_width, image_channels)
        # Convert image tensor to numpy array if it's a tf.Tensor
        if hasattr(image, 'numpy'):
            image_np = image.numpy()
        else:
            image_np = image
        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)
        # Save to buffer and encode as base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image=img_base64  # Use the base64 encoded image for the agent
        break

    # Example usage of the single multimodal agent
    next_step = "find target element"
    text_input = "analyze the interface and identify actionable elements"
    image_input = image

    result = run_agent(goal, next_step, text_input, image_input)
    print(result['next_step'])



if __name__ == "__main__":
    main()