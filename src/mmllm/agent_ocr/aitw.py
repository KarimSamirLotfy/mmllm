import json
import tensorflow as tf

from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent
from mmllm.utils.episode_loader import EpisodeLoader
from mmllm.utils.visualization import plot_episode

agent = SimpleOCRAgent(ocr_module=True)
dataset_name = 'google_apps'  #@param ["general", "google_apps", "install", "single", "web_shopping"]

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

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

filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

episode_tf = get_episode(raw_dataset)

# 
episode_loader = EpisodeLoader()

episode = episode_loader.load_episode_with_history(episode_tf)
model_actions = []
number_of_steps = len(episode['episode_images'])
for step in range(8, number_of_steps):
    image = episode['episode_images'][step]
    ui_annotations_list = episode['ui_annotations_list'][step]
    goal = episode['goal']
    thread_id = episode['episode_id']
    ground_truth_actions = episode['ground_truth_actions'][step] if 'ground_truth_actions' in episode else None
    # Run the agent for each step in the episode
    result = agent.run_step(
        image=image,  # In real usage, this would be a PIL Image
        ocr_text=None,
        ui_description=ui_annotations_list,
        goal=goal,
        thread_id=thread_id
    )
    print(f"Step {step + 1} Action:")
    # Print the action decided by the agent
    print(json.dumps(result.get('action', {}), indent=2))
    model_actions.append(result.get('action', {}))
    # Print the ground truth action if available
    if ground_truth_actions:
        print("Ground Truth Action:")
        print(json.dumps(ground_truth_actions, indent=2))
    print(f"Task completed: {result.get('task_completed', False)}\n")
    # Visualize the episode
    plot_episode(
        episode=episode,
        model_actions=model_actions,
        show_actions=True,
    )
    # Check if task is done
    if result.get('task_completed', False):
        print("Task completed!")
        break
