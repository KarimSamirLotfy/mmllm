import os

import jax

from mmllm.android_in_the_wild.action_matching import check_actions_match
# Suppress TensorFlow warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

from mmllm import setup_logging
import json
import tensorflow as tf

from mmllm.agent_ocr.simple_ocr_agent import SimpleOCRAgent
from mmllm.utils.episode_loader import EpisodeLoader
from mmllm.utils.visualization import plot_episode

import logging
logger = logging.getLogger(__name__)

agent = SimpleOCRAgent(ocr_module=True)
dataset_name = 'google_apps'  #@param ["general", "google_apps", "install", "single", "web_shopping"]

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

def get_episodes(dataset, start_episode=0, end_episode=None):
    """
    Iterator that yields episodes within a specified range.
    
    Args:
        dataset: TensorFlow dataset iterator
        start_episode: Starting episode index (0-based)
        end_episode: Ending episode index (exclusive). If None, yields all remaining episodes.
        
    Yields:
        list: Each complete episode as a list of tf.train.Example objects
    """
    current_episode_id = None
    episode = []
    episode_count = 0
    for d in dataset:
        step = tf.train.Example()
        step.ParseFromString(d)
        ep_id = step.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        
        # If this is the first step, initialize current_episode_id
        if current_episode_id is None:
            current_episode_id = ep_id
            episode.append(step)
        # If we are still in the same episode, append the step
        elif ep_id == current_episode_id:
            episode.append(step)
        elif ep_id != current_episode_id: # End of current episode
            if episode_count >= start_episode and (end_episode is None or episode_count < end_episode):
                yield episode
            # Reset for the next episode
            episode_count += 1

            current_episode_id = None
            episode = []

        if end_episode is not None and episode_count >= end_episode:
            break
        


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
if __name__ == "__main__":
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    # Process multiple episodes using the iterator
    episode_loader = EpisodeLoader()

    # Example: Process first 3 episodes (change the range as needed)
    for episode_idx, episode_tf in enumerate(get_episodes(raw_dataset, start_episode=0, end_episode=3)):
        logger.info(f"\n=== Processing Episode {episode_idx} ===")
        
        episode = episode_loader.load_episode_with_history(episode_tf)
        model_actions = []
        number_of_steps = len(episode['episode_images'])
        
        for step in range(min(12, number_of_steps-2), number_of_steps):
            image = episode['episode_images'][step]
            ui_annotations = episode['ui_annotations_list'][step]
            goal = episode['goal']
            thread_id = episode['episode_id']
            ground_truth_action = episode['ground_truth_actions'][step] if 'ground_truth_actions' in episode else None
            
            # Run the agent for each step in the episode
            result = agent.run_step(
                image=image,  # In real usage, this would be a PIL Image
                ocr_text=None,
                ui_description=ui_annotations,
                goal=goal,
                thread_id=thread_id
            )
            logger.info(f"Episode {episode_idx}, Step {step + 1} Action:")
            # Print the action decided by the agent
            logger.info(json.dumps(result.get('action', {}), indent=2))
            model_actions.append(result.get('action', {}))
            
            # Print the ground truth action if available
            if ground_truth_action:
                logger.info("Ground Truth Action:")
                logger.info(json.dumps(ground_truth_action, indent=2))

            action = result['action']
            # Check action matching status 
        
            matches = check_actions_match(
            action_1_touch_yx = action.get('coordinates', [0, 0]),
            action_1_lift_yx = action.get('lift_coordinates', [0, 0]),
            action_1_action_type = action['action_type'],
            action_2_touch_yx = ground_truth_action.get('coordinates', [0, 0]),
            action_2_lift_yx = ground_truth_action.get('lift_coordinates', [0, 0]),
            action_2_action_type = ground_truth_action['action_type'],
            # exatract the ui elements [N, 4] list 
            annotation_positions = jax.numpy.array([ui_element['position'] for ui_element in ui_annotations])
            )
            logger.info(f"Actions match: {matches}")
            logger.info(f"Task completed: {result.get('task_completed', False)}\n")
            
            # Visualize the episode
            plot_episode(
                episode=episode,
                model_actions=model_actions,
                show_actions=True,
            )
            
            # Check if task is done
            if result.get('task_completed', False):
                logger.info(f"Episode {episode_idx} - Task completed!")
                break
