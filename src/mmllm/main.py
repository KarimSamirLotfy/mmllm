import langgraph
from agent import run_agent

from android_in_the_wild import visualization_utils
import tensorflow as tf
import base64
import io
from PIL import Image

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
    raw_dataset = get_dataset()
    ep = get_episode(raw_dataset)
    for i, ex in enumerate(ep):
        example = ex
        print(f'Example {i}:')
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

    # Example usage of the multimodal agent
    next_step = "open google chrome"
    text_input = "draw a bounding box arround google chrome"
    image_input = image  # Replace with actual image data if available

    result = run_agent(goal, next_step, text_input, image_input)
    print("Agent Result:", result)



if __name__ == "__main__":
    main()