#!/usr/bin/env python3
"""
Simple script to plot a specific episode by index.
Usage: python plot_episode.py <episode_index>
"""

import os
import sys
import argparse
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

from mmllm.utils.episode_loader import EpisodeLoader
from mmllm.utils.visualization import plot_episode

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
        elif ep_id != current_episode_id:  # End of current episode
            if episode_count >= start_episode and (end_episode is None or episode_count < end_episode):
                yield episode
            # Reset for the next episode
            episode_count += 1
            
            if end_episode is not None and episode_count >= end_episode:
                break
                
            current_episode_id = ep_id
            episode = [step]
        
    # Yield the last episode if it's in range
    if episode and episode_count >= start_episode and (end_episode is None or episode_count < end_episode):
        yield episode

def main():
    parser = argparse.ArgumentParser(description='Plot a specific episode by index')
    parser.add_argument('episode_index', type=int, help='Episode index to plot (0-based)')
    parser.add_argument('--dataset', default='google_apps', 
                       choices=['general', 'google_apps', 'install', 'single', 'web_shopping'],
                       help='Dataset to use')
    parser.add_argument('--output_dir', default='./plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset directories
    dataset_directories = {
        'general': 'gs://gresearch/android-in-the-wild/general/*',
        'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
        'install': 'gs://gresearch/android-in-the-wild/install/*',
        'single': 'gs://gresearch/android-in-the-wild/single/*',
        'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
    }
    
    # Load dataset
    filenames = tf.io.gfile.glob(dataset_directories[args.dataset])
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
    
    # Load the specific episode
    episode_loader = EpisodeLoader()
    
    # Get the requested episode
    target_episode = None
    for episode_idx, episode_tf in enumerate(get_episodes(raw_dataset, 
                                                         start_episode=args.episode_index, 
                                                         end_episode=args.episode_index + 1)):
        target_episode = episode_loader.load_episode_with_history(episode_tf)
        break
    
    if target_episode is None:
        print(f"Episode {args.episode_index} not found in dataset '{args.dataset}'")
        sys.exit(1)
    
    # Plot the episode
    print(f"Plotting episode {args.episode_index} to {args.output_dir}/")
    plot_episode(
        episode=target_episode,
        model_actions=[],  # No model actions, just plotting the episode
        show_actions=True,
        # save_path=f"{args.output_dir}/episode_{args.episode_index}.png"
    )
    
    print(f"Episode {args.episode_index} plotted successfully!")

if __name__ == "__main__":
    main()
