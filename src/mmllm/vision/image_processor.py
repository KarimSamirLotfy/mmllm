"""Image preprocessing for multimodal models."""

import base64
import io
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf


class ImageProcessor:
    """Handles image preprocessing for multimodal vision models."""
    
    def __init__(self):
        self.max_image_size = 1024  # Maximum dimension for processed images
    
    def encode_image_for_model(self, image: np.ndarray) -> str:
        """
        Encode numpy image array to base64 string for model input.
        
        Args:
            image: Numpy array representing the image
            
        Returns:
            Base64 encoded image string
        """
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Resize if too large
        if max(pil_image.size) > self.max_image_size:
            pil_image = self._resize_image(pil_image, self.max_image_size)
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def decode_aitw_image(self, example: tf.train.Example) -> np.ndarray:
        """
        Decode image from AiTW dataset example.
        
        Args:
            example: TensorFlow example from AiTW dataset
            
        Returns:
            Decoded image as numpy array
        """
        # Extract image metadata
        image_height = example.features.feature['image/height'].int64_list.value[0]
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_channels = example.features.feature['image/channels'].int64_list.value[0]
        
        # Decode raw image data
        image = tf.io.decode_raw(
            example.features.feature['image/encoded'].bytes_list.value[0],
            out_type=tf.uint8,
        )
        
        # Reshape to original dimensions
        height = tf.cast(image_height, tf.int32)
        width = tf.cast(image_width, tf.int32)
        n_channels = tf.cast(image_channels, tf.int32)
        
        image = tf.reshape(image, (height, width, n_channels))
        
        # Convert to numpy
        if hasattr(image, 'numpy'):
            return image.numpy()
        else:
            return image
    
    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        width, height = image.size
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def prepare_multimodal_input(self, image: np.ndarray, text: str) -> Dict[str, Any]:
        """
        Prepare multimodal input for vision-language models.
        
        Args:
            image: Image as numpy array
            text: Text prompt
            
        Returns:
            Dictionary with formatted multimodal input
        """
        base64_image = self.encode_image_for_model(image)
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
    
    def get_image_metadata(self, image: np.ndarray) -> Dict[str, Any]:
        """Get metadata about the image."""
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size_mb": image.nbytes / (1024 * 1024),
            "dimensions": {"height": image.shape[0], "width": image.shape[1]}
        }
