"""TensorFlow configuration utility to prevent double-free errors."""

import tensorflow as tf

# Configure TensorFlow once when this module is imported
def configure_tensorflow():
    
    """Configure TensorFlow to prevent memory conflicts and double-free errors."""
    # Enable eager execution
    tf.config.run_functions_eagerly(True)
    #tf.data.experimental.enable_debug_mode()
    
    # Configure GPU memory growth
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # Ignore GPU configuration errors
    
    return
    # Additional configurations to prevent memory conflicts
    try:
        # Disable oneDNN optimizations that can cause conflicts
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    except:
        pass

# Auto-configure when module is imported
configure_tensorflow()
