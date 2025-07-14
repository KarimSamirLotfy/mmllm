import logging
import os
from datetime import datetime
import colorlog

def setup_logging(logging_level=logging.INFO):
    """Set up logging configuration with colored output and file logging."""
    # Silence TensorFlow and system warnings/errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0=all, 1=info, 2=warning, 3=error)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
    
    # Set PyDevD timeout to 5 seconds to avoid slow repr warnings
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    
    # Silence specific loggers
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"{timestamp}.log")
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create colored console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging_level)
    
    # Define colored formatter for console
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging_level)
    
    # Define formatter for file (no colors)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set specific logger level for mmllm
    logging.getLogger("mmllm").setLevel(logging_level)
    
    # Silence additional verbose loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.ERROR)
    logging.getLogger('google.auth').setLevel(logging.ERROR)
    logging.getLogger('google.cloud').setLevel(logging.ERROR)
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('jax._src.dispatch').setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("pydevd ").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("pytesseract").setLevel(logging.ERROR)
    # Log the setup completion
    logger.info(f"Logging initialized - Console: colored output, File: {log_filename}")

setup_logging(logging_level=logging.DEBUG)