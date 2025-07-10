import logging
import os
from datetime import datetime
import colorlog

def setup_logging(logging_level=logging.INFO):
    """Set up logging configuration with colored output and file logging."""
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
    
    # Log the setup completion
    logger.info(f"Logging initialized - Console: colored output, File: {log_filename}")

setup_logging()