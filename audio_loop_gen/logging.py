import logging
from logging.handlers import RotatingFileHandler
import os

def setup_global_logging(logs_file:str = "global.log", logs_path: str = os.path.join(".", "logs"), log_level: int = logging.INFO) -> logging.Logger:
    log_file = os.path.join(logs_path, logs_file)

    # Create log directory if it doesn't exist
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)

    # Create a logger
    logger = logging.getLogger('global')
    logger.setLevel(logging.INFO if log_level < 0 else log_level)

    # Create a handler that writes log messages to a file, with log rotation
    handler = RotatingFileHandler(
        log_file, maxBytes=1024*1024*5, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger