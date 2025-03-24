import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    handler.flush = sys.stdout.flush  # Ensure immediate output
    logger.addHandler(handler)
    
    logger.propagate = False  # Prevent interference
    return logger


# def get_logger(name: str) -> logging.Logger:
#     """
#     Create and return a logger with a standardized format.
    
#     Parameters:
#         name (str): The name for the logger.
    
#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     logger = logging.getLogger(name)
#     if not logger.handlers:
#         logger.setLevel(logging.DEBUG)
#         handler = logging.StreamHandler(sys.stdout)
#         handler.setLevel(logging.DEBUG)
#         formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#     return logger
