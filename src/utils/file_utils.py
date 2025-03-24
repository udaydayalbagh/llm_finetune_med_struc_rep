import os
import json
import logging
import torch

logger = logging.getLogger(__name__)

def save_json(data, filepath: str):
    """
    Save data as JSON to the specified file path.
    
    Parameters:
        data: Data to be saved.
        filepath (str): Destination file path.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data successfully saved to {filepath}")
    except Exception as e:
        logger.exception(f"Error saving data to {filepath}: {e}")
        raise

def load_json(filepath: str):
    """
    Load JSON data from the specified file path.
    
    Parameters:
        filepath (str): Source file path.
    
    Returns:
        The data loaded from JSON.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data successfully loaded from {filepath}")
        return data
    except Exception as e:
        logger.exception(f"Error loading data from {filepath}: {e}")
        raise

def save_checkpoint(model, filepath: str):
    """
    Save a PyTorch model checkpoint.
    
    Parameters:
        model: The model to save.
        filepath (str): File path to save the checkpoint.
    """
    try:
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model checkpoint saved at {filepath}")
    except Exception as e:
        logger.exception(f"Error saving model checkpoint to {filepath}: {e}")
        raise

def load_checkpoint(model, filepath: str):
    """
    Load a PyTorch model checkpoint.
    
    Parameters:
        model: The model to load the checkpoint into.
        filepath (str): File path of the checkpoint.
    """
    try:
        model.load_state_dict(torch.load(filepath))
        logger.info(f"Model checkpoint loaded from {filepath}")
    except Exception as e:
        logger.exception(f"Error loading model checkpoint from {filepath}: {e}")
        raise
