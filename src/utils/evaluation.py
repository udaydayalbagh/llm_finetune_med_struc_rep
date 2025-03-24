import re
import logging

logger = logging.getLogger(__name__)

def validate_neo4j_output(output_text: str) -> bool:
    """
    Validate if the given output text conforms to a Neo4j-compatible Cypher format.
    
    Checks include:
      - Presence of key Cypher commands (e.g., "CREATE" or "MERGE")
      - A basic node pattern: e.g., (:Label {property: value})
      - A basic relationship pattern: e.g., -[:RELATES_TO]->
    
    Parameters:
        output_text (str): The text output from the model.
    
    Returns:
        bool: True if the output is valid; otherwise False.
    """
    try:
        if not output_text:
            logger.debug("Empty output text")
            return False
        
        output_upper = output_text.upper()
        if not ("CREATE" in output_upper or "MERGE" in output_upper):
            logger.debug("Missing CREATE/MERGE keyword")
            return False
        
        # Check for a basic node definition pattern.
        node_pattern = r'\(:\w+\s*\{[^}]+\}\)'
        if not re.search(node_pattern, output_text):
            logger.debug("Node pattern not found")
            return False
        
        # Check for a basic relationship pattern.
        rel_pattern = r'-\s*\[:\w+\]\s*->'
        if not re.search(rel_pattern, output_text):
            logger.debug("Relationship pattern not found")
            return False
        
        logger.debug("Output validated as Neo4j-compatible")
        return True
    except Exception as e:
        logger.exception(f"Error validating neo4j output: {e}")
        return False

def compute_metrics(output_text: str) -> dict:
    """
    Compute evaluation metrics for the given output text.
    
    Metrics include:
      - validity: Whether the output passes Neo4j format validation.
      - length: Character length of the output text.
    
    Parameters:
        output_text (str): The model-generated output text.
    
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        is_valid = validate_neo4j_output(output_text)
        metrics = {
            "validity": is_valid,
            "length": len(output_text)
        }
        logger.debug(f"Computed metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.exception(f"Error computing metrics: {e}")
        return {"validity": False, "length": 0}
