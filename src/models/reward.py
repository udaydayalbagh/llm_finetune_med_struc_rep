import logging
import re

logger = logging.getLogger(__name__)

def structured_report_reward(prompts, completions):
    headings = [
        "Patient Details",
        "Service", 
        "Allergies",
        "Attending",
        "Chief Complaint",
        "Major Surgical or Invasive Procedure",
        "History of Present Illness",
        "Past Medical History",
        "Social History", 
        "Physical Examination",
        "Discharge PE",
        "tinent Results",
        "Brief Hospital Course",
        "Medications on Admission",
        "Discharge Medications", 
        "Discharge Disposition",
        "Discharge Diagnosis",
        "Discharge Condition",
        "Discharge Instructions",
        "Followup Instructions",
    ]
    rewards = []
    for output_text in completions:
        try:
            reward = 0.0
            output_upper = output_text.upper()

            if len(output_text.strip()) > 100:
                reward += 0.2

            for heading in headings:
                if f"**{heading.upper()}" in output_upper:
                    reward += 0.2

            logger.debug(f"Computed reward: {reward} for output: {output_text}")
            rewards.append(reward)
        except Exception as e:
            logger.exception(f"Error computing reward for output: {e}")
            rewards.append[0.0]
    return rewards


def compute_neo4j_graph_reward(prompts: str, completions):
    """
    Compute a reward based on how well the generated output conforms to a Neo4j-compatible format.
    
    The reward function uses simple heuristics such as checking for:
      - Presence of key Cypher keywords (e.g., "CREATE", "MERGE")
      - Basic node definitions (e.g., '(:Label {property: value})')
      - Relationship patterns (e.g., '-[:RELATES_TO]->')

    Parameters:
        output_text (str): The generated output text from the model.

    Returns:
        float: Reward score based on the quality of the output.
    """
    rewards = []
    for output_text in completions:
        try:
            reward = 0.0
            output_upper = output_text.upper()

            # Reward for presence of key Cypher commands
            if "CREATE" in output_upper or "MERGE" in output_upper:
                # reward += 1.0
                reward += 0.5

            # Check for a basic node definition pattern (e.g., (:Label {property: value}))
            node_pattern = r'\(:\w+\s*\{[^}]*\}\)'
            if re.search(node_pattern, output_text):
                reward += 1.0
                # reward += 0.5

            # Check for a basic relationship pattern (e.g., -[:RELATES_TO]->)
            rel_pattern = r'-\s*\[:\w+\]\s*->'
            if re.search(rel_pattern, output_text):
                reward += 1.0
                # reward += 0.5

            # # Additional heuristic: reward longer, non-empty outputs
            # if len(output_text.strip()) > 50:
            #     reward += 0.5

            logger.debug(f"Computed reward: {reward} for output: {output_text}")
            # print(f"Computed reward: {reward} for output: {output_text}")
            rewards.append(reward)
        except Exception as e:
            logger.exception(f"Error computing reward for output: {e}")
            rewards.append[0.0]
    return rewards
