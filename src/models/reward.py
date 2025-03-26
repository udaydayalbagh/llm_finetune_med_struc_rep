import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

def remove_think(text:str):
    s = text.split("</think>")
    return s[1].strip()

# Extract all the headings. Improve the headings reward. Give reward for number of headings. Max reward for 20 headings. Lesser reward for less or more headings.
# Also check that under each heading there are bullet points, not a paragraph. 
# Reduce reward for repetitive sentences.
# ASK CHATGPT!

def structured_report_reward(prompts, completions):
    required_headings = {
        "Patient Details", "Service", "Allergies", "Attending", "Chief Complaint",
        "Major Surgical or Invasive Procedure", "History of Present Illness", "Past Medical History",
        "Social History", "Physical Examination", "Discharge PE", "tinent Results",
        "Brief Hospital Course", "Medications on Admission", "Discharge Medications", 
        "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition", 
        "Discharge Instructions", "Followup Instructions"
    }

    rewards = []
    for output_text in completions:
        # print(output_text)
        try:
            if "</think>" in output_text:
                output_text = output_text.split("</think>")
                output_text = output_text[1].strip()

            length_reward = 0
            if len(output_text.strip()) > 200:
                length_reward += 0.2

            # Extract headings using regex (assuming headings are followed by a newline)
            found_headings = set(re.findall(r'\*\*(.*?)\*\*\s*', output_text))
            if found_headings:
                found_headings_len = len(found_headings)
            else:
                found_headings_len = 0
            # print(f"found headings : {found_headings} .........................")

            heading_reward = 0
            extra_headings = 0

            for required_heading in required_headings:
                exists = False
                previous_headings = []
                for found_heading in found_headings:
                    if required_heading in found_heading and found_heading not in previous_headings:
                        heading_reward += 0.2
                        exists = True
                    if found_heading in previous_headings:
                        heading_reward -= 0.05
                    previous_headings.append(found_heading)
                if not exists:
                    extra_headings += 1
            missing_headings = len(required_headings) - (found_headings_len - extra_headings)

            heading_penalty = -0.01 * (extra_headings + missing_headings)
            
            # Check for phrase repetition using n-grams
            words = re.findall(r'\b\w+\b', output_text.lower())  # Tokenize words
            n_gram_size = 4  # Define n-gram size
            n_grams = [tuple(words[i:i + n_gram_size]) for i in range(len(words) - n_gram_size + 1)]
            n_gram_counts = Counter(n_grams)
            repetition_penalty = 0.01 * -sum(count - 1 for count in n_gram_counts.values() if count > 1)
            # print(f"repetition_penalty : {repetition_penalty} .........................")
            
            # Total reward calculation
            # total_reward = heading_reward + heading_penalty + repetition_penalty
            total_reward = heading_reward + length_reward
            rewards.append(total_reward)
        except Exception as e:
            logger.exception(f"Error computing reward for output: {e}")
            rewards.append(0.0)
    return rewards
    
    


# def structured_report_reward(prompts, completions):
#     headings = [
#         "Patient Details",
#         "Service", 
#         "Allergies",
#         "Attending",
#         "Chief Complaint",
#         "Major Surgical or Invasive Procedure",
#         "History of Present Illness",
#         "Past Medical History",
#         "Social History", 
#         "Physical Examination",
#         "Discharge PE",
#         "tinent Results",
#         "Brief Hospital Course",
#         "Medications on Admission",
#         "Discharge Medications", 
#         "Discharge Disposition",
#         "Discharge Diagnosis",
#         "Discharge Condition",
#         "Discharge Instructions",
#         "Followup Instructions",
#     ]
#     rewards = []
#     for output_text in completions:
#         print(output_text)
#         try:
#             reward = 0.0
#             if "</think>" in output_text:
#                 output_text = output_text.split("</think>")
#                 output_text = output_text[1].strip()
#             output_upper = output_text.upper()

#             if len(output_upper.strip()) > 100:
#                 reward += 0.2
#             else:
#                 reward -= 0.1

#             for heading in headings:
#                 if f"**{heading.upper()}" in output_upper:
#                     reward += 0.2

#             logger.debug(f"Computed reward: {reward} for output: {output_text}")
#             rewards.append(reward)
#         except Exception as e:
#             logger.exception(f"Error computing reward for output: {e}")
#             rewards.append[0.0]
#     return rewards


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
