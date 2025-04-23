import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

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
        try:
            if "</think>" in output_text:
                output_text = output_text.split("</think>")
                output_text = output_text[1].strip()

            length_reward = 0
            if len(output_text.strip()) > 500:
                length_reward += 0.2

            found_headings = set(re.findall(r'\*\*(.*?)\*\*\s*', output_text))
            if found_headings:
                found_headings_len = len(found_headings)
            else:
                found_headings_len = 0

            heading_reward = 0
            extra_headings = 0

            for required_heading in required_headings:
                exists = False
                for found_heading in found_headings:
                    if required_heading in found_heading and exists:
                        heading_reward -= 1
                    elif required_heading in found_heading and not exists:
                        heading_reward += 1
                        exists = True
                
            total_reward = heading_reward + length_reward
            total_reward = 0 if total_reward < 0 else total_reward
            rewards.append(total_reward)
        except Exception as e:
            logger.exception(f"Error computing reward for output: {e}")
            rewards.append(0.0)
    return rewards

