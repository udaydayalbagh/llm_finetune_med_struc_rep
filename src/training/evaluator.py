import logging
import torch
from src.models.model import load_model, generate_output
from src.data.loader import load_rl_test_data
from src.models.reward import structured_report_reward
import os
import numpy as np
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator class to handle the evaluation of LLMs for generating structured reports.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model(self.config)

        # Load dataset
        self.data = load_rl_test_data(self.config, self.tokenizer.eos_token)


    def evaluate(self):
        # logs_dir = self.config.get("logs_dir", None)
        # logs_file_name = self.config.get("logs_file_name", None)
        # log_file = os.path.join(logs_dir, logs_file_name) if logs_dir and logs_file_name else None

        # Loop over each example in the dataset
        rewards = []
        for index, example in enumerate(self.data):
            print(f"Example {index+1}/{len(self.data)}")
            # Format the input for the model
            input_text = example["prompt"]
            max_length = self.config.get("max_length", 3000)
            output_text = generate_output((self.model, self.tokenizer), input_text, max_length=max_length, device=self.device)
            reward = structured_report_reward([example["prompt"]], [output_text])
            rewards.append(reward[0])
            # print(f"Generated Output: {output_text}")
            print(f"Reward: {reward[0]}")

        # Calculate the mean and standard deviation of the reward
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        print(f"Mean Reward: {reward_mean}, Std Dev: {reward_std}")

        # Save the results to a log file

        