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

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model(self.config)

        # Load dataset
        self.data = load_rl_test_data(self.config, self.tokenizer.eos_token)


    def evaluate(self):
        rewards = []
        for index, example in enumerate(self.data):
            print(f"Example {index+1}/{len(self.data)}")
            # Format the input for the model
            input_text = example["prompt"]
            max_length = self.config.get("max_length", 3000)
            output_text = generate_output((self.model, self.tokenizer), input_text, max_length=max_length, device=self.device)
            reward = structured_report_reward([example["prompt"]], [output_text])
            rewards.append(reward[0])
            print(f"Reward: {reward[0]}")

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        print(f"Mean Reward: {reward_mean}, Std Dev: {reward_std}")

        