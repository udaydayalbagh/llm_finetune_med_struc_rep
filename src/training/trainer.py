import logging
import torch
import yaml
from src.models.model import load_model
from src.data.loader import load_rl_data, load_sft_data
from src.algorithms.grpo import GRPO
from src.algorithms.sft import SFT
import os
import json

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.algorithm = self.config.get('algorithm', 'grpo')

        # Load model and tokenizer
        self.model, self.tokenizer = load_model(self.config)

        # Load dataset
        if self.algorithm == 'sft':
            self.data = load_sft_data(self.config, self.tokenizer.eos_token)
        else:
            self.data = load_rl_data(self.config, self.tokenizer.eos_token)


    def train(self):
        logs_dir = self.config.get("logs_dir", None)
        logs_file_name = self.config.get("logs_file_name", None)
        log_file = os.path.join(logs_dir, logs_file_name) if logs_dir and logs_file_name else None

        if self.algorithm == 'grpo':
            trainer = GRPO(self.config, self.model, self.tokenizer, self.data)
            logs = trainer.train()
            if log_file:
                with open(log_file, "w") as f:
                    json.dump(logs, f, indent=4)

        elif self.algorithm == 'ppo':
            # TODO: Implement PPO
            raise NotImplementedError()
        
        elif self.algorithm == 'sft':
            trainer = SFT(self.config, self.model, self.tokenizer, self.data)
            trainer_stats = trainer.train()
            if log_file:
                with open(log_file, "w") as f:
                    json.dump(trainer_stats, f, indent=4)
        else:
            logger.error(f"Invalid algorithm {self.algorithm}")
