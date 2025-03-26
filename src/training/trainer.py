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
    """
    Trainer class to handle the reinforcement learning training for fine-tuning the LLM.
    """

    def __init__(self, config: dict):
        """
        Initialize the Trainer with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters including model_checkpoint, data_path,
                           learning_rate, num_epochs, max_length, and checkpoint_dir.
        """
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

        # Create checkpoint directory if it doesn't exist
        # self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        # logger.info(f"Checkpoint directory set to: {self.checkpoint_dir}")


    def train(self):
        """
        Execute the training over the dataset.
        
        Returns:
            The trained model.
        """
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
            model, tokenizer, trainer_stats = trainer.train()
            output_dir = self.config.get("output_dir", None)
            if output_dir:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                # model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit",)
            if log_file:
                with open(log_file, "w") as f:
                    json.dump(trainer_stats, f, indent=4)
        else:
            logger.error(f"Invalid algorithm {self.algorithm}")

        
if __name__ == "__main__":
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        trainer = Trainer(config)
        trainer.train()
    except Exception as e:
        logger.exception(f"Training failed: {e}")
