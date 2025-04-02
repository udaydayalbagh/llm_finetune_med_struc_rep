import sys
sys.path.insert(0, '/home/udevulap/llm_finetune_med_struc_rep/llm_finetune_med_struc_rep')
import argparse
import yaml
from src.training.trainer import Trainer
from src.utils.logger import get_logger

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Run reinforcement learning training for structured medical report generation."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml", 
        help="Path to the configuration file (default: config/config.yaml)"
    )
    args = parser.parse_args()
    
    # Initialize logger.
    logger = get_logger("run_training")
    logger.info(f"Loading configuration from {args.config}")

    # Load configuration file.
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.exception("Failed to load configuration file.")
        raise e

    # Initialize Trainer and start training.
    trainer = Trainer(config)
    logger.info("Starting training process...")
    trainer.train()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
