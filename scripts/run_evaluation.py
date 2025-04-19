import sys
sys.path.insert(0, '/home/udevulap/llm_finetune_med_struc_rep/llm_finetune_med_struc_rep')
import argparse
import yaml
from src.training.evaluator import Evaluator
from src.utils.logger import get_logger

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation for structured medical report generation."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/eval.yaml", 
        help="Path to the configuration file (default: config/eval.yaml)"
    )
    args = parser.parse_args()
    
    # Initialize logger.
    logger = get_logger("run_evaluation")
    logger.info(f"Loading configuration from {args.config}")

    # Load configuration file.
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.exception("Failed to load configuration file.")
        raise e

    # Initialize Trainer and start training.
    evaluator = Evaluator(config)
    logger.info("Starting evaluation process...")
    evaluator.evaluate()
    logger.info("Evaluation completed.")

if __name__ == "__main__":
    main()
