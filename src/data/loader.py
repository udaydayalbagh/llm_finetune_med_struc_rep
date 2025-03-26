import os
import json
import csv
import logging
import yaml
from datasets import load_from_disk, load_dataset
from src.data.preprocessor import Preprocessor
import polars as pl

logger = logging.getLogger(__name__)

def load_rl_data(config, EOS_Token):
    """
    Load structured medical reports from the given data path.

    Parameters:
        data_path (str): The path to a data file or directory containing the reports.

    Returns:
        list: A list of medical report dictionaries.
    """
    data = []
    preprocessor = Preprocessor(config, EOS_Token)
    data_path = config['data_path']
    
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    system_prompt = config.get("system_prompt", "")

    # If the data_path is a directory, load all JSON files
    if os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                file_path = os.path.join(data_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        report = json.load(f)
                        prompt = system_prompt + '\n\n' + str(report)
                        # Support both list of reports or a single report per file
                        if isinstance(report, list):
                            data.extend({'prompt': prompt})
                        elif isinstance(report, dict):
                            data.append({'prompt': prompt})
                        else:
                            logger.error(f"Unexpected JSON structure in file: {file_path}")
                except Exception as e:
                    logger.exception(f"Failed to load file {file_path}: {e}")
            elif filename.endswith('.txt'):
                file_path = os.path.join(data_path, filename)
                try:
                    with open("file.txt", "r") as file:
                        content = file.read()
                except:
                    print("ERROR")

    # If the data_path is a file, load based on the file extension
    elif os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        try:
            if ext == '.json':
                with open(data_path, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        prompt = system_prompt + '\n\n' + str(loaded[0])
                        data.append(prompt)
                    elif isinstance(loaded, dict):
                        prompt = system_prompt + '\n\n' + str(loaded)
                        data.append(prompt)
                    else:
                        logger.error("Unsupported JSON structure in file.")
                        raise ValueError("Unsupported JSON structure.")
            elif ext == '.csv':
                n_rows = config.get('n_rows', 50)
                df = pl.read_csv(data_path, n_rows=n_rows)
                df = df["text"]
                data = preprocessor.format_rl_prompts(df)
            
            elif ext == '.pdf':
                # TODO: Implement PDF
                raise(NotImplementedError)
            else:
                logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            logger.exception(f"Failed to load file {data_path}: {e}")
            raise e
    else:
        logger.error(f"Data path is neither a file nor a directory: {data_path}")
        raise ValueError(f"Data path is neither a file nor a directory: {data_path}")
    
    logger.info(f"Loaded {len(data)} reports from {data_path}")
    return data


def load_sft_data(config: dict, EOS_token):
    dataset_name = config.get('dataset_name', "FreedomIntelligence/medical-o1-reasoning-SFT")
    dataset_path = config.get("data_path", None)
    preprocessor = Preprocessor(config=config, EOS_Token=EOS_token)
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
        dataset = dataset.map(preprocessor.format_sft_prompts, batched = True,)
    else:
        dataset = load_dataset(dataset_name,"en", split = "train[0:20000]", trust_remote_code=True)
        dataset = dataset.map(preprocessor.format_sft_prompts, batched = True,)
        if dataset_path:
            logger.info(f"Saving dataset to local directory: {dataset_path}")
            os.makedirs(dataset_path, exist_ok=True)
            dataset.save_to_disk(dataset_path)
    return dataset

