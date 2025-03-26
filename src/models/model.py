import os
import logging
import torch
import yaml
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

def load_model(config: dict):
    """
    Load the deepseek-r1 model and its tokenizer from the given checkpoint.
    Optionally loads the model in 4-bit quantized mode to reduce memory usage.
    If a local_model_dir is provided, attempt to load the model from that directory.
    If the model is not present locally, download it from Hugging Face and save it.

    Parameters:
        model_checkpoint (str): The Hugging Face model identifier or remote checkpoint.
        local_model_dir (str, optional): Path to a local directory to load/save the model.
        load_in_4bit (bool): Whether to load the model in 4-bit quantized mode.

    Returns:
        tuple: (model, tokenizer) ready for inference or fine-tuning.
    """
    try:
        # self.config['model_checkpoint'], self.config['local_model_dir']
        model_checkpoint = config.get("model_checkpoint", "")
        local_model_dir = config.get("local_model_dir", "")
        load_in_4bit = config.get("load_in_4bit", True)
        load_in_8bit = config.get("load_in_8bit", False)
        max_seq_length = config.get("max_seq_length", 1024)
        lora_rank = config.get("lora_rank", 32)
        fast_inference = config.get("fast_inference", False)
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.6)

        if local_model_dir and os.path.exists(local_model_dir):
            logger.info(f"Loading model from local directory: {local_model_dir}")
        else:
            logger.info(f"Downloading model from checkpoint: {model_checkpoint}")
        
        if load_in_4bit:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=local_model_dir if local_model_dir and os.path.exists(local_model_dir) else model_checkpoint,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=None,
                fast_inference=fast_inference,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=gpu_memory_utilization,
                # device_map="auto"
            )
            # tokenizer.padding_side='left'
        elif load_in_8bit:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=local_model_dir if local_model_dir and os.path.exists(local_model_dir) else model_checkpoint,
                max_seq_length=max_seq_length,
                load_in_8bit=True,
                fast_inference=fast_inference,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=gpu_memory_utilization,
                # device_map="auto"
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=local_model_dir if local_model_dir and os.path.exists(local_model_dir) else model_checkpoint,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                fast_inference=fast_inference,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=gpu_memory_utilization,
                # device_map="auto"
            )
        
        if local_model_dir and not os.path.exists(local_model_dir):
            logger.info(f"Saving model to local directory: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            model.save_pretrained(local_model_dir)
            tokenizer.save_pretrained(local_model_dir)

        # tokenizer.padding_side = "left"
        # print("Tokenizer eos_token: ", tokenizer.eos_token)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        # print("Model pad_token_id: ", model.config.padding_side)
        return model, tokenizer
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise e


def generate_output(model_tuple, input_text: str, max_length: int = 256, device: torch.device = None) -> str:
    """
    Generate output text from the model given an input medical report.

    Parameters:
        model_tuple (tuple): Tuple containing the (model, tokenizer).
        input_text (str): The input medical report text.
        max_length (int): Maximum length for the generated output.
        device (torch.device): Device to be used for generation.

    Returns:
        str: The generated output text.
    """
    model, tokenizer = model_tuple
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    except Exception as e:
        logger.exception(f"Error generating output from model: {e}")
        raise e


















# import os
# import logging
# import torch
# import yaml
# from transformers import AutoTokenizer, AutoModelForCausalLM

# logger = logging.getLogger(__name__)

# def load_model(model_checkpoint: str, local_model_dir: str = None):
#     """
#     Load the deepseek-r1 model and its tokenizer from the given checkpoint.
#     If a local_model_dir is provided, attempt to load the model from that directory.
#     If the model is not present locally, download it from Hugging Face and save it.

#     Parameters:
#         model_checkpoint (str): The Hugging Face model identifier or remote checkpoint.
#         local_model_dir (str, optional): Path to a local directory to load/save the model.

#     Returns:
#         tuple: (model, tokenizer) ready for inference or fine-tuning.
#     """
#     try:
#         # Check if a local directory is provided and exists.
#         if local_model_dir and os.path.exists(local_model_dir):
#             logger.info(f"Loading model from local directory: {local_model_dir}")
#             tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
#             model = AutoModelForCausalLM.from_pretrained(local_model_dir)
#         else:
#             # Download model from Hugging Face.
#             logger.info(f"Downloading model from checkpoint: {model_checkpoint}")
#             tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#             model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
#             # Save the model locally if a directory is provided.
#             if local_model_dir:
#                 logger.info(f"Saving model to local directory: {local_model_dir}")
#                 os.makedirs(local_model_dir, exist_ok=True)
#                 model.save_pretrained(local_model_dir)
#                 tokenizer.save_pretrained(local_model_dir)
#         model.eval()  # Set model to evaluation mode.
#         return model, tokenizer
#     except Exception as e:
#         logger.exception(f"Error loading model: {e}")
#         raise e

# def generate_output(model_tuple, input_text: str, max_length: int = 256, device: torch.device = None) -> str:
#     """
#     Generate output text from the model given an input medical report.

#     Parameters:
#         model_tuple (tuple): Tuple containing the (model, tokenizer).
#         input_text (str): The input medical report text.
#         max_length (int): Maximum length for the generated output.
#         device (torch.device): Device to be used for training

#     Returns:
#         str: The generated output text.
#     """
#     with open("config/config.yaml", "r") as f:
#         config = yaml.safe_load(f)
#     system_prompt = config["system_prompt"]
#     text = system_prompt + "\n\n" + input_text
#     model, tokenizer = model_tuple
#     try:
#         input_ids = tokenizer.encode(text, return_tensors='pt')
#         input_ids = input_ids.to(device)
#         with torch.no_grad():
#             output_ids = model.generate(
#                 input_ids,
#                 max_length=max_length,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.95,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         return output_text
#     except Exception as e:
#         logger.exception(f"Error generating output from model: {e}")
#         raise e
