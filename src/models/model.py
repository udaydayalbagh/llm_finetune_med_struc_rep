import os
import logging
import torch
import yaml
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

def load_model(config: dict):
    try:
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
            )
        elif load_in_8bit:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=local_model_dir if local_model_dir and os.path.exists(local_model_dir) else model_checkpoint,
                max_seq_length=max_seq_length,
                load_in_8bit=True,
                fast_inference=fast_inference,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=gpu_memory_utilization,
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
            )
        
        if local_model_dir and not os.path.exists(local_model_dir):
            logger.info(f"Saving model to local directory: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            model.save_pretrained(local_model_dir)
            tokenizer.save_pretrained(local_model_dir)
        return model, tokenizer
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise e


def generate_output(model_tuple, input_text: str, max_length: int = 256, device: torch.device = None) -> str:
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
    