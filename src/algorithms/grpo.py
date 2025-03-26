import torch.nn.functional as F
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from src.models.reward import structured_report_reward
import os
import json

class GRPO:
    def __init__(self, config: dict, model, tokenizer, data):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.data = data

    def train(self):
        max_seq_length = self.config.get("max_seq_length", 1024)
        max_prompt_length = self.config.get("max_prompt_length", 256)

        model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.get("lora_rank", 32),  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules=self.config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_alpha=self.config.get("lora_rank", 32),
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=self.config.get("random_state", 100),
            max_seq_length=max_seq_length,
        )
        
        training_args = GRPOConfig(
            learning_rate=float(self.config.get("learning_rate", 5e-6)),
            adam_beta1=float(self.config.get("adam_beta1", 0.9)),
            adam_beta2=float(self.config.get("adam_beta2", 0.99)),
            weight_decay=float(self.config.get("weight_decay", 0.1)),
            warmup_ratio=float(self.config.get("warmup_ratio", 0.1)),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            logging_steps=float(self.config.get("logging_steps", 1)),
            per_device_train_batch_size=int(self.config.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
            num_generations=int(self.config.get("num_generations", 2)),
            max_prompt_length=max_prompt_length,
            max_completion_length=max_seq_length - max_prompt_length,
            num_train_epochs=float(self.config.get("num_train_epochs", 1)),
            max_steps=int(self.config.get("max_steps", 250)),
            save_steps=float(self.config.get("save_steps", 250)),
            max_grad_norm=float(self.config.get("max_grad_norm", 0.1)),
            report_to="none",  # Can use Weights & Biases
            output_dir=self.config.get("output_dir", None),
        )

        # print("Setting the tokenizer pad token..............")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"
        # model.config.pad_token_id = self.tokenizer.pad_token_id

        trainer = GRPOTrainer(
            model=model,
            processing_class=self.tokenizer,
            reward_funcs=structured_report_reward,
            args=training_args,
            train_dataset=self.data,
        )

        trainer.train()

        return trainer.state.log_history
            