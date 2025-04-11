from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

class SFT:
    def __init__(self, config, model, tokenizer, dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = self.config.get('max_seq_length', 4096)

    def train(self):
        model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.get("lora_rank", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_alpha=self.config.get("lora_rank", 32),
            use_gradient_checkpointing="unsloth",
            random_state=self.config.get("random_state", 100),
            max_seq_length=self.max_seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(
                num_train_epochs=float(self.config.get("num_train_epochs", 1)),
                per_device_train_batch_size=int(self.config.get("per_device_train_batch_size", 2)),
                gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
                warmup_steps=int(self.config.get("warmup_steps", 1)),
                max_steps=int(self.config.get("max_steps", -1)),
                learning_rate=float(self.config.get("learning_rate", 2e-4)),
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=float(self.config.get("logging_steps", 1)),
                optim=self.config.get("optim", "adamw_8bit"),
                weight_decay=float(self.config.get("weight_decay", 0.01)),
                lr_scheduler_type=self.config.get("lr_scheduler_type", "linear"),
                save_steps=self.config.get("save_steps", 500),
                save_total_limit=self.config.get("save_total_limit", 2),
                output_dir=self.config.get("output_dir", None),
            ),
        )

        trainer.train()

        return trainer.state.log_history
