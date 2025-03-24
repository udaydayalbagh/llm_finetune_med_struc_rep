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
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                # Use num_train_epochs = 1, warmup_ratio for full training runs!
                warmup_steps=5,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        trainer_stats = trainer.train()

        return model, self.tokenizer, trainer_stats
