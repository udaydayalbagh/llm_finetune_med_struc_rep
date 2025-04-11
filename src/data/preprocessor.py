import logging
import re

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config: dict, EOS_Token):
        self.include_think_tag = config.get('include_think_tag', True)
        if self.include_think_tag:
            self.train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

            ### Instruction:
            {}

            ### Question:
            {}

            ### Response:
            <think>
            {}
            </think>
            {}"""
        else:
            self.train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

            ### Instruction:
            {}

            ### Question:
            {}

            ### Response:
            {}"""

        self.rl_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

            ### Instruction:
            {}

            ### Unstructured Report:
            {}

            ### Structured Report:
            """
        
        self.EOS_Token = EOS_Token
        self.instruction = config.get('system_prompt', '')

        # You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        # Please answer the following medical question.

    def format_sft_prompts(self, examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            if self.include_think_tag:
                text = self.train_prompt_style.format(self.instruction, input, cot, output) + self.EOS_Token
            else:
                text = self.train_prompt_style.format(self.instruction, input, output) + self.EOS_Token
            texts.append(text)
        return {"text": texts}
    
    def format_rl_prompts(self, data):
        texts = []
        for report in data:
            text = {'prompt': self.rl_prompt_style.format(self.instruction, report)}
            texts.append(text)
        return texts
