import pytest
from src.models.model import load_model, generate_output

class DummyModel:
    def __init__(self):
        self.state = "dummy"
    def to(self, device):
        return self
    def generate(self, input_ids, max_length, do_sample, temperature, top_p, pad_token_id):
        return [[42]] 

class DummyTokenizer:
    def encode(self, text, return_tensors):
        return [1, 2, 3]
    def decode(self, token_ids, skip_special_tokens):
        return "Dummy output text with CREATE (:Label {property: value}) -[:RELATES_TO]->"

def test_load_model(monkeypatch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda checkpoint: DummyTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda checkpoint: DummyModel())
    
    model, tokenizer = load_model("dummy_checkpoint")
    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)

def test_generate_output():
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    
    output = generate_output((dummy_model, dummy_tokenizer), "Test input", max_length=50)
    assert isinstance(output, str)
    assert "Dummy output text" in output
