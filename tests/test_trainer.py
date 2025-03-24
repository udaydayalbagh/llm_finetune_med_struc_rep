import os
import tempfile
import pytest
from src.training.trainer import Trainer

# Dummy classes and functions to simulate training dependencies.
class DummyModel:
    def __init__(self):
        self.state = "dummy"
    def to(self, device):
        return self
    def parameters(self):
        # Return an empty list; optimizer operations are monkeypatched.
        return []
    def state_dict(self):
        return {"state": self.state}

class DummyTokenizer:
    def encode(self, text, return_tensors):
        return [1, 2, 3]
    def decode(self, token_ids, skip_special_tokens):
        return "Dummy output text with CREATE (:Label {property: value}) -[:RELATES_TO]->"

def dummy_generate_output(model_tuple, input_text, max_length):
    return "Dummy output text with CREATE (:Label {property: value}) -[:RELATES_TO]->"

def dummy_compute_reward(output_text):
    return 3.5

def dummy_load_data(data_path):
    # Return a single dummy report.
    return [{"text": "Test report", "patient_id": "001", "date": "2025-03-15"}]

class DummyOptimizer:
    def zero_grad(self):
        pass
    def step(self):
        pass

@pytest.fixture
def dummy_config(tmp_path):
    # Create a temporary checkpoint directory.
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    config = {
        "model_checkpoint": "dummy_checkpoint",
        "data_path": "dummy_path",  # This will be overridden by monkeypatch.
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "max_length": 50,
        "checkpoint_dir": str(checkpoint_dir)
    }
    return config

def test_trainer_train(monkeypatch, dummy_config):
    # Monkeypatch dependencies in the Trainer.
    from src.models.model import load_model, generate_output
    from src.models.reward import compute_reward
    from src.data.loader import load_data

    monkeypatch.setattr("src.training.trainer.load_model", lambda checkpoint: (DummyModel(), DummyTokenizer()))
    monkeypatch.setattr("src.training.trainer.load_data", dummy_load_data)
    monkeypatch.setattr("src.training.trainer.generate_output", dummy_generate_output)
    monkeypatch.setattr("src.training.trainer.compute_reward", dummy_compute_reward)
    # Replace the optimizer with our dummy optimizer to bypass torch.optim.Adam.
    monkeypatch.setattr("src.training.trainer.torch.optim.Adam", lambda params, lr: DummyOptimizer())

    trainer = Trainer(dummy_config)
    model = trainer.train()

    # Check that at least one checkpoint file was created.
    checkpoint_files = os.listdir(dummy_config["checkpoint_dir"])
    assert len(checkpoint_files) >= 1
    # Verify that the returned model is not None.
    assert model is not None
