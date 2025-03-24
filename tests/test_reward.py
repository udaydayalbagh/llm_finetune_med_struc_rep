import pytest
from src.models.reward import compute_reward

def test_compute_reward_valid_output():
    # Create an output string containing key elements: "CREATE", a node, and a relationship.
    output_text = (
        "CREATE (:Patient {id: '123', name: 'John Doe'}) "
        "-[:VISITED]-> (:Hospital {name: 'General Hospital'})"
    )
    reward = compute_reward(output_text)
    # With our heuristics, expect reward to be at least 3.5:
    # +1 for keyword, +1 for node, +1 for relationship, +0.5 for sufficient length.
    assert reward >= 3.5

def test_compute_reward_invalid_output():
    # Output with no proper Neo4j structure.
    output_text = "Some random text without proper structure."
    reward = compute_reward(output_text)
    # Expect a low reward since key patterns are missing.
    assert reward < 1.0
