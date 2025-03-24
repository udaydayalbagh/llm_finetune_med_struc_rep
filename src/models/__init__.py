"""
The models package provides functionality for loading the deepseek-r1 model,
generating outputs, and computing rewards for reinforcement learning.
"""

from .model import load_model, generate_output
from .reward import structured_report_reward, compute_neo4j_graph_reward
