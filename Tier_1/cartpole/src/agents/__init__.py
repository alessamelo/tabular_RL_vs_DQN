from .tabular import QLearningAgent

__all__ = ["QLearningAgent"]
# DQNAgent is imported from src.agents.dqn directly so the tabular code never needs PyTorch.
