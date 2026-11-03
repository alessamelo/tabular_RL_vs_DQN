"""Fixed-capacity experience replay buffer (circular, preallocated NumPy arrays).

Why replay: consecutive CartPole transitions are strongly correlated (state t+1 is
almost state t). SGD assumes roughly i.i.d. samples; training on a stream of
correlated samples gives biased, high-variance gradient steps and makes the network
overfit the most recent part of the trajectory (catastrophic forgetting). Sampling
uniformly from a large buffer of past transitions mixes data from many episodes and
policies, which (1) breaks the temporal correlation inside a minibatch and
(2) re-uses each transition many times (better sample efficiency).

The ``terminated`` flag is stored, **not** ``terminated or truncated``. A transition
that ended by time-limit truncation must still be bootstrapped (see ``dqn.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionBatch:
    observations: np.ndarray       # (B, obs_dim) float32
    actions: np.ndarray            # (B,)        int64
    rewards: np.ndarray            # (B,)        float32
    next_observations: np.ndarray  # (B, obs_dim) float32
    terminated: np.ndarray         # (B,)        float32 (1.0 = absorbing next state)


class ReplayBuffer:
    """Circular buffer: once full, the oldest transition is overwritten."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        if capacity <= 0 or obs_dim <= 0:
            raise ValueError("capacity and obs_dim must be positive")
        self.capacity = capacity
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._next_index
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.terminated[i] = float(terminated)
        self._next_index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        """Sample ``batch_size`` transitions uniformly at random (with replacement)."""
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            terminated=self.terminated[idx],
        )
