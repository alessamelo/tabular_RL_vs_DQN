"""Deep Q-Network (Mnih et al., 2015) for CartPole, implemented from scratch in PyTorch.

Components
----------
* ``QNetwork``: an MLP that maps the 4-D observation to 2 Q-values, Q(s, .; theta).
  One forward pass gives the values of every action, so the greedy action is an argmax.
* Experience replay (``ReplayBuffer``): breaks temporal correlation (see that module).
* Target network theta^-: a copy of the online network, frozen between periodic hard
  syncs. The regression target is

      y = r + gamma * (1 - terminated) * max_a' Q(s', a'; theta^-)

  If the online network were used for y, every gradient step would also move the
  target (the "moving target" problem). Together with bootstrapping and
  off-policy data this is the "deadly triad" and can make Q-values oscillate or
  diverge. Freezing theta^- for C gradient steps turns each block of C steps into a
  nearly supervised regression problem toward fixed targets.
* Loss: Huber (smooth L1) on (Q(s, a; theta) - y). It is quadratic for small errors
  and linear for large ones, so rare large TD errors (e.g. right after a terminal
  transition) do not produce exploding gradients. Gradient-norm clipping adds a
  second safeguard.
* (1 - terminated) zeroes the bootstrap term only for true terminal transitions.
  Time-limit truncations keep their bootstrap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.agents.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Fully connected network: obs_dim -> hidden -> hidden -> n_actions (ReLU)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = obs_dim
        for width in hidden_sizes:
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            in_features = width
        layers.append(nn.Linear(in_features, n_actions))  # linear output: Q-values are unbounded
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class DQNAgent:
    """DQN with uniform experience replay, a hard-updated target network and Huber loss."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        rng: np.random.Generator,
        hidden_sizes: tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_frequency: int = 1,
        target_update_interval: int = 1000,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ) -> None:
        if learning_starts < batch_size:
            raise ValueError("learning_starts must be >= batch_size")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if min(train_frequency, target_update_interval, batch_size) <= 0:
            raise ValueError("train_frequency, target_update_interval and batch_size must be positive")

        self.n_actions = n_actions
        self.rng = rng
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.online_network = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # never trained directly
        for p in self.target_network.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss with delta = 1
        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim)

        self.env_steps = 0
        self.gradient_steps = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def greedy_action(self, observation: np.ndarray) -> int:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online_network(obs_t).argmax(dim=1).item())

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(observation)

    # ---------------------------------------------------------------- learning
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> float | None:
        """Store the transition and, when due, perform one gradient step.

        Returns the minibatch loss when a gradient step happened, otherwise None.
        """
        self.replay_buffer.add(observation, action, reward, next_observation, terminated)
        self.env_steps += 1
        if len(self.replay_buffer) < self.learning_starts or self.env_steps % self.train_frequency != 0:
            return None
        return self._gradient_step()

    def compute_targets(self, rewards: torch.Tensor, next_observations: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """y = r + gamma * (1 - terminated) * max_a' Q_target(s', a')."""
        with torch.no_grad():
            next_q_max = self.target_network(next_observations).max(dim=1).values
            return rewards + self.gamma * (1.0 - terminated) * next_q_max

    def _gradient_step(self) -> float:
        batch = self.replay_buffer.sample(self.batch_size, self.rng)
        obs = torch.as_tensor(batch.observations, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_obs = torch.as_tensor(batch.next_observations, device=self.device)
        terminated = torch.as_tensor(batch.terminated, device=self.device)

        # Q(s, a; theta) for the action actually taken: gather picks column a for each row.
        q_taken = self.online_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.compute_targets(rewards, next_obs, terminated)
        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())  # hard sync
        return float(loss.item())

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, float]:
        n_params = sum(p.numel() for p in self.online_network.parameters())
        buffer_bytes = sum(
            a.nbytes
            for a in (
                self.replay_buffer.observations,
                self.replay_buffer.next_observations,
                self.replay_buffer.actions,
                self.replay_buffer.rewards,
                self.replay_buffer.terminated,
            )
        )
        return {
            "n_parameters": int(n_params),
            "network_bytes_float32": int(n_params * 4),
            "replay_buffer_bytes": int(buffer_bytes),
            "gradient_steps": int(self.gradient_steps),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_network.state_dict(), path)
