"""Tabular Q-Learning over a discretized CartPole state.

Update rule (Watkins & Dayan, 1992), applied after every environment step:

    target = r                                   if the episode TERMINATED at s'
    target = r + gamma * max_a' Q(s', a')        otherwise (including TRUNCATION)
    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

Why ``terminated`` and not ``terminated or truncated``: a terminated state (pole fell,
cart left the track) is absorbing, so its future value is exactly 0. A truncated
state (500-step time limit) is an ordinary state where the pole is still up; the
episode was cut for practical reasons and the true return continues, so we must still
bootstrap from it. The observation contains no clock, so treating the time limit as
terminal would teach the agent that well-balanced states are worth nothing.

Q-Learning is *off-policy*: the target uses ``max_a'`` (the greedy policy) even
though actions are chosen epsilon-greedily, so it learns Q* rather than Q^{eps}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.environments.discretizer import UniformDiscretizer


class QLearningAgent:
    """Epsilon-greedy Q-Learning with a dense ``(n_states, n_actions)`` table."""

    def __init__(
        self,
        discretizer: UniformDiscretizer,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        rng: np.random.Generator,
        initial_q: float = 0.0,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate (alpha) must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if n_actions < 1:
            raise ValueError("n_actions must be positive")

        self.discretizer = discretizer
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rng = rng
        self.q_table = np.full((discretizer.n_states, n_actions), initial_q, dtype=np.float64)
        # Diagnostics (not used for learning).
        self.visit_counts = np.zeros(discretizer.n_states, dtype=np.int64)
        self.out_of_range_counts = np.zeros(discretizer.n_dims, dtype=np.int64)
        self.n_updates = 0

    # ------------------------------------------------------------------ acting
    def greedy_action(self, observation: np.ndarray) -> int:
        """argmax_a Q(s, a). Ties -> lowest index, so evaluation never touches the RNG."""
        return int(np.argmax(self.q_table[self.discretizer(observation)]))

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: random action with probability epsilon, else greedy.

        During training, ties in the greedy action are broken uniformly at random.
        Otherwise every unvisited state (all-zero row) would always pick action 0,
        which biases early behaviour toward pushing left.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q_values = self.q_table[self.discretizer(observation)]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best_actions))

    # ---------------------------------------------------------------- learning
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> float:
        """Apply one Q-Learning update and return the TD error (target - Q(s, a))."""
        state = self.discretizer(observation)
        if terminated:
            target = reward  # absorbing state: no bootstrap
        else:
            next_state = self.discretizer(next_observation)
            target = reward + self.gamma * float(self.q_table[next_state].max())

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        self.visit_counts[state] += 1
        self.out_of_range_counts += self.discretizer.out_of_range(observation)
        self.n_updates += 1
        return float(td_error)

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, float]:
        """Coverage and saturation statistics, logged at the end of training."""
        visited = self.visit_counts > 0
        return {
            "n_states": int(self.discretizer.n_states),
            "states_visited": int(visited.sum()),
            "fraction_states_visited": float(visited.mean()),
            "median_visits_per_visited_state": float(np.median(self.visit_counts[visited])) if visited.any() else 0.0,
            "q_table_bytes": int(self.q_table.nbytes),
            **{
                f"out_of_range_fraction_dim{i}": float(c / max(self.n_updates, 1))
                for i, c in enumerate(self.out_of_range_counts)
            },
        }

    def save(self, path: str | Path) -> None:
        """Save the Q-table, visit counts and grid definition to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q_table=self.q_table,
            visit_counts=self.visit_counts,
            lows=self.discretizer.lows,
            highs=self.discretizer.highs,
            bins=self.discretizer.bins,
        )
