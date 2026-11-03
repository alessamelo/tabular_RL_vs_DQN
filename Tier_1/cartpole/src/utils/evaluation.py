"""Greedy policy evaluation and the metrics used to compare runs."""

from __future__ import annotations

from typing import Protocol

import gymnasium as gym
import numpy as np

SUCCESS_THRESHOLD = 475.0


class GreedyAgent(Protocol):
    def greedy_action(self, observation: np.ndarray) -> int: ...


def evaluate_policy(agent: GreedyAgent, env_id: str, n_episodes: int, base_seed: int) -> np.ndarray:
    """Run ``n_episodes`` greedy (epsilon = 0) episodes on a *fresh* environment.

    Episode ``i`` is reset with seed ``base_seed + i``, so every checkpoint of every run
    is evaluated on a known, reproducible set of initial states. A separate environment
    is used so that evaluation never changes the state of the training environment.

    Returns:
        Array of undiscounted episode returns (capped at 500 by the TimeLimit wrapper).
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    env = gym.make(env_id)
    returns = np.zeros(n_episodes, dtype=np.float64)
    try:
        for i in range(n_episodes):
            observation, _ = env.reset(seed=base_seed + i)
            done = False
            while not done:
                observation, reward, terminated, truncated, _ = env.step(agent.greedy_action(observation))
                returns[i] += reward
                done = terminated or truncated
    finally:
        env.close()
    return returns


# --------------------------------------------------------------------- metrics
def steps_to_threshold(steps: np.ndarray, mean_returns: np.ndarray, threshold: float = SUCCESS_THRESHOLD) -> float:
    """First checkpoint step whose evaluation mean is >= threshold; NaN if never reached."""
    hits = np.flatnonzero(np.asarray(mean_returns) >= threshold)
    return float(steps[hits[0]]) if hits.size else float("nan")


def steps_to_sustained_threshold(
    steps: np.ndarray, mean_returns: np.ndarray, threshold: float = SUCCESS_THRESHOLD, consecutive: int = 2
) -> float:
    """First step that starts a run of ``consecutive`` checkpoints all >= threshold; NaN if none.

    A 10-episode checkpoint is noisy, so a single lucky checkpoint can pass 475. Requiring
    consecutive passes makes the "steps to threshold" metric less sensitive to luck.
    """
    ok = np.asarray(mean_returns) >= threshold
    for i in range(len(ok) - consecutive + 1):
        if ok[i:i + consecutive].all():
            return float(steps[i])
    return float("nan")


def count_post_success_drops(mean_returns: np.ndarray, threshold: float = SUCCESS_THRESHOLD, drop: float = 100.0) -> int:
    """Number of checkpoint-to-checkpoint drops larger than ``drop`` after the first success.

    A stability measure: a stable learner that reached the threshold rarely loses
    more than ``drop`` return between consecutive evaluations.
    """
    r = np.asarray(mean_returns, dtype=np.float64)
    hits = np.flatnonzero(r >= threshold)
    if hits.size == 0:
        return 0
    after = r[hits[0]:]
    return int(np.sum(np.diff(after) < -drop))


# Two-sided 95% Student-t critical values t_{0.975, df}; avoids a SciPy dependency.
_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_confidence_interval(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and half-width of a 95% Student-t confidence interval across seeds.

    half_width = t_{0.975, n-1} * s / sqrt(n), with s the sample standard deviation
    (ddof = 1). With n = 5 seeds, t_{0.975, 4} = 2.776, so the interval is wide. That is
    intentional: it shows how little 5 seeds can tell us.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    t_crit = _T_975.get(n - 1, 1.96)  # normal approximation for large n
    sem = values.std(axis=axis, ddof=1) / np.sqrt(n)
    return mean, t_crit * sem
