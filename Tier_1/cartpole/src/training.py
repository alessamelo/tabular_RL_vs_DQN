"""One training loop shared by both agents.

Using the *same* loop for Q-Learning and DQN is part of the fairness argument: both
agents see the same environment-step budget, the same epsilon indexing, the same
evaluation checkpoints and the same timing method. Only the agent object differs.

Agent interface (duck-typed):
    select_action(obs, epsilon) -> int     # behaviour policy (epsilon-greedy)
    greedy_action(obs) -> int              # evaluation policy (epsilon = 0)
    update(obs, a, r, next_obs, terminated) -> float | None   # learning step
    diagnostics() -> dict
    save(path) -> None
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.utils.evaluation import evaluate_policy
from src.utils.schedules import LinearSchedule


@dataclass
class TrainingConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    final_eval_episodes: int = 100
    stats_every: int = 1_000  # window for averaging loss / |TD error|


@dataclass
class TrainingResult:
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[np.ndarray] = field(default_factory=list)
    final_returns: np.ndarray | None = None
    train_time_s: float = 0.0
    eval_time_s: float = 0.0


def checkpoint_eval_seed(run_seed: int, checkpoint_index: int) -> int:
    """Evaluation seeds are disjoint from training seeds and from each other."""
    return 10_000 + 1_000 * run_seed + 10 * checkpoint_index


def final_eval_seed(run_seed: int) -> int:
    return 1_000_000 + 1_000 * run_seed


def train(
    agent: Any,
    epsilon_schedule: LinearSchedule,
    config: TrainingConfig,
    seed: int,
    log_dir: Path,
    model_path: Path,
    run_metadata: dict[str, Any],
    verbose: bool = True,
) -> TrainingResult:
    """Train ``agent`` for ``config.total_steps`` environment steps and log everything.

    Files written to ``log_dir``:
        episodes.csv     one row per *training* episode (epsilon-greedy returns)
        evals.csv        one row per greedy evaluation checkpoint
        train_stats.csv  epsilon and mean loss (DQN) or mean |TD error| (tabular) per window
        summary.json     config, final 100-episode evaluation, timings, diagnostics
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(config.env_id)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)  # seed once; later resets continue the env RNG

    result = TrainingResult()
    episode_rows: list[tuple] = []
    eval_rows: list[tuple] = []
    stats_rows: list[tuple] = []

    def run_checkpoint(step: int) -> None:
        idx = len(result.eval_steps)
        t0 = time.perf_counter()
        returns = evaluate_policy(agent, config.env_id, config.eval_episodes, checkpoint_eval_seed(seed, idx))
        result.eval_time_s += time.perf_counter() - t0
        result.eval_steps.append(step)
        result.eval_returns.append(returns)
        eval_rows.append((step, returns.mean(), returns.std(), returns.min(), returns.max(), ";".join(f"{x:.0f}" for x in returns)))
        if verbose:
            print(f"[{run_metadata.get('run_name')}|seed {seed}] step {step:>7d}  eval mean {returns.mean():6.1f}  eps {epsilon_schedule(step):.3f}", flush=True)

    run_checkpoint(0)  # untrained policy

    episode_return, episode_length, episode_index = 0.0, 0, 0
    window_values: list[float] = []
    train_clock = time.perf_counter()

    for step in range(1, config.total_steps + 1):
        epsilon = epsilon_schedule(step - 1)
        action = agent.select_action(observation, epsilon)
        next_observation, reward, terminated, truncated, _ = env.step(action)

        # Learn from (s, a, r, s', terminated). Truncation is NOT passed as terminal.
        value = agent.update(observation, action, float(reward), next_observation, bool(terminated))
        if value is not None:
            window_values.append(abs(value))

        episode_return += reward
        episode_length += 1
        if terminated or truncated:
            episode_rows.append((episode_index, step, episode_return, episode_length, epsilon, int(terminated), int(truncated)))
            episode_index += 1
            episode_return, episode_length = 0.0, 0
            observation, _ = env.reset()
        else:
            observation = next_observation

        if step % config.stats_every == 0:
            stats_rows.append((step, epsilon, float(np.mean(window_values)) if window_values else float("nan"), len(window_values)))
            window_values = []

        if step % config.eval_every == 0:
            result.train_time_s += time.perf_counter() - train_clock  # pause the training clock
            run_checkpoint(step)
            train_clock = time.perf_counter()

    result.train_time_s += time.perf_counter() - train_clock
    env.close()

    t0 = time.perf_counter()
    result.final_returns = evaluate_policy(agent, config.env_id, config.final_eval_episodes, final_eval_seed(seed))
    result.eval_time_s += time.perf_counter() - t0

    agent.save(model_path)
    _write_csv(log_dir / "episodes.csv", ("episode", "end_step", "return", "length", "epsilon", "terminated", "truncated"), episode_rows)
    _write_csv(log_dir / "evals.csv", ("step", "mean_return", "std_return", "min_return", "max_return", "returns"), eval_rows)
    _write_csv(log_dir / "train_stats.csv", ("step", "epsilon", "mean_abs_loss_or_td", "n_updates_in_window"), stats_rows)

    summary = {
        **run_metadata,
        "seed": seed,
        "training_config": config.__dict__,
        "final_eval_mean": float(result.final_returns.mean()),
        "final_eval_std": float(result.final_returns.std()),
        "final_eval_returns": result.final_returns.tolist(),
        "train_time_s": result.train_time_s,
        "eval_time_s": result.eval_time_s,
        "n_training_episodes": episode_index,
        "diagnostics": agent.diagnostics(),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[{run_metadata.get('run_name')}|seed {seed}] FINAL 100-ep mean {summary['final_eval_mean']:.1f}  train time {result.train_time_s:.1f}s", flush=True)
    return result


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
