"""Train one agent for one seed.

Examples (run from the repository root):
    python -m scripts.train --agent tabular --bins 10 --ranges primary --seed 0
    python -m scripts.train --agent dqn --seed 3
    python -m scripts.train --agent dqn --seed 100 --set dqn.learning_rate=0.001 --tag lr1e-3
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.tabular import QLearningAgent  # noqa: E402
from src.environments.discretizer import UniformDiscretizer  # noqa: E402
from src.training import TrainingConfig, train  # noqa: E402
from src.utils.reproducibility import set_global_seeds  # noqa: E402
from src.utils.schedules import LinearSchedule  # noqa: E402


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    """Load the YAML config and apply ``section.key=value`` overrides (values parsed as YAML)."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    config = yaml.safe_load(path.read_text())
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"unknown config section {key!r} in override {item!r}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown config key {leaf!r} in override {item!r}")
        node[leaf] = yaml.safe_load(raw)
    return config


def run_name_for(agent: str, bins: int | None, ranges: str | None, tag: str | None) -> str:
    name = f"tabular_b{bins}_{ranges}" if agent == "tabular" else "dqn"
    return f"{name}_{tag}" if tag else name


def build_and_train(
    agent_kind: str,
    seed: int,
    config: dict[str, Any],
    bins: int | None = None,
    ranges: str = "primary",
    tag: str | None = None,
    results_root: Path = ROOT / "results",
    verbose: bool = True,
) -> Path:
    """Build the requested agent, train it, and return its log directory."""
    config = copy.deepcopy(config)
    rng = set_global_seeds(seed)
    env = gym.make(config["training"]["env_id"])
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    env.close()

    run_name = run_name_for(agent_kind, bins, ranges, tag)
    subdir = "tabular" if agent_kind == "tabular" else "dqn"
    log_dir = results_root / subdir / "logs" / run_name / f"seed_{seed}"

    if agent_kind == "tabular":
        tab = config["tabular"]
        if bins is None:
            raise ValueError("--bins is required for the tabular agent")
        if ranges not in tab["ranges"]:
            raise ValueError(f"unknown range set {ranges!r}; choose from {list(tab['ranges'])}")
        r = tab["ranges"][ranges]
        discretizer = UniformDiscretizer(r["low"], r["high"], bins)
        agent = QLearningAgent(discretizer, n_actions, tab["learning_rate"], config["gamma"], rng, tab["initial_q"])
        eps_cfg = tab["epsilon"]
        model_path = results_root / subdir / "models" / run_name / f"seed_{seed}.npz"
        metadata = {"run_name": run_name, "agent": "tabular", "bins": bins, "ranges": ranges, "agent_config": tab}
    elif agent_kind == "dqn":
        from src.agents.dqn import DQNAgent  # PyTorch only needed here

        d = config["dqn"]
        agent = DQNAgent(
            obs_dim, n_actions, rng,
            hidden_sizes=tuple(d["hidden_sizes"]),
            learning_rate=d["learning_rate"],
            gamma=config["gamma"],
            buffer_size=d["buffer_size"],
            batch_size=d["batch_size"],
            learning_starts=d["learning_starts"],
            train_frequency=d["train_frequency"],
            target_update_interval=d["target_update_interval"],
            max_grad_norm=d["max_grad_norm"],
        )
        eps_cfg = d["epsilon"]
        model_path = results_root / subdir / "models" / run_name / f"seed_{seed}.pt"
        metadata = {"run_name": run_name, "agent": "dqn", "agent_config": d}
    else:
        raise ValueError(f"unknown agent {agent_kind!r}")

    metadata["gamma"] = config["gamma"]
    schedule = LinearSchedule(eps_cfg["start"], eps_cfg["end"], eps_cfg["duration"])
    train(agent, schedule, TrainingConfig(**config["training"]), seed, log_dir, model_path, metadata, verbose)
    return log_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["tabular", "dqn"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bins", type=int, help="bins per dimension (tabular only)")
    parser.add_argument("--ranges", default="primary", help="range set name from the config (tabular only)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cartpole.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. dqn.learning_rate=1e-3")
    parser.add_argument("--tag", help="suffix for the run name (use with --set)")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    build_and_train(args.agent, args.seed, config, args.bins, args.ranges, args.tag, args.results_root)


if __name__ == "__main__":
    main()
