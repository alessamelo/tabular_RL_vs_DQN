"""Targeted unit tests for the parts that are easy to get subtly wrong.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.replay_buffer import ReplayBuffer  # noqa: E402
from src.agents.tabular import QLearningAgent  # noqa: E402
from src.environments.discretizer import UniformDiscretizer  # noqa: E402
from src.utils.evaluation import count_post_success_drops, steps_to_sustained_threshold, steps_to_threshold, t_confidence_interval  # noqa: E402
from src.utils.schedules import LinearSchedule  # noqa: E402

LOW = [-2.4, -2.0, -0.2095, -3.0]
HIGH = [2.4, 2.0, 0.2095, 3.0]


# ------------------------------------------------------------------ discretizer
@pytest.mark.parametrize("bins", [6, 10, 20])
def test_state_count_and_range(bins: int) -> None:
    d = UniformDiscretizer(LOW, HIGH, bins)
    assert d.n_states == bins**4
    rng = np.random.default_rng(0)
    for obs in rng.uniform(-5, 5, size=(500, 4)):
        assert 0 <= d(obs) < d.n_states


def test_out_of_range_saturates_to_edge_bins() -> None:
    d = UniformDiscretizer(LOW, HIGH, 6)
    assert d.bin_indices(np.array([-100.0, -100.0, -100.0, -100.0])).tolist() == [0, 0, 0, 0]
    assert d.bin_indices(np.array([100.0, 100.0, 100.0, 100.0])).tolist() == [5, 5, 5, 5]
    assert d.out_of_range(np.array([0.0, 2.5, 0.0, -3.1])).tolist() == [False, True, False, True]


def test_zero_is_a_bin_edge_so_sign_is_preserved() -> None:
    d = UniformDiscretizer(LOW, HIGH, 10)
    left = d.bin_indices(np.array([-1e-6] * 4))
    right = d.bin_indices(np.array([1e-6] * 4))
    assert np.all(left == 4) and np.all(right == 5)


def test_distinct_cells_map_to_distinct_states() -> None:
    d = UniformDiscretizer(LOW, HIGH, 6)
    centers = [np.linspace(lo, hi, 7)[:-1] + (hi - lo) / 12 for lo, hi in zip(LOW, HIGH)]
    grid = np.array(np.meshgrid(*centers, indexing="ij")).reshape(4, -1).T
    assert len({d(obs) for obs in grid}) == 6**4


def test_discretizer_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        UniformDiscretizer([0, 0], [0, 1], 4)          # high == low
    with pytest.raises(ValueError):
        UniformDiscretizer(LOW, HIGH, 1)               # fewer than 2 bins
    d = UniformDiscretizer(LOW, HIGH, 6)
    with pytest.raises(ValueError):
        d(np.zeros(3))                                  # wrong shape
    with pytest.raises(ValueError):
        d(np.array([0.0, np.nan, 0.0, 0.0]))            # non-finite


# -------------------------------------------------------------------- Q-Learning
def _agent() -> QLearningAgent:
    return QLearningAgent(UniformDiscretizer(LOW, HIGH, 6), 2, learning_rate=0.5, gamma=0.9, rng=np.random.default_rng(0))


def test_terminal_update_does_not_bootstrap() -> None:
    agent = _agent()
    s, s_next = np.zeros(4), np.full(4, 1.0)
    agent.q_table[agent.discretizer(s_next)] = 100.0  # would leak in if we bootstrapped
    agent.update(s, 1, 1.0, s_next, terminated=True)
    assert agent.q_table[agent.discretizer(s), 1] == pytest.approx(0.5 * 1.0)


def test_non_terminal_update_bootstraps_including_truncation() -> None:
    agent = _agent()
    s, s_next = np.zeros(4), np.full(4, 1.0)
    agent.q_table[agent.discretizer(s_next)] = [100.0, 40.0]
    agent.update(s, 0, 1.0, s_next, terminated=False)  # a truncated step is passed as terminated=False
    assert agent.q_table[agent.discretizer(s), 0] == pytest.approx(0.5 * (1.0 + 0.9 * 100.0))


def test_greedy_and_epsilon_one() -> None:
    agent = _agent()
    s = np.zeros(4)
    agent.q_table[agent.discretizer(s)] = [0.0, 1.0]
    assert agent.greedy_action(s) == 1
    actions = {agent.select_action(s, epsilon=1.0) for _ in range(200)}
    assert actions == {0, 1}


# ---------------------------------------------------------------- replay buffer
def test_replay_buffer_circular_overwrite_and_sample_shapes() -> None:
    buf = ReplayBuffer(capacity=3, obs_dim=4)
    for i in range(5):
        buf.add(np.full(4, i), i % 2, 1.0, np.full(4, i + 1), terminated=(i == 4))
    assert len(buf) == 3
    assert sorted(buf.observations[:, 0].tolist()) == [2.0, 3.0, 4.0]  # 0 and 1 overwritten
    batch = buf.sample(8, np.random.default_rng(0))
    assert batch.observations.shape == (8, 4) and batch.actions.shape == (8,)
    assert set(batch.terminated.tolist()) <= {0.0, 1.0}


def test_replay_buffer_empty_sample_raises() -> None:
    with pytest.raises(RuntimeError):
        ReplayBuffer(4, 4).sample(2, np.random.default_rng(0))


# ------------------------------------------------------------------------- DQN
def test_dqn_target_zeroes_bootstrap_only_for_terminal() -> None:
    torch = pytest.importorskip("torch")
    from src.agents.dqn import DQNAgent

    agent = DQNAgent(4, 2, np.random.default_rng(0), batch_size=2, learning_starts=2, gamma=0.5)
    next_obs = torch.zeros(2, 4)
    with torch.no_grad():
        next_max = agent.target_network(next_obs).max(dim=1).values
    y = agent.compute_targets(torch.tensor([1.0, 1.0]), next_obs, torch.tensor([0.0, 1.0]))
    assert y[0].item() == pytest.approx(1.0 + 0.5 * next_max[0].item(), rel=1e-6)
    assert y[1].item() == pytest.approx(1.0)


def test_dqn_target_network_syncs_periodically() -> None:
    torch = pytest.importorskip("torch")
    from src.agents.dqn import DQNAgent

    agent = DQNAgent(4, 2, np.random.default_rng(0), batch_size=4, learning_starts=4, target_update_interval=3, learning_rate=1e-2)

    def same() -> bool:
        return all(torch.equal(a, b) for a, b in zip(agent.online_network.parameters(), agent.target_network.parameters()))

    rng = np.random.default_rng(1)
    for _ in range(4):  # fills buffer; 4th call performs gradient step 1
        agent.update(rng.normal(size=4), 0, 1.0, rng.normal(size=4), False)
    assert not same()  # online moved, target frozen
    for _ in range(2):  # gradient steps 2 and 3 -> sync at 3
        agent.update(rng.normal(size=4), 1, 1.0, rng.normal(size=4), False)
    assert same()


# --------------------------------------------------------------------- utilities
def test_linear_schedule() -> None:
    s = LinearSchedule(1.0, 0.05, 100)
    assert s(0) == 1.0 and s(100) == pytest.approx(0.05) and s(10_000) == pytest.approx(0.05)
    assert s(50) == pytest.approx(0.525)


def test_metrics() -> None:
    steps = np.array([0, 5, 10, 15, 20])
    curve = np.array([10, 480, 300, 500, 490])
    assert steps_to_threshold(steps, curve) == 5
    assert np.isnan(steps_to_threshold(steps, np.zeros(5)))
    assert count_post_success_drops(curve) == 1
    assert steps_to_sustained_threshold(steps, curve) == 15
    mean, half = t_confidence_interval(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert mean == 3.0 and half == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
