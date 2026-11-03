"""Plotting helpers (Matplotlib, static PNG figures).

Every configuration keeps the same color in every figure (color follows the entity),
and every multi-series figure has both a legend and direct end-of-line labels, so
identity never depends on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.evaluation import SUCCESS_THRESHOLD, t_confidence_interval  # noqa: E402

# Fixed categorical order (validated colorblind-safe palette, light surface).
RUN_COLORS = {
    "dqn": "#2a78d6",
    "tabular_b6_primary": "#eb6834",
    "tabular_b10_primary": "#1baf7a",
    "tabular_b20_primary": "#eda100",
    "tabular_b10_tight": "#e87ba4",
    "tabular_b10_primary_eps50k": "#6250d6",
    "tabular_b6_primary_eps50k": "#6250d6",
    "tabular_b20_primary_eps50k": "#6250d6",
}
RUN_LABELS = {
    "dqn": "DQN",
    "tabular_b6_primary": "Q-Learning, 6 bins",
    "tabular_b10_primary": "Q-Learning, 10 bins",
    "tabular_b20_primary": "Q-Learning, 20 bins",
    "tabular_b10_tight": "Q-Learning, 10 bins (tight ranges)",
    "tabular_b6_primary_eps50k": "Q-Learning, 6 bins, ε decay 50k (post-hoc)",
    "tabular_b10_primary_eps50k": "Q-Learning, 10 bins, ε decay 50k (post-hoc)",
    "tabular_b20_primary_eps50k": "Q-Learning, 20 bins, ε decay 50k (post-hoc)",
    "tabular_b10_tight_eps50k": "Q-Learning, 10 bins tight, ε decay 50k (post-hoc)",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _threshold(ax: plt.Axes) -> None:
    ax.axhline(SUCCESS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[0], SUCCESS_THRESHOLD + 6, " success threshold (475)", color=MUTED, fontsize=8, va="bottom")


def plot_learning_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    title: str,
    band: str = "ci",
) -> None:
    """Mean greedy evaluation return across seeds with a shaded 95% CI or +/- 1 std band.

    Args:
        curves: run_name -> (steps of shape (T,), returns of shape (n_seeds, T)).
        band: "ci" for a 95% Student-t interval, "std" for +/- one standard deviation.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ends = []
    for name, (steps, returns) in curves.items():
        if band == "ci":
            mean, half = t_confidence_interval(returns, axis=0)
        else:
            mean, half = returns.mean(axis=0), returns.std(axis=0, ddof=1)
        color = RUN_COLORS.get(name, INK)
        ax.plot(steps, mean, color=color, linewidth=2, label=f"{RUN_LABELS.get(name, name)} (n={returns.shape[0]})")
        ax.fill_between(steps, np.clip(mean - half, 0, 500), np.clip(mean + half, 0, 500), color=color, alpha=0.15, linewidth=0)
        ends.append((mean[-1], RUN_LABELS.get(name, name)))
    ax.set_xlim(0, max(s.max() for s, _ in curves.values()) * 1.0)
    ax.set_ylim(0, 520)
    _style(ax, "Environment steps", "Mean greedy evaluation return (10 episodes)")
    _threshold(ax)
    # Direct labels at line ends, nudged apart vertically to avoid collisions.
    placed: list[float] = []
    for y, label in sorted(ends):
        y_lab = y
        while any(abs(y_lab - p) < 18 for p in placed):
            y_lab += 18
        placed.append(y_lab)
        ax.annotate(label.replace("Q-Learning, ", "QL "), xy=(ax.get_xlim()[1], y), xytext=(6, y_lab - y),
                    textcoords="offset points", color=INK, fontsize=8, va="center", annotation_clip=False)
    band_text = "shaded: 95% t-interval across seeds" if band == "ci" else "shaded: ±1 std across seeds"
    ax.set_title(f"{title}\n{band_text}", color=INK, fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_seed_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], seeds: list[int], path: Path) -> None:
    """Small multiples: one panel per configuration, one thin line per seed."""
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (steps, returns)) in zip(axes, curves.items()):
        color = RUN_COLORS.get(name, INK)
        for r in returns:
            ax.plot(steps, r, color=color, linewidth=1, alpha=0.55)
        ax.plot(steps, returns.mean(axis=0), color=INK, linewidth=2, label="mean")
        ax.axhline(SUCCESS_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(RUN_LABELS.get(name, name), fontsize=9, color=INK)
        ax.set_ylim(0, 520)
        _style(ax, "Env steps", "Eval return" if ax is axes[0] else "")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Per-seed greedy evaluation curves (thin lines = seeds {seeds}; black = mean)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_returns(final: dict[str, np.ndarray], path: Path) -> None:
    """Final 100-episode evaluation mean, one dot per seed, with the across-seed mean."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(final)
    for i, name in enumerate(names):
        vals = final[name]
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=40, color=RUN_COLORS.get(name, INK), edgecolor="white", linewidth=1.5, zorder=3)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.28, vals.mean(), f"{vals.mean():.0f}", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(names)), [RUN_LABELS.get(n, n).replace("Q-Learning, ", "QL\n") for n in names], fontsize=8)
    ax.set_ylim(0, 520)
    _style(ax, "", "Final greedy return (mean of 100 episodes)")
    _threshold(ax)
    ax.set_title("Final policy performance (dots = seeds, bar = mean across seeds)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_returns(train_curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """Epsilon-greedy *training* episode returns (kept separate from evaluation returns)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, (grid, returns) in train_curves.items():
        ax.plot(grid, returns.mean(axis=0), color=RUN_COLORS.get(name, INK), linewidth=2, label=RUN_LABELS.get(name, name))
    ax.set_ylim(0, 520)
    _style(ax, "Environment steps", "Training episode return (ε-greedy)")
    ax.set_title("Training returns under the ε-greedy behaviour policy\n(mean of episodes ending in each 5k-step window, averaged across seeds)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dqn_diagnostics(steps: np.ndarray, losses: np.ndarray, epsilons: np.ndarray, path: Path) -> None:
    """Two separate panels (no dual axis): Huber loss per seed, and the epsilon schedule."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, loss in enumerate(losses):
        ax1.plot(steps, loss, color=RUN_COLORS["dqn"], linewidth=1, alpha=0.6, label="seeds" if i == 0 else None)
    ax1.plot(steps, np.nanmean(losses, axis=0), color=INK, linewidth=2, label="mean")
    ax1.set_yscale("log")
    _style(ax1, "", "Mean Huber loss per 1k steps (log)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("DQN training diagnostics", fontsize=10, color=INK, loc="left")
    ax2.plot(steps, epsilons, color=MUTED, linewidth=2)
    _style(ax2, "Environment steps", "ε")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
