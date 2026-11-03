"""Aggregate all run logs into tables and figures.

    python -m scripts.analyze                      # reported seeds -> results/
    python -m scripts.analyze --root results/tuning

Outputs:
    <root>/comparison/per_run_metrics.csv   one row per (configuration, seed)
    <root>/comparison/summary.csv           one row per configuration (aggregated over seeds)
    <root>/comparison/figures/*.png, <root>/tabular/figures/*.png, <root>/dqn/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import plotting  # noqa: E402
from src.utils.evaluation import SUCCESS_THRESHOLD, count_post_success_drops, steps_to_sustained_threshold, steps_to_threshold, t_confidence_interval  # noqa: E402

ORDER = ["dqn", "tabular_b6_primary", "tabular_b10_primary", "tabular_b20_primary", "tabular_b10_tight",
         "tabular_b6_primary_eps50k", "tabular_b10_primary_eps50k", "tabular_b20_primary_eps50k"]  # eps50k = post-hoc ablation


def load_runs(root: Path) -> dict[str, list[dict]]:
    """run_name -> list of {seed, summary, evals, episodes, stats} sorted by seed."""
    runs: dict[str, list[dict]] = {}
    for summary_path in sorted(root.glob("*/logs/*/seed_*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        runs.setdefault(summary["run_name"], []).append({
            "seed": summary["seed"],
            "summary": summary,
            "evals": pd.read_csv(run_dir / "evals.csv"),
            "episodes": pd.read_csv(run_dir / "episodes.csv"),
            "stats": pd.read_csv(run_dir / "train_stats.csv"),
        })
    for name in runs:
        runs[name].sort(key=lambda r: r["seed"])
    if not runs:
        raise FileNotFoundError(f"no summary.json files found under {root}")
    return runs


def per_run_table(runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for name, seeds in runs.items():
        for r in seeds:
            s, ev = r["summary"], r["evals"]
            steps, means = ev["step"].to_numpy(), ev["mean_return"].to_numpy()
            diag = s["diagnostics"]
            rows.append({
                "run_name": name,
                "seed": r["seed"],
                "final_eval_mean_100": s["final_eval_mean"],
                "final_eval_std_100": s["final_eval_std"],
                "final_success": s["final_eval_mean"] >= SUCCESS_THRESHOLD,
                "steps_to_threshold": steps_to_threshold(steps, means),
                "steps_to_sustained_threshold": steps_to_sustained_threshold(steps, means),
                "best_checkpoint_mean_10": means.max(),
                "best_checkpoint_step": int(steps[means.argmax()]),
                "last_checkpoint_mean_10": means[-1],
                "mean_eval_over_training": means.mean(),  # normalised area under the curve
                "post_success_drops_gt100": count_post_success_drops(means),
                "train_time_s": s["train_time_s"],
                "n_training_episodes": s["n_training_episodes"],
                **{f"diag_{k}": v for k, v in diag.items()},
            })
    df = pd.DataFrame(rows)
    df["order"] = df["run_name"].map({n: i for i, n in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)


def summary_table(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in per_run.groupby("run_name", sort=False):
        final = g["final_eval_mean_100"].to_numpy()
        mean, half = t_confidence_interval(final)
        reached = g["steps_to_threshold"].dropna()
        sustained = g["steps_to_sustained_threshold"].dropna()
        row = {
            "run_name": name,
            "n_seeds": len(g),
            "final_mean": mean,
            "final_ci95_half_width": half,
            "final_std_across_seeds": final.std(ddof=1) if len(final) > 1 else 0.0,
            "final_min": final.min(),
            "final_max": final.max(),
            "seeds_final_success": int(g["final_success"].sum()),
            "seeds_reaching_threshold_ever": len(reached),
            "median_steps_to_threshold": reached.median() if len(reached) else np.nan,
            "seeds_sustained_threshold": len(sustained),
            "median_steps_to_sustained_threshold": sustained.median() if len(sustained) else np.nan,
            "best_checkpoint_mean": g["best_checkpoint_mean_10"].mean(),
            "mean_eval_over_training": g["mean_eval_over_training"].mean(),
            "post_success_drops_total": int(g["post_success_drops_gt100"].sum()),
            "train_time_s_mean": g["train_time_s"].mean(),
            "train_time_s_std": g["train_time_s"].std(ddof=1) if len(g) > 1 else 0.0,
        }
        for col in ("diag_n_states", "diag_fraction_states_visited", "diag_q_table_bytes", "diag_n_parameters",
                    "diag_replay_buffer_bytes", "diag_out_of_range_fraction_dim1", "diag_out_of_range_fraction_dim3"):
            if col in g and g[col].notna().any():
                row[col.removeprefix("diag_")] = g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def eval_curves(runs: dict[str, list[dict]], names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for n in names:
        if n in runs:
            steps = runs[n][0]["evals"]["step"].to_numpy()
            out[n] = (steps, np.stack([r["evals"]["mean_return"].to_numpy() for r in runs[n]]))
    return out


def training_curves(runs: dict[str, list[dict]], names: list[str], window: int = 5_000, total: int = 200_000):
    """Mean epsilon-greedy training return of episodes ending in each ``window``-step bin."""
    grid = np.arange(window, total + 1, window)
    out = {}
    for n in names:
        if n not in runs:
            continue
        per_seed = []
        for r in runs[n]:
            ep = r["episodes"]
            bins = np.ceil(ep["end_step"].to_numpy() / window).astype(int) * window
            means = pd.Series(ep["return"].to_numpy()).groupby(bins).mean()
            per_seed.append(means.reindex(grid).to_numpy())
        out[n] = (grid, np.stack(per_seed))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root

    runs = load_runs(root)
    per_run = per_run_table(runs)
    summary = summary_table(per_run)
    (root / "comparison").mkdir(parents=True, exist_ok=True)
    per_run.to_csv(root / "comparison" / "per_run_metrics.csv", index=False)
    summary.to_csv(root / "comparison" / "summary.csv", index=False)

    seeds = sorted({r["seed"] for v in runs.values() for r in v})
    main_names = ["dqn", "tabular_b6_primary", "tabular_b10_primary", "tabular_b20_primary"]
    tab_names = ["tabular_b6_primary", "tabular_b10_primary", "tabular_b20_primary"]
    abl_names = ["tabular_b10_primary", "tabular_b10_tight"]
    cmp_fig, tab_fig, dqn_fig = root / "comparison" / "figures", root / "tabular" / "figures", root / "dqn" / "figures"
    for d in (cmp_fig, tab_fig, dqn_fig):
        d.mkdir(parents=True, exist_ok=True)

    plotting.plot_learning_curves(eval_curves(runs, main_names), cmp_fig / "learning_curves_ci95.png", "Q-Learning vs DQN on CartPole-v1")
    plotting.plot_learning_curves(eval_curves(runs, main_names), cmp_fig / "learning_curves_std.png", "Q-Learning vs DQN on CartPole-v1", band="std")
    plotting.plot_per_seed_curves(eval_curves(runs, [n for n in ORDER[:5] if n in runs]), seeds, cmp_fig / "per_seed_curves.png")
    plotting.plot_final_returns({n: per_run.loc[per_run.run_name == n, "final_eval_mean_100"].to_numpy() for n in ORDER[:5] if n in runs}, cmp_fig / "final_returns.png")
    plotting.plot_training_returns(training_curves(runs, main_names), cmp_fig / "training_returns.png")
    plotting.plot_learning_curves(eval_curves(runs, tab_names), tab_fig / "granularity_ci95.png", "Q-Learning: effect of bins per dimension")
    plotting.plot_learning_curves(eval_curves(runs, abl_names), tab_fig / "range_ablation_ci95.png", "Q-Learning (10 bins): primary vs tight velocity ranges")
    eps_names = ["tabular_b10_primary", "tabular_b10_primary_eps50k"]
    if eps_names[1] in runs:
        plotting.plot_learning_curves(eval_curves(runs, eps_names), tab_fig / "epsilon_ablation_ci95.png", "Q-Learning (10 bins): ε decay over 100k (pre-registered) vs 50k (post-hoc)")
    if "dqn" in runs:
        stats = [r["stats"] for r in runs["dqn"]]
        plotting.plot_dqn_diagnostics(
            stats[0]["step"].to_numpy(),
            np.stack([s["mean_abs_loss_or_td"].to_numpy() for s in stats]),
            stats[0]["epsilon"].to_numpy(),
            dqn_fig / "loss_and_epsilon.png",
        )

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summary.T)


if __name__ == "__main__":
    main()
