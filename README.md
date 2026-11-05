# Tabular RL vs DQN on Gymnasium Environments

Intelligent Agents, Project 1 (Yachay Tech University). Authors: Alessa Melo and Daniel Troya.

This repository compares **tabular Q-learning** and **Deep Q-Networks (DQN)**, both implemented from scratch in plain
Python and PyTorch (no ready-made agents, no Stable-Baselines3), on Gymnasium environments.

| Tier | Environment | What is done | Notebook |
| --- | --- | --- | --- |
| 1 | CartPole-v1 | Tabular Q-learning (6, 10, 20 bins and 3 bin ranges) vs DQN, same 200k step budget, 5 seeds | `Tier_1_Cartpole/tier1_cartpole_1.ipynb` |
| 2 | LunarLander-v3 | DQN, hyperparameter study (learning rate, target period C) and ablation (no target, no replay, no both), 3 seeds | `Tier_2_LunarLander/tier2_lunarlander.ipynb` |
| 3 (option B) | LunarLander-v3 | Double DQN, Dueling DQN and Prioritized Experience Replay vs vanilla DQN, 5 seeds | `Tier_3_LunarLander/tier3_dqn_improvements.ipynb` |

The analysis-only component (BipedalWalker, MuJoCo locomotion and Atari) has no code; it is Section 7 of the paper.

## Repository structure

```
requirements.txt                      # dependencies for all tiers
Tier_1_Cartpole/
  tier1_cartpole_1.ipynb              # tabular agent, discretizer, DQN, experiments, figures, GIFs
  requirements.txt                    # snapshot of the exact versions used for Tier 1 (written by the notebook)
  results/tier1/
    logs/        *.json               # one log per run: config, eval curve, training returns, times, hardware
    models/      *.npz, *.pt          # Q-tables and network weights
    figures/     *.png, *.gif         # learning curves, stability plots, agent GIFs
    configs.json                      # configuration of every run
    summary.md                        # summary table (source of Table 4 of the paper)
Tier_2_LunarLander/
  tier2_lunarlander.ipynb
  results/tier2/{logs, models, figures, summary.md}
Tier_3_LunarLander/
  tier3_dqn_improvements.ipynb
  results/tier3/{logs, models, figures, summary.md}
```

Each notebook is self-contained: it defines the agents, runs the experiments, and writes the logs, models,
figures, GIFs and `summary.md` of its tier.

## Setup

Python 3.11 (CPU only, no GPU needed).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install swig                  # needed to build Box2D for LunarLander
pip install -r requirements.txt
```

## Reproducing every figure and table

Run each command from the repository root. One command per experiment (in Windows PowerShell 5, replace `&&` with `;`):

| Experiment | Command | Time to retrain from scratch |
| --- | --- | --- |
| Tier 1 | `cd Tier_1_Cartpole && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 tier1_cartpole_1.ipynb` | about 1 h (30 runs, sequential) |
| Tier 2 | `cd Tier_2_LunarLander && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 tier2_lunarlander.ipynb` | about 1 h (24 runs, 12 in parallel) |
| Tier 3 | `cd Tier_3_LunarLander && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 tier3_dqn_improvements.ipynb` | about 1.5 h (25 runs, 12 in parallel) |

**Important: saved runs are reused.** Before training, each notebook checks `results/tierX/logs/`. A run whose log
already exists is loaded instead of trained again. So:

- With the committed logs, the command above only **regenerates all figures, tables and GIFs** (a few minutes).
- To **retrain from scratch**, delete `results/tierX/logs/` first (the models and figures are overwritten).

Times were measured on an Intel Core i9 with 32 logical cores, one PyTorch thread per run. Tiers 2 and 3 train up
to `N_JOBS = 12` runs in parallel with `joblib`; set `N_JOBS = 1` in the configuration cell to train sequentially
(much slower).

### Where each figure and table of the paper comes from

Paper figures were copied from the files below and renamed in the paper folder (for example
`dqn_stability.png` became `t1_dqn_stability.png`).

| Paper | Repository file | Notebook section |
| --- | --- | --- |
| Fig. 3 left | `Tier_1_Cartpole/results/tier1/figures/tab_vs_dqn_eval.png` | Tier 1, §6.3 |
| Fig. 3 center | `Tier_1_Cartpole/results/tier1/figures/tab_ranges_eval.png` | Tier 1, §6.2 |
| Fig. 3 right | `Tier_1_Cartpole/results/tier1/figures/tab_vs_dqn_walltime.png` | Tier 1, §6.3 |
| Fig. 4 | `Tier_1_Cartpole/results/tier1/figures/dqn_stability.png` | Tier 1, §6.4 |
| Table 4 | `Tier_1_Cartpole/results/tier1/summary.md` | Tier 1, §6.5 |
| Fig. 5 left | `Tier_2_LunarLander/results/tier2/figures/study_lr.png` (left panel, evaluation) | Tier 2, §5.2 |
| Fig. 5 right | `Tier_2_LunarLander/results/tier2/figures/study_C.png` (left panel, evaluation) | Tier 2, §5.3 |
| Fig. 6 | `Tier_2_LunarLander/results/tier2/figures/ablation.png` | Tier 2, §5.4 |
| Table 5 | `Tier_2_LunarLander/results/tier2/summary.md` | Tier 2, §5.6 |
| Fig. 7 | `Tier_3_LunarLander/results/tier3/figures/all_configs.png` | Tier 3, §7.2 |
| Fig. 8 top | `Tier_3_LunarLander/results/tier3/figures/overestimation.png` | Tier 3, §7.3 |
| Fig. 8 bottom | `Tier_3_LunarLander/results/tier3/figures/final_returns.png` | Tier 3, §7.5 |
| Table 6 | `Tier_3_LunarLander/results/tier3/summary.md` | Tier 3, §7.6 |

Other numbers quoted in the paper are printed by the notebooks: visited cells per bin count (Tier 1, §6.1), DQN
stability per seed (Tier 1, §6.4), steps where the mean curve passes 200 (Tier 1, §9), random policy return
(Tier 2, §1), estimation bias (Tier 3, §7.3) and bootstrap confidence intervals (Tier 3, §7.5).
Figures 1, 2 and 9 of the paper are environment images from the Gymnasium documentation, not results.

Figures produced but not shown in the paper (for example the training curves of Tier 1, `tab_vs_dqn_train.png`
and `tab_bins_train.png`, and the per-improvement plots of Tier 3) are in the same `figures/` folders.

## Main results

| Tier | Configuration | Final return (100 greedy episodes) | Seeds solved |
| --- | --- | --- | --- |
| 1 (threshold 475) | Best tabular (10 bins, narrow range) | 330.1 ± 100.9 | 0/5 |
| 1 | DQN | 458.3 ± 54.9 | 3/5 |
| 2 (threshold 200) | Default DQN | 255.6 ± 9.3 | 3/3 |
| 2 | Without replay buffer | -399.0 ± 382.3 | 0/3 |
| 3 (threshold 200) | Vanilla DQN | 230.0 ± 42.3 | 4/5 |
| 3 | Dueling DQN | 251.1 ± 27.5 | 4/5 |

Mean ± standard deviation over seeds. The full tables are in each `summary.md`.

## Trained agents (GIFs)

| Environment | Files |
| --- | --- |
| CartPole-v1 | `Tier_1_Cartpole/results/tier1/figures/tabular_best.gif`, `dqn_best.gif` |
| LunarLander-v3 (Tier 2) | `Tier_2_LunarLander/results/tier2/figures/dqn_default_best.gif`, `dqn_no_target_no_replay_best.gif` |
| LunarLander-v3 (Tier 3) | `Tier_3_LunarLander/results/tier3/figures/vanilla_best.gif`, `double_best.gif`, `dueling_best.gif`, `double_dueling_best.gif`, `double_dueling_per_best.gif` |

Each GIF shows the seed with the highest final return of that configuration.

## Where to change the key settings

| Setting | Where |
| --- | --- |
| Step budget, seeds, evaluation period | Configuration cell of each notebook (`STEP_BUDGET`, `SEEDS`, `EVAL_EVERY`) |
| Bins per dimension and bin ranges (Tier 1) | `BIN_COUNTS` (§5) and `RANGES` (§3.1) |
| Tabular α and ε schedule (Tier 1) | Default arguments of `q_learning_control` (§3.2) |
| DQN learning rate, target period C, buffer, batch, ε schedule | Default arguments of `train_dqn` in each notebook |
| Hyperparameter study and ablation (Tier 2) | `EXPERIMENTS` dictionary (§4) |
| Double, Dueling and PER switches (Tier 3) | `EXPERIMENTS` dictionary (§6) |

Because saved runs are reused, give a modified experiment a new name (file prefix) or delete its logs, otherwise the
old result is loaded.

## Reproducibility notes

- Seeds are fixed for Python `random`, NumPy, PyTorch, the training environment and the evaluation environments.
  Evaluation uses fixed seeds (100,000 and up during training, 200,000 and up for the final 100 episodes), different
  from the training seeds.
- Every log stores its full configuration and the hardware it ran on.
- Training time excludes evaluation time.
- Results can still change slightly with other hardware, library versions or a different number of parallel runs.

## Use of generative AI

Generative AI tools were used under the course policy. The tools, the tasks and the main prompts are listed in
Section 8 and Appendix A of the paper.
