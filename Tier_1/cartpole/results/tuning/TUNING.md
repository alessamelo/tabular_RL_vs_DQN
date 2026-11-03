# Hyperparameter selection on tuning seeds (100, 101, 102)

These runs are **not** reported results. Each method got the same tuning effort: its Stage 1 default plus **one** alternative, with the full 200k-step budget on 3 tuning seeds.
Decision rule (fixed before looking at the results): keep the variant with the higher mean final 100-episode greedy return across tuning seeds.

| Run | seed 100 | seed 101 | seed 102 | mean |
|---|---|---|---|---|
| dqn | 231.0 | 84.5 | 290.9 | 202.1 |
| dqn_lr1e-4_tu1000 | 500.0 | 500.0 | 500.0 | 500.0 |
| tabular_b10_primary | 450.9 | 224.6 | 219.1 | 298.2 |
| tabular_b10_primary_alpha0.05 | 152.1 | 170.2 | 171.8 | 164.7 |
| tabular_b10_tight | 442.9 | 407.4 | 313.2 | 387.8 |
| tabular_b10_tight_alpha0.05 | 274.1 | 155.7 | 92.2 | 174.0 |
| tabular_b20_primary | 377.1 | 243.7 | 338.4 | 319.7 |
| tabular_b20_primary_alpha0.05 | 102.9 | 66.9 | 154.3 | 108.0 |
| tabular_b6_primary | 181.2 | 241.3 | 212.0 | 211.5 |
| tabular_b6_primary_alpha0.05 | 201.4 | 184.2 | 190.0 | 191.9 |

**Outcome**
- DQN: `lr=1e-4, target_update_interval=1000` (alternative) replaces the default `lr=5e-4, target_update_interval=500`. The default reached 500 at several checkpoints but repeatedly collapsed afterwards (see `dqn/logs/dqn/seed_*/evals.csv`).
- Q-Learning: the default `alpha=0.1` is kept; `alpha=0.05` was worse at every granularity.
- Note: on the tuning seeds, the tight-range 10-bin agent scored higher than the primary-range one. The ranges are an experimental factor, not a tuned hyperparameter, so both are run on the reported seeds.
- Naming: in this folder the untagged `dqn` run is the *old default* (lr 5e-4, sync 500), produced before the config was updated. `dqn_lr1e-4_tu1000` equals the current config.
