"""Run the full experimental grid (or the tuning grid) in parallel worker processes.

    python -m scripts.run_all                 # reported seeds, all 5 configurations
    python -m scripts.run_all --tuning        # tuning seeds only (written under results/tuning)
    python -m scripts.run_all --workers 4

Runs whose summary.json already exists are skipped, so the script can be resumed.
Each run is a separate process, so seeding inside one run cannot affect another.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import build_and_train, load_config, run_name_for  # noqa: E402


def job_list(config: dict, seeds: list[int]) -> list[dict]:
    jobs = []
    for seed in seeds:
        jobs.append({"agent_kind": "dqn", "seed": seed})  # longest jobs first
        for bins in config["tabular"]["bins"]:
            jobs.append({"agent_kind": "tabular", "seed": seed, "bins": bins, "ranges": "primary"})
        jobs.append({"agent_kind": "tabular", "seed": seed, "bins": config["tabular"]["range_ablation_bins"], "ranges": "tight"})
    return jobs


def _run(job: dict, config: dict, results_root: Path, tag: str | None) -> str:
    build_and_train(config=config, results_root=results_root, tag=tag, verbose=False, **job)
    return f"{run_name_for(job['agent_kind'], job.get('bins'), job.get('ranges'), tag)} seed {job['seed']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cartpole.yaml")
    parser.add_argument("--tuning", action="store_true", help="use tuning seeds and results/tuning")
    parser.add_argument("--only", choices=["tabular", "dqn"], help="restrict to one agent family")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--tag")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    seeds = config["tuning_seeds"] if args.tuning else config["seeds"]
    results_root = ROOT / "results" / "tuning" if args.tuning else ROOT / "results"

    jobs = [j for j in job_list(config, seeds) if args.only in (None, j["agent_kind"])]
    pending = []
    for j in jobs:
        name = run_name_for(j["agent_kind"], j.get("bins"), j.get("ranges"), args.tag)
        sub = "tabular" if j["agent_kind"] == "tabular" else "dqn"
        if not (results_root / sub / "logs" / name / f"seed_{j['seed']}" / "summary.json").exists():
            pending.append(j)
    print(f"{len(pending)} of {len(jobs)} runs pending", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, j, config, results_root, args.tag): j for j in pending}
        for fut in as_completed(futures):
            try:
                print("done:", fut.result(), flush=True)
            except Exception as exc:  # report and continue with the remaining runs
                print("FAILED:", futures[fut], repr(exc), flush=True)


if __name__ == "__main__":
    main()
