"""Create a small GRPO rollout assignment with cached RoboTwin episode_info."""

import argparse
import json
import os
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "RoboTwin" and not os.environ.get("ALLOW_NON_ROBOTWIN_CONDA"):
    raise SystemExit(
        "GRPO assignment generation must run in the conda RoboTwin environment. "
        "Run `conda activate RoboTwin` first, or set ALLOW_NON_ROBOTWIN_CONDA=1 "
        "if your RoboTwin environment has another name."
    )

_ORIGINAL_CWD = Path.cwd()

from evaluation.robotwin.collect_seeds import collect_valid_seeds_for_task


def main():
    parser = argparse.ArgumentParser(description="Create a GRPO assignment JSON")
    parser.add_argument("--task", type=str, default="turn_switch")
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=10000)
    parser.add_argument("--max_attempts_ratio", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    records = collect_valid_seeds_for_task(
        task_name=args.task,
        task_config=args.task_config,
        test_num=args.num_groups,
        st_seed=args.start_seed,
        max_attempts_ratio=args.max_attempts_ratio,
        validation_replays=1,
    )
    assignment = [
        {
            "task": args.task,
            "seed": item["seed"],
            "episode_idx": idx,
            "episode_info": item["episode_info"],
        }
        for idx, item in enumerate(records)
    ]

    out = Path(args.output)
    if not out.is_absolute():
        out = _ORIGINAL_CWD / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(assignment, f, indent=2)
    print(f"Wrote {len(assignment)} GRPO assignment item(s) to {out}")


if __name__ == "__main__":
    main()
