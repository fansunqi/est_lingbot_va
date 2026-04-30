"""
Collect valid seeds for RoboTwin evaluation tasks.

Runs expert_check (play_once) for each task to find seeds where the expert
can successfully complete the task. These seeds are then used for policy
evaluation, eliminating the need for runtime expert checking.

Usage:
    python -m evaluation.robotwin.collect_seeds \
        --tasks "lift_pot place_shoe blocks_ranking_rgb" \
        --test_num 100 --seed 0 \
        --task_config demo_clean \
        --output valid_seeds.json

    # Collect for all 50 tasks:
    python -m evaluation.robotwin.collect_seeds \
        --test_num 100 --seed 0 \
        --task_config demo_clean \
        --output valid_seeds.json
"""

import sys
import os
from pathlib import Path

robowin_root = Path("/home/cxy/WAM/RoboTwin")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import argparse
import importlib
import json
import traceback

import yaml

from evaluation.robotwin.test_render import Sapien_TEST
from evaluation.robotwin.balance_tasks import ALL_TASKS


def class_decorator(task_name):
    """Create a task environment instance by importing the task module."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit(f"Failed to create env for task: {task_name}")
    return env_instance


def collect_valid_seeds_for_task(task_name, task_config, test_num, st_seed, max_attempts_ratio=3):
    """
    Collect valid seeds for a single task by running expert_check.

    Args:
        task_name: Name of the task
        task_config: Task config name (e.g., 'demo_clean')
        test_num: Number of valid seeds to collect
        st_seed: Starting seed
        max_attempts_ratio: Max attempts = test_num * ratio (to avoid infinite loop)

    Returns:
        List of valid seeds
    """
    # Load task config
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False

    # Setup embodiment
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = _embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Camera config
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # Create environment
    TASK_ENV = class_decorator(task_name)

    valid_seeds = []
    now_seed = st_seed
    max_attempts = test_num * max_attempts_ratio
    attempts = 0

    print(f"\n\033[34mCollecting seeds for: {task_name} (need {test_num})\033[0m")

    while len(valid_seeds) < test_num and attempts < max_attempts:
        attempts += 1
        try:
            TASK_ENV.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()

            if TASK_ENV.plan_success and TASK_ENV.check_success():
                valid_seeds.append(now_seed)
                print(f"  [{len(valid_seeds)}/{test_num}] seed={now_seed} ✓", end="\r")
            else:
                pass  # expert failed, skip this seed

        except UnStableError:
            TASK_ENV.close_env()
        except Exception as e:
            TASK_ENV.close_env()
            print(f"  Error at seed {now_seed}: {e}")
            traceback.print_exc()

        now_seed += 1

    print(f"\n  \033[32mCollected {len(valid_seeds)} valid seeds "
          f"(tried {attempts} seeds, fail rate: {1 - len(valid_seeds)/attempts:.1%})\033[0m")

    if len(valid_seeds) < test_num:
        print(f"  \033[33mWarning: only got {len(valid_seeds)}/{test_num} valid seeds "
              f"after {max_attempts} attempts\033[0m")

    return valid_seeds


def main():
    parser = argparse.ArgumentParser(description="Collect valid seeds for RoboTwin evaluation")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Space-separated task names (default: all 50 tasks)")
    parser.add_argument("--test_num", type=int, default=100,
                        help="Number of valid seeds to collect per task")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed multiplier (st_seed = 10000 * (1 + seed))")
    parser.add_argument("--task_config", type=str, default="demo_clean",
                        help="Task config file name")
    parser.add_argument("--output", type=str, default="valid_seeds.json",
                        help="Output JSON file path")
    parser.add_argument("--max_attempts_ratio", type=int, default=3,
                        help="Max attempts per task = test_num * ratio")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already collected tasks)")
    args = parser.parse_args()

    if args.tasks:
        tasks = args.tasks.split()
    else:
        tasks = ALL_TASKS

    st_seed = 10000 * (1 + args.seed)

    # Initialize renderer
    Sapien_TEST()

    # Load existing results if resuming
    existing = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = json.load(f)
        print(f"Resuming: loaded {len(existing)} tasks from {args.output}")

    results = dict(existing)

    for task_name in tasks:
        if task_name in results and len(results[task_name]) >= args.test_num:
            print(f"Skipping {task_name} (already have {len(results[task_name])} seeds)")
            continue

        valid_seeds = collect_valid_seeds_for_task(
            task_name=task_name,
            task_config=args.task_config,
            test_num=args.test_num,
            st_seed=st_seed,
            max_attempts_ratio=args.max_attempts_ratio,
        )
        results[task_name] = valid_seeds

        # Save incrementally (in case of crash)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\033[32mDone! Saved valid seeds for {len(results)} tasks to {args.output}\033[0m")
    total_seeds = sum(len(v) for v in results.values())
    print(f"Total valid seeds: {total_seeds}")


if __name__ == "__main__":
    main()
