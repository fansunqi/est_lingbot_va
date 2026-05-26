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

robowin_root = Path("/root/WAM/RoboTwin")
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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def seed_entry_has_episode_info(entry):
    return isinstance(entry, dict) and "seed" in entry and "episode_info" in entry


def count_seed_entries(task_entries):
    return len(task_entries) if isinstance(task_entries, list) else 0


def task_entries_have_episode_info(task_entries):
    return (
        isinstance(task_entries, list)
        and all(seed_entry_has_episode_info(entry) for entry in task_entries)
    )


def class_decorator(task_name):
    """Create a task environment instance by importing the task module."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit(f"Failed to create env for task: {task_name}")
    return env_instance


def close_env_safely(task_env):
    try:
        task_env.close_env()
    except Exception:
        pass


def progress_write(message):
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def write_collect_progress(progress_path, task_name, test_num, valid_count, attempts,
                           max_attempts, current_seed, status, skipped):
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "task": task_name,
        "total": test_num,
        "valid": valid_count,
        "attempts": attempts,
        "max_attempts": max_attempts,
        "current_seed": current_seed,
        "status": status,
        "skipped": skipped,
    }
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    tmp_path.replace(progress_path)


def validate_seed_replays(task_name, args, seed, replay_count):
    """Rebuild the env for the same seed to catch non-reproducible unstable setups."""
    for replay_idx in range(1, replay_count):
        task_env = class_decorator(task_name)
        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        except UnStableError:
            close_env_safely(task_env)
            progress_write(f"  Seed {seed} failed replay validation ({replay_idx + 1}/{replay_count}); skip")
            return False
        except (IndexError, ValueError, RuntimeError) as e:
            close_env_safely(task_env)
            progress_write(f"  Seed {seed} failed replay validation ({replay_idx + 1}/{replay_count}): {e}; skip")
            return False
        except Exception as e:
            close_env_safely(task_env)
            progress_write(f"  Seed {seed} unexpected replay validation error ({replay_idx + 1}/{replay_count}): {e}; skip")
            traceback.print_exc()
            return False
        close_env_safely(task_env)
    return True


def update_seed_progress(progress_bar, now_seed, attempts, max_attempts, skipped):
    if progress_bar is None:
        return
    progress_bar.set_postfix_str(
        f"attempts={attempts}/{max_attempts}"
    )


def collect_valid_seeds_for_task(task_name, task_config, test_num, st_seed, max_attempts_ratio=3,
                                 validation_replays=1, progress_path=None):
    """
    Collect valid seeds for a single task by running expert_check.

    Args:
        task_name: Name of the task
        task_config: Task config name (e.g., 'demo_clean')
        test_num: Number of valid seeds to collect
        st_seed: Starting seed
        max_attempts_ratio: Max attempts = test_num * ratio (to avoid infinite loop)

    Returns:
        List of valid seed records. Each record contains the seed and the
        cached episode_info needed to build eval-time language instructions.
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

    valid_seed_records = []
    now_seed = st_seed
    max_attempts = test_num * max_attempts_ratio
    attempts = 0
    skipped = {
        "unstable": 0,
        "expert": 0,
        "known": 0,
        "replay": 0,
        "unexpected": 0,
    }

    progress_write(f"\n\033[34mCollecting seeds for: {task_name} (need {test_num})\033[0m")
    if validation_replays > 1:
        progress_write(f"  Replay validation: {validation_replays} setup(s) per accepted seed")
    write_collect_progress(
        progress_path, task_name, test_num, len(valid_seed_records),
        attempts, max_attempts, now_seed, "running", skipped
    )

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=test_num,
            desc=f"{task_name}",
            unit="seed",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        while len(valid_seed_records) < test_num and attempts < max_attempts:
            attempts += 1
            try:
                TASK_ENV.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                close_env_safely(TASK_ENV)

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    if not validate_seed_replays(task_name, args, now_seed, validation_replays):
                        skipped["replay"] += 1
                        update_seed_progress(progress_bar, now_seed, attempts, max_attempts, skipped)
                        write_collect_progress(
                            progress_path, task_name, test_num, len(valid_seed_records),
                            attempts, max_attempts, now_seed, "running", skipped
                        )
                        now_seed += 1
                        continue
                    valid_seed_records.append({
                        "seed": now_seed,
                        "episode_info": episode_info["info"],
                    })
                    if progress_bar is not None:
                        progress_bar.update(1)
                    else:
                        print(f"  [{len(valid_seed_records)}/{test_num}] seed={now_seed} ✓", end="\r")
                else:
                    skipped["expert"] += 1

            except UnStableError:
                skipped["unstable"] += 1
                close_env_safely(TASK_ENV)
            except (IndexError, ValueError, RuntimeError) as e:
                # Known env issues: some seeds produce invalid expert plans
                skipped["known"] += 1
                close_env_safely(TASK_ENV)
            except Exception as e:
                skipped["unexpected"] += 1
                close_env_safely(TASK_ENV)
                progress_write(f"  Unexpected error at seed {now_seed}: {e}")
                traceback.print_exc()

            update_seed_progress(progress_bar, now_seed, attempts, max_attempts, skipped)
            write_collect_progress(
                progress_path, task_name, test_num, len(valid_seed_records),
                attempts, max_attempts, now_seed, "running", skipped
            )
            now_seed += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(f"\n  \033[32mCollected {len(valid_seed_records)} valid seeds "
          f"(tried {attempts} seeds, fail rate: {1 - len(valid_seed_records)/attempts:.1%})\033[0m")
    print("  Skipped seeds: "
          f"unstable={skipped['unstable']}, expert={skipped['expert']}, "
          f"known={skipped['known']}, replay={skipped['replay']}, "
          f"unexpected={skipped['unexpected']}")
    final_status = "done" if len(valid_seed_records) >= test_num else "incomplete"
    write_collect_progress(
        progress_path, task_name, test_num, len(valid_seed_records),
        attempts, max_attempts, now_seed - 1, final_status, skipped
    )

    if len(valid_seed_records) < test_num:
        print(f"  \033[33mWarning: only got {len(valid_seed_records)}/{test_num} valid seeds "
              f"after {max_attempts} attempts\033[0m")

    return valid_seed_records


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
    parser.add_argument("--validation_replays", type=int, default=1,
                        help="Number of env setup replays required before accepting a seed")
    parser.add_argument("--progress_dir", type=str, default=None,
                        help="Optional directory for per-task JSON progress files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already collected tasks)")
    # Parallel worker args
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker index for parallel collection (0-indexed)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of parallel workers")
    args = parser.parse_args()

    if args.tasks:
        tasks = args.tasks.split()
    else:
        tasks = ALL_TASKS

    # Shard tasks across workers
    if args.num_workers > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.num_workers == args.worker_id]
        print(f"\033[36mWorker {args.worker_id}/{args.num_workers}: "
              f"handling {len(tasks)} tasks\033[0m")

    st_seed = 10000 * (1 + args.seed)

    # Resolve output path to absolute BEFORE os.chdir in Sapien_TEST/class_decorator
    # (the script does os.chdir(robowin_root) at import time)
    output_path = Path(args.output).resolve()

    # Initialize renderer
    Sapien_TEST()

    # Load existing results if resuming
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        print(f"Resuming: loaded {len(existing)} tasks from {output_path}")

    # When using multiple workers, each writes to its own shard file
    if args.num_workers > 1:
        shard_path = output_path.parent / f"{output_path.stem}_worker{args.worker_id}{output_path.suffix}"
        results = {}
    else:
        shard_path = output_path
        results = dict(existing)

    progress_dir = Path(args.progress_dir).resolve() if args.progress_dir else None

    for task_name in tasks:
        existing_entries = existing.get(task_name)
        if (count_seed_entries(existing_entries) >= args.test_num
                and task_entries_have_episode_info(existing_entries)):
            print(f"Skipping {task_name} (already have {len(existing_entries)} cached seeds)")
            if progress_dir is not None:
                write_collect_progress(
                    progress_dir / f"worker_{args.worker_id}_{task_name}.json",
                    task_name,
                    args.test_num,
                    count_seed_entries(existing_entries),
                    0,
                    args.test_num * args.max_attempts_ratio,
                    st_seed,
                    "cached",
                    {"unstable": 0, "expert": 0, "known": 0, "replay": 0, "unexpected": 0},
                )
            results[task_name] = existing_entries
            continue
        if task_name in existing:
            print(f"Re-collecting {task_name}: cached seeds are missing episode_info")

        valid_seed_records = collect_valid_seeds_for_task(
            task_name=task_name,
            task_config=args.task_config,
            test_num=args.test_num,
            st_seed=st_seed,
            max_attempts_ratio=args.max_attempts_ratio,
            validation_replays=args.validation_replays,
            progress_path=(progress_dir / f"worker_{args.worker_id}_{task_name}.json"
                           if progress_dir is not None else None),
        )
        results[task_name] = valid_seed_records

        # Save incrementally (in case of crash)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\033[32mDone! Saved valid seeds for {len(results)} tasks to {shard_path}\033[0m")
    total_seeds = sum(count_seed_entries(v) for v in results.values())
    print(f"Total valid seeds: {total_seeds}")


def merge_shards():
    """Merge worker shard files into a single valid_seeds.json, then delete shards."""
    parser = argparse.ArgumentParser(description="Merge seed collection shards")
    parser.add_argument("--output", type=str, required=True,
                        help="Final merged output file path")
    parser.add_argument("--shards_dir", type=str, default=None,
                        help="Directory containing shard files (default: same dir as output)")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    shards_dir = Path(args.shards_dir).resolve() if args.shards_dir else output_path.parent

    # Find all shard files
    pattern = f"{output_path.stem}_worker*{output_path.suffix}"
    shard_files = sorted(shards_dir.glob(pattern))

    if not shard_files:
        print(f"No shard files matching '{pattern}' in {shards_dir}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files:")
    merged = {}
    for sf in shard_files:
        print(f"  {sf}")
        with open(sf, "r") as f:
            data = json.load(f)
        merged.update(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Delete shard files after successful merge
    for sf in shard_files:
        sf.unlink()
        print(f"  Deleted: {sf}")

    print(f"\n\033[32mMerged {len(merged)} tasks into {output_path}\033[0m")
    total_seeds = sum(count_seed_entries(v) for v in merged.values())
    print(f"Total valid seeds: {total_seeds}")


if __name__ == "__main__":
    # Support subcommand: python -m evaluation.robotwin.collect_seeds merge --output ...
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        sys.argv.pop(1)  # remove 'merge' so argparse works
        merge_shards()
    else:
        main()
